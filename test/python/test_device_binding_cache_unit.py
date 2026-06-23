# test_device_binding_cache_unit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.

# RUN: %pytest %s
"""Unit tests for target-device-sensitive generation and cache identity."""

from pathlib import Path

import pytest

from aie.iron.device import NPU1Col1, NPU2Col1, NPU2Col2
from aie.utils import set_current_device
from aie.utils.compile.jit import CompileTime, In, Out
from aie.utils.compile.jit.compilabledesign import CompilableDesign


def _gemm_gen():
    def gemm(
        a: In,
        b: In,
        c: Out,
        *,
        M: CompileTime[int],
        K: CompileTime[int],
        N: CompileTime[int],
    ):
        pass

    return gemm


def test_generated_cache_tracks_active_device(monkeypatch):
    """MLIR generation cache entries are keyed by the active IRON device."""

    def gen():
        pass

    cd = CompilableDesign(gen)
    calls = []

    def fake_generate_uncached(self):
        calls.append(self._generation_cache_key())
        return (f"mlir-{len(calls)}", [])

    monkeypatch.setattr(CompilableDesign, "_generate_uncached", fake_generate_uncached)

    set_current_device(NPU1Col1())
    first_aie2 = cd._generated
    second_aie2 = cd._generated

    set_current_device(NPU2Col1())
    first_aie2p = cd._generated
    second_aie2p = cd._generated

    set_current_device(None)

    assert first_aie2 == second_aie2
    assert first_aie2p == second_aie2p
    assert first_aie2 != first_aie2p
    assert len(calls) == 2


def test_uncached_generation_clears_context_bound_external_functions():
    """Fresh MLIR generation never reuses factory objects from another context."""
    from aie.iron.kernels import _common as kernel_common

    def gen():
        pass

    kernel_common._EXTERN_CACHE["stale"] = object()
    CompilableDesign(gen)._generate_uncached()

    assert not kernel_common._EXTERN_CACHE


def test_generated_root_binds_runtime_device_before_cache_key(monkeypatch):
    """Direct MLIR generation binds the runtime device before cache lookup."""
    import aie.utils as utils

    class FakeRuntime:
        def device(self):
            return NPU2Col1()

    def gen():
        pass

    set_current_device(None)
    monkeypatch.setattr(utils, "_get_default_npu_runtime", lambda: FakeRuntime())

    cd = CompilableDesign(gen)

    def fake_generate_uncached(self):
        assert (
            type(utils.get_current_device(probe_runtime=False)).__name__ == "NPU2Col1"
        )
        return ("mlir", [])

    monkeypatch.setattr(CompilableDesign, "_generate_uncached", fake_generate_uncached)

    try:
        assert cd._generated == ("mlir", [])
    finally:
        set_current_device(None)

    keys = list(cd._generated_cache)
    assert len(keys) == 1
    assert "NPU2Col1" in keys[0][1]


def test_cache_hit_refreshes_tensor_metadata_for_the_selected_artifact(
    monkeypatch, tmp_path
):
    """Returning to a cached target restores that artifact's validation metadata."""
    import aie.utils as utils
    import aie.utils.compile.jit.compilabledesign as compilabledesign_module

    def gen():
        pass

    artifacts = {
        "npu1": [16],
        "npu2": [32],
    }
    for name in artifacts:
        kernel_dir = tmp_path / name
        kernel_dir.mkdir()
        (kernel_dir / "final.xclbin").touch()
        (kernel_dir / "insts.bin").touch()

    cd = CompilableDesign(gen)
    hashes = iter(("npu1", "npu2", "npu1"))
    monkeypatch.setattr(cd, "_compute_cache_hash", lambda: next(hashes))
    monkeypatch.setattr(compilabledesign_module, "NPU_CACHE_HOME", tmp_path)
    monkeypatch.setattr(
        compilabledesign_module,
        "parse_dma_sizes",
        lambda kernel_dir: artifacts[kernel_dir.name],
    )
    monkeypatch.setattr(utils, "ensure_current_device", lambda: None)

    cd.compile()
    assert cd._expected_tensor_sizes == [16]

    cd.compile()
    assert cd._expected_tensor_sizes == [32]

    cd.compile()
    assert cd._expected_tensor_sizes == [16]


def test_compute_hash_changes_when_active_device_width_changes():
    """The filesystem cache key includes the active device identity, not just arch."""
    gen = _gemm_gen()
    cd = CompilableDesign(gen, compile_kwargs={"M": 64, "K": 64, "N": 64})

    set_current_device(NPU2Col1())
    h_one_col = cd._compute_cache_hash()

    set_current_device(NPU2Col2())
    h_two_col = cd._compute_cache_hash()

    set_current_device(None)

    assert h_one_col != h_two_col


def test_python_compile_binds_before_cache_lookup(monkeypatch, tmp_path):
    """Generated designs bind the target before selecting a cache entry."""
    import aie.utils as utils
    import aie.utils.compile.jit.compilabledesign as compilabledesign_module

    def gen():
        pass

    cd = CompilableDesign(gen)
    calls = []

    def fake_ensure_current_device():
        calls.append("bind")
        return NPU2Col1()

    def fake_cache_hash():
        assert calls == ["bind"]
        calls.append("hash")
        return "target"

    cache_dir = tmp_path / "target"
    cache_dir.mkdir()
    (cache_dir / "final.xclbin").touch()
    (cache_dir / "insts.bin").touch()

    monkeypatch.setattr(utils, "ensure_current_device", fake_ensure_current_device)
    monkeypatch.setattr(cd, "_compute_cache_hash", fake_cache_hash)
    monkeypatch.setattr(compilabledesign_module, "NPU_CACHE_HOME", tmp_path)
    monkeypatch.setattr(compilabledesign_module, "parse_dma_sizes", lambda _dir: None)

    cd.compile()
    assert calls == ["bind", "hash"]


def test_static_mlir_compile_does_not_bind_runtime(monkeypatch, tmp_path):
    """Static MLIR compilation does not initialize or bind the runtime."""
    import aie.utils as utils
    import aie.utils.compile as compile_utils
    import aie.utils.compile.jit.compilabledesign as compilabledesign_module

    mlir_path = tmp_path / "design.mlir"
    mlir_path.write_text("module {}")
    cd = CompilableDesign(mlir_path, use_cache=False)
    set_current_device(None)

    monkeypatch.setattr(
        utils,
        "ensure_current_device",
        lambda: pytest.fail("static MLIR compilation must not bind the runtime"),
    )
    monkeypatch.setattr(
        utils,
        "_get_default_npu_runtime",
        lambda: pytest.fail("static MLIR compilation must not initialize XRT"),
    )
    monkeypatch.setattr(cd, "_generate_mlir", lambda _external_function: object())
    monkeypatch.setattr(compile_utils, "resolve_target_arch", lambda _device: "aie2")

    def fake_compile_mlir_module(**kwargs):
        Path(kwargs["xclbin_path"]).touch()
        Path(kwargs["insts_path"]).touch()

    monkeypatch.setattr(
        compilabledesign_module, "compile_mlir_module", fake_compile_mlir_module
    )

    xclbin_path = tmp_path / "out.xclbin"
    inst_path = tmp_path / "out.insts"
    try:
        assert cd.compile(xclbin_path=xclbin_path, inst_path=inst_path) == (
            xclbin_path.resolve(),
            inst_path.resolve(),
        )
    finally:
        set_current_device(None)


def test_cache_hash_does_not_probe_runtime(monkeypatch):
    """Inspecting an unbound design cache key never initializes XRT."""
    import aie.utils as utils

    cd = CompilableDesign(_gemm_gen(), compile_kwargs={"M": 64, "K": 64, "N": 64})
    set_current_device(None)
    monkeypatch.setattr(
        utils,
        "_get_default_npu_runtime",
        lambda: pytest.fail("cache-key inspection must not initialize XRT"),
    )

    try:
        assert cd._compute_cache_hash()
    finally:
        set_current_device(None)


def test_iron_reexports_set_current_device():
    """The documented IRON device-selection entry point is public."""
    import aie.iron as iron

    assert iron.set_current_device is set_current_device
