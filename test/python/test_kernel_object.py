# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Artifact ownership and concurrent compilation; no compiler or NPU required."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
import gc
import threading
import weakref

import pytest
from aie.dialects.aie import AIEDevice, device
from aie.extras.context import mlir_mod_ctx
from aie.iron.kernel import ExternalFunction, Kernel, KernelObject
from aie.utils.compile import utils


def _function(name, **kwargs):
    return ExternalFunction(
        name, source_string="void reduce_max() {} void compute_max() {}", **kwargs
    )


def _resolve_in_device(kernel):
    with mlir_mod_ctx():
        device(AIEDevice.npu2_1col)(lambda: kernel.resolve())


def test_exported_symbols_share_recipe_and_owner():
    first = _function("reduce_max", object_file_name="reduce_max.cc.o")
    second = _function("compute_max", object_file_name="reduce_max.cc.o")
    assert first.object_file is second.object_file
    assert first != second
    compute = first.object_file.bind("compute_max", [])
    assert compute.object_file is first.object_file


@pytest.mark.parametrize("inline", [False, True])
def test_sibling_alone_rediscovers_source_owner_after_registry_reset(
    tmp_path, monkeypatch, inline
):
    original = _function("reduce_max", inline=inline)
    sibling = original.object_file.bind("compute_max", [])
    original_ref = weakref.ref(original)
    ExternalFunction._instances.clear()
    del original
    gc.collect()
    assert original_ref() is None

    calls = []

    def compile_stub(output_path, **kwargs):
        calls.append(kwargs)
        Path(output_path).write_text("complete")

    monkeypatch.setattr(utils, "compile_cxx_core_function", compile_stub)
    _resolve_in_device(sibling)
    discovered = list(ExternalFunction._instances)
    assert len(discovered) == 1
    assert discovered[0].object_file is sibling.object_file
    utils.compile_external_kernels(discovered, tmp_path, "aie2p")
    assert (tmp_path / sibling.object_file_name).exists()
    assert calls[0]["symbol_name"] == ("reduce_max" if inline else "compute_max")

    ExternalFunction._instances.clear()
    _resolve_in_device(sibling)
    assert len(ExternalFunction._instances) == 1


def test_reduce_max_specializations_do_not_share_objects(
    tmp_path, monkeypatch, npu2_device
):
    from aie.iron.kernels.reduce import compute_max, reduce_max

    first, second, third = (
        reduce_max(tile_size=16),
        reduce_max(tile_size=256),
        compute_max(),
    )
    assert len({fn.object_file_name for fn in (first, second, third)}) == 3
    assert first._symbol_prefix != second._symbol_prefix
    assert first._symbol_prefix and second._symbol_prefix
    calls = []

    def compile_stub(output_path, **kwargs):
        calls.append(output_path)
        Path(output_path).write_text("complete")

    monkeypatch.setattr(utils, "compile_cxx_core_function", compile_stub)
    monkeypatch.setattr(utils, "prefix_symbols_in_object", lambda *args: None)
    monkeypatch.setattr(
        utils, "_defined_symbols", lambda path: {first.name, second.name}
    )
    monkeypatch.setenv("AIE_KERNEL_COMPILE_JOBS", "2")
    utils.compile_external_kernels([first, second, third], tmp_path, "aie2p")
    assert len(calls) == 3


def test_prebuilt_object_can_be_shared():
    artifact = KernelObject("shared.bc", "merge")
    first, second = (Kernel(name, artifact) for name in ("first", "second"))
    assert first.object_file is second.object_file
    assert first.object_file_name == "shared.bc"
    assert first.link_with_mode == "merge"
    with pytest.raises(ValueError, match="link_with_mode conflicts"):
        Kernel("third", artifact, link_with_mode="other")
    with pytest.raises(ValueError, match="link_with_mode conflicts"):
        artifact.bind("third", link_with_mode="other")


def test_inline_sibling_retains_object_link_policy():
    first = _function("reduce_max", inline=True)
    compute = first.object_file.bind("compute_max", [])
    assert compute.link_with_mode == "merge"


@pytest.mark.parametrize(
    "difference",
    [
        {"compile_flags": ["-DOTHER"]},
        {"include_dirs": ["other"]},
        {"use_chess": True},
        {"symbol_prefix": "other"},
    ],
)
def test_different_symbols_cannot_hide_conflicting_recipes(difference):
    _function("reduce_max", object_file_name="shared.o")
    with pytest.raises(ValueError, match="would collide"):
        _function("compute_max", object_file_name="shared.o", **difference)


def test_recipe_copies_input_lists_and_preserves_flag_order():
    flags = ["-DX=1", "-UX"]
    includes = ["first", "second"]
    first = _function("reduce_max", compile_flags=flags, include_dirs=includes)
    flags.reverse()
    includes.reverse()
    second = _function("reduce_max", compile_flags=flags, include_dirs=includes)
    assert first.compile_flags == ["-DX=1", "-UX"]
    assert first.include_dirs == [
        str(Path("first").absolute()),
        str(Path("second").absolute()),
    ]
    assert first.object_file_name != second.object_file_name
    assert hash(first) != hash(second)


def test_distinct_output_names_remain_registered():
    first = _function("reduce_max", object_file_name="first.o")
    second = _function("reduce_max", object_file_name="second.o")
    assert first != second
    assert len(ExternalFunction._instances) == 2


def test_changed_source_file_does_not_reuse_previous_owner(tmp_path):
    source = tmp_path / "kernel.cc"
    source.write_text("void kernel() {}")
    first = ExternalFunction("kernel", source_file=str(source))
    source.write_text("void kernel() { volatile int x = 1; }")
    second = ExternalFunction("kernel", source_file=str(source))
    assert first.object_file is not second.object_file
    assert first.object_file_name != second.object_file_name


def test_aliasing_output_paths_reject_conflicting_recipes(tmp_path):
    first = _function("first", object_file_name="shared.o")
    second = _function("second", object_file_name="./shared.o", compile_flags=["-DX"])
    with pytest.raises(ValueError, match="Conflicting kernel compile recipes"):
        utils.compile_external_kernels([first, second], tmp_path, "aie2p")


@pytest.mark.parametrize("batch", [False, True])
def test_shared_object_compiles_once(tmp_path, monkeypatch, batch):
    first = _function("reduce_max", object_file_name="reduce_max.cc.o")
    second = _function("compute_max", object_file_name="reduce_max.cc.o")
    calls = []

    def compile_stub(output_path, **kwargs):
        calls.append(output_path)
        Path(output_path).write_text("complete")

    monkeypatch.setattr(utils, "compile_cxx_core_function", compile_stub)
    monkeypatch.setenv("AIE_KERNEL_COMPILE_JOBS", "2")
    if batch:
        utils.compile_external_kernels([first, second], tmp_path, "aie2p")
    else:
        with ThreadPoolExecutor(2) as pool:
            list(
                pool.map(
                    lambda f: utils.compile_external_kernel(f, tmp_path, "aie2p"),
                    [first, second],
                )
            )
    assert len(calls) == 1
    assert utils._compiled_into(first, tmp_path)
    assert utils._compiled_into(second, tmp_path)
    (tmp_path / first.object_file_name).unlink()
    utils.compile_external_kernel(second, tmp_path, "aie2p")
    assert len(calls) == 2


@pytest.mark.parametrize("shared_output", [False, True])
def test_direct_calls_lock_output_and_source_until_compile_finishes(
    tmp_path, monkeypatch, shared_output
):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_source, second_source = first_dir / "same.cc", second_dir / "same.cc"
    first_source.write_text("first")
    second_source.write_text("second")
    first = ExternalFunction(
        "first", source_file=str(first_source), object_file_name="first.o"
    )
    second = ExternalFunction(
        "second",
        source_file=str(first_source if shared_output else second_source),
        object_file_name="first.o" if shared_output else "second.o",
    )
    entered, release, attempted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    calls = []

    def compile_stub(source_path, output_path, **kwargs):
        calls.append(output_path)
        Path(output_path).write_text("incomplete")
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            assert Path(source_path).read_text() == "first"
        Path(output_path).write_text("complete")

    def compile_second():
        attempted.set()
        utils.compile_external_kernel(second, tmp_path, "aie2p")

    monkeypatch.setattr(utils, "compile_cxx_core_function", compile_stub)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(utils.compile_external_kernel, first, tmp_path, "aie2p")
        try:
            assert entered.wait(5)
            b = pool.submit(compile_second)
            assert attempted.wait(5)
            with pytest.raises(TimeoutError):
                b.result(timeout=0.1)
        finally:
            release.set()
        a.result()
        b.result()
    assert len(calls) == (1 if shared_output else 2)
    assert (tmp_path / first.object_file_name).read_text() == "complete"


def test_failed_compile_removes_partial_object_before_retry(tmp_path, monkeypatch):
    first = _function("reduce_max")

    def fail(output_path, **kwargs):
        Path(output_path).write_text("partial")
        Path(output_path + ".d").write_text("partial dependencies")
        raise RuntimeError("failed")

    monkeypatch.setattr(utils, "compile_cxx_core_function", fail)
    with pytest.raises(RuntimeError, match="failed"):
        utils.compile_external_kernel(first, tmp_path, "aie2p")
    assert not (tmp_path / first.object_file_name).exists()
    assert not (tmp_path / (first.object_file_name + ".d")).exists()
    assert not utils._compiled_into(first, tmp_path)

    monkeypatch.setattr(
        utils,
        "compile_cxx_core_function",
        lambda output_path, **kwargs: Path(output_path).write_text("complete"),
    )
    utils.compile_external_kernel(first, tmp_path, "aie2p")
    assert utils._compiled_into(first, tmp_path)
