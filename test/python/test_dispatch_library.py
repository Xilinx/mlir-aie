# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Host-only rebuild tests: real compiler, ABI probe, manifest and loader."""

import ctypes
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aie.utils.compile.jit import _dispatch_compile, _manifest
from aie.utils.compile.jit import compilabledesign as design_module
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.compile.jit._dispatch_compile import DispatchCompileError
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In
from aie.utils.compile.utils import _cleanup_failed_compilation
from test_dispatch_bridge import _EXPORT_MACRO, _FIXTURE_ABI, _FIXTURE_BODY


@pytest.mark.parametrize("compile_kwargs", [{}, {"bound": 8}])
def test_defaulted_compile_param_does_not_consume_dispatch_argument(compile_kwargs):
    def generator(
        a: In,
        bound: CompileTime[int] = 4,
        scale: DispatchTime[np.int32] = 1,  # pyright: ignore[reportArgumentType]
    ):
        pass

    design = CompilableDesign(generator, compile_kwargs=compile_kwargs)
    tensor = object()
    tensors, scalars = design.split_runtime_args((tensor, 7), {})
    assert tensors == [tensor]
    assert scalars == {"scale": 7}


@pytest.fixture
def dispatch_design(tmp_path, monkeypatch, npu2_device):
    """Mock only NPU compilation/translation; build and load real host libraries."""

    def generator(scale: DispatchTime[np.int32], n_tiles: DispatchTime[np.uintp]):
        pass

    state = {"offset": 0, "abi": "int32_t,size_t", "fail": False, "builds": 0}
    includes = tmp_path / "include"
    header = includes / "aie/Runtime/TxnEncoding.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define DISPATCH_FIXTURE_OFFSET 0\n")
    monkeypatch.setattr(
        _dispatch_compile.config, "runtime_header_path", lambda: str(includes)
    )
    design = CompilableDesign(generator, use_cache=False)
    monkeypatch.setattr(design_module, "NPU_CACHE_HOME", tmp_path)
    monkeypatch.setattr(design, "_compute_cache_hash", lambda: "design")
    monkeypatch.setattr(design, "_generate_mlir", lambda *args: None)
    monkeypatch.setattr(design, "_resolve_fold_ddr_addr_offset", lambda: False)
    monkeypatch.setattr(design_module, "compile_external_kernels", lambda *a, **k: None)
    monkeypatch.setattr(design_module, "parse_dma_sizes", lambda *args: [])

    def compile_mlir(**kwargs):
        state["builds"] += 1
        Path(kwargs["xclbin_path"]).touch()
        if state["fail"]:
            raise RuntimeError("NPU build failed")

    def translate(_lowered, kernel_dir, _fold):
        source = kernel_dir / "dispatch_gen.cpp"
        body = _FIXTURE_BODY.replace(
            "static_cast<uint32_t>(scale) + i",
            f"static_cast<uint32_t>(scale) + i + {state['offset']} + DISPATCH_FIXTURE_OFFSET",
        )
        source.write_text(
            '#include "aie/Runtime/TxnEncoding.h"\n'
            + _EXPORT_MACRO
            + _FIXTURE_ABI.replace("int32_t,size_t", state["abi"])
            + body
        )
        return source

    monkeypatch.setattr(design_module, "compile_mlir_module", compile_mlir)
    monkeypatch.setattr(
        _dispatch_compile, "_lower_dynamic_runtime_sequence", lambda path: path
    )
    monkeypatch.setattr(_dispatch_compile, "_translate_to_cpp", translate)
    return design, state


def _load(design):
    path = design.get_dispatch_lib_path()
    assert path is not None
    return path, DispatchBridge(path, ["scale", "n_tiles"])


def _words(bridge):
    return list(bridge.generate({"scale": 10, "n_tiles": 2}))


def test_rebuild_keeps_old_and_new_generations_executable(dispatch_design):
    design, state = dispatch_design
    design.compile()
    old_path, old_bridge = _load(design)
    state["offset"] = 100
    design.compile()
    new_path, new_bridge = _load(design)
    assert old_path != new_path
    assert old_path.name == f"dispatch-{_manifest._digest(old_path)}{old_path.suffix}"
    assert _words(old_bridge) == [10, 11]
    assert _words(new_bridge) == [110, 111]
    assert old_path.is_file()
    assert not list(new_path.parent.glob("dispatch.staging.*"))
    design.use_cache = True
    design.compile()
    assert state["builds"] == 2
    assert design.get_dispatch_lib_path() == new_path


def test_runtime_header_change_invalidates_dispatch_cache(dispatch_design):
    design, state = dispatch_design
    design.use_cache = True
    design.compile()
    old_path, old_bridge = _load(design)
    header = (
        Path(_dispatch_compile.config.runtime_header_path())
        / "aie/Runtime/TxnEncoding.h"
    )
    payload = json.loads((old_path.parent / _manifest.MANIFEST_NAME).read_text())
    assert str(header) in [entry["path"] for entry in payload["inputs"]]
    design.compile()
    assert state["builds"] == 1
    header.write_text("#define DISPATCH_FIXTURE_OFFSET 100\n")
    assert not _manifest.is_valid(old_path.parent)
    design.compile()
    new_path, new_bridge = _load(design)
    assert state["builds"] == 2
    assert new_path != old_path
    assert _words(old_bridge) == [10, 11]
    assert _words(new_bridge) == [110, 111]


def test_identical_rebuild_does_not_replace_mapped_generation(
    dispatch_design, monkeypatch
):
    design, _ = dispatch_design
    design.compile()
    path, bridge = _load(design)
    replace = Path.replace

    def guarded_replace(source, target):
        assert target != path, "must not replace an already published DLL"
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", guarded_replace)
    design.compile()
    assert design.get_dispatch_lib_path() == path
    assert _words(bridge) == [10, 11]


@pytest.mark.parametrize("failure", ["abi", "build"])
def test_failed_rebuild_preserves_loaded_generation(dispatch_design, failure):
    design, state = dispatch_design
    design.compile()
    path, bridge = _load(design)
    if failure == "abi":
        state["abi"] = "int64_t,int32_t"
    else:
        state["fail"] = True
    with pytest.raises(RuntimeError):
        design.compile()
    assert path.is_file()
    assert _words(bridge) == [10, 11]
    assert design.get_dispatch_lib_path() is None
    assert not list(path.parent.glob("dispatch.staging.*"))
    state.update(abi="int32_t,size_t", fail=False, offset=20)
    design.compile()
    _, rebuilt = _load(design)
    assert _words(rebuilt) == [30, 31]


def test_abi_probe_never_loads_library_in_compiler(dispatch_design, monkeypatch):
    design, _ = dispatch_design

    def forbidden(*args, **kwargs):
        pytest.fail("compiler must not load the library")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    design.compile()
    assert design.get_dispatch_lib_path() is not None


@pytest.mark.parametrize(
    "manifest",
    [
        None,
        "{",
        "[]",
        "{}",
        '{"version": 1, "dispatch_library": null}',
        '{"version": 1, "dispatch_library": "dispatch.so"}',
        json.dumps(
            {"version": 1, "dispatch_library": "../dispatch-" + "a" * 64 + ".so"}
        ),
        json.dumps({"version": 1, "dispatch_library": "dispatch-" + "a" * 64 + ".so"}),
    ],
)
def test_invalid_manifest_rebuilds_dispatch_companion(dispatch_design, manifest):
    design, state = dispatch_design
    design.compile()
    path, bridge = _load(design)
    manifest_path = path.parent / _manifest.MANIFEST_NAME
    if manifest is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(manifest)
    assert design.get_dispatch_lib_path() is None
    design.use_cache = True
    state["offset"] = 20
    design.compile()
    assert state["builds"] == 2
    _, rebuilt = _load(design)
    assert _words(bridge) == [10, 11]
    assert _words(rebuilt) == [30, 31]


def test_cleanup_preserves_mapped_and_not_yet_loaded_generations(
    dispatch_design, monkeypatch
):
    design, state = dispatch_design
    design.compile()
    old_path = design.get_dispatch_lib_path()
    state["offset"] = 20
    design.compile()
    path, bridge = _load(design)
    remove = os.remove

    def guarded_remove(name):
        assert Path(name) not in (path, old_path), "mapped DLL deletion attempted"
        return remove(name)

    monkeypatch.setattr(os, "remove", guarded_remove)
    _cleanup_failed_compilation(path.parent)
    assert old_path is not None
    assert _words(DispatchBridge(old_path, ["scale", "n_tiles"])) == [10, 11]
    assert _words(bridge) == [30, 31]
    assert not (path.parent / _manifest.MANIFEST_NAME).exists()


def test_cleanup_does_not_swallow_io_errors(tmp_path, monkeypatch):
    (tmp_path / "partial.o").touch()

    def fail_remove(name):
        raise PermissionError("unrelated permissions failure")

    monkeypatch.setattr(os, "remove", fail_remove)
    with pytest.raises(PermissionError, match="unrelated permissions failure"):
        _cleanup_failed_compilation(tmp_path)


@pytest.mark.parametrize("incomplete", ["chess", "no_depfile", "depfile", "digest"])
def test_incomplete_manifest_keeps_dispatch_output(tmp_path, monkeypatch, incomplete):
    name = "dispatch-" + "a" * 64 + ".so"
    (tmp_path / name).touch()
    source = tmp_path / "source.cc"
    source.touch()

    def unreadable(*args):
        raise OSError("unreadable")

    if incomplete == "depfile":
        (tmp_path / "source.d").touch()
        monkeypatch.setattr(_manifest, "_parse_depfile", unreadable)
    elif incomplete == "digest":
        monkeypatch.setattr(_manifest, "_entry", unreadable)
    _manifest.record(
        tmp_path,
        [SimpleNamespace(_source_file=source)] if incomplete == "no_depfile" else [],
        [source],
        used_chess=incomplete == "chess",
        dispatch_library=name,
    )
    payload = json.loads((tmp_path / _manifest.MANIFEST_NAME).read_text())
    assert payload["complete"] is False
    assert _manifest.resolve_dispatch_library(tmp_path) == tmp_path / name
    assert _manifest.is_valid(tmp_path)


def test_malformed_abi_probe_failure_is_actionable(dispatch_design):
    design, state = dispatch_design
    state["abi"] = "int32_t,,int64_t"
    with pytest.raises(DispatchCompileError, match="malformed dispatch ABI"):
        design.compile()


def test_partial_host_library_is_removed_after_compiler_failure(
    dispatch_design, monkeypatch
):
    design, _ = dispatch_design

    def failed_compile(source, output):
        output.write_text("partial library")
        raise DispatchCompileError("host compiler failed")

    monkeypatch.setattr(_dispatch_compile, "_compile_so", failed_compile)
    with pytest.raises(DispatchCompileError, match="host compiler failed"):
        design.compile()
    assert not list(design_module.NPU_CACHE_HOME.rglob("dispatch.staging.*"))


def test_failed_manifest_publication_preserves_loaded_generation(
    dispatch_design, monkeypatch
):
    design, state = dispatch_design
    design.compile()
    path, bridge = _load(design)
    state["offset"] = 20
    replace = os.replace

    def failed_replace(source, target):
        if Path(target).name == _manifest.MANIFEST_NAME:
            raise OSError("manifest publication failed")
        return replace(source, target)

    monkeypatch.setattr(_manifest.os, "replace", failed_replace)
    with pytest.raises(OSError, match="manifest publication failed"):
        design.compile()
    assert _words(bridge) == [10, 11]
    assert path.is_file()
    assert design.get_dispatch_lib_path() is None
