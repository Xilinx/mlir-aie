# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Compiler-only integration tests using real MLIR and the host C++ compiler."""

from pathlib import Path
import time

import numpy as np
import pytest
from aie.ir import Context, Module
from aie.utils.compile import utils as compile_utils
from aie.utils.compile.jit import _manifest
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.compile.jit._dispatch_compile import (
    DispatchCompileError,
    _check_runtime_sequence_abi,
    compile_dispatch_bridge,
)
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In
from aie.utils.compile.utils import _run_aiecc


@pytest.mark.parametrize("emit_shim", [False, True])
def test_compile_mlir_module_requests_cpp_with_device_outputs(
    tmp_path, monkeypatch, emit_shim
):
    calls = []
    monkeypatch.setattr(
        compile_utils.config, "peano_install_dir", lambda: tmp_path / "peano"
    )
    monkeypatch.setattr(
        compile_utils,
        "_run_aiecc",
        lambda path, args, *, cwd: calls.append((path, args, cwd)),
    )
    cpp = tmp_path / "dispatch_gen.cpp"
    xclbin = tmp_path / "design.xclbin"
    compile_utils.compile_mlir_module(
        "module {}",
        xclbin_path=xclbin,
        work_dir=tmp_path,
        npu_cpp_path=cpp,
        npu_cpp_emit_dispatch_shim=emit_shim,
        fold_ddr_addr_offset=False,
        options=["--get=npu_lowered.mlir"],
    )
    assert len(calls) == 1
    _, args, cwd = calls[0]
    assert Path(cwd) == tmp_path
    assert "--get-xclbin" in args
    assert f"--xclbin-name={xclbin}" in args
    assert "--get-npu-cpp" in args
    assert f"--npu-cpp-name={cpp}" in args
    assert ("--npu-cpp-emit-dispatch-shim" in args) == emit_shim
    assert "--get=npu_lowered.mlir" in args
    assert "--fold-ddr-addr-offset=false" in args


@pytest.mark.parametrize("bound", [{}, {"bar": 3}, {"baz": 7}])
def test_reordered_parameters_reach_generated_instructions(
    tmp_path, npu2_device, bound
):
    from aie.dialects.aiex import npu_address_patch
    from aie.iron import Program, Runtime
    from aie.iron.device import NPU2Col1

    def generator(*, bar: DispatchTime[np.int32] = 3, baz: DispatchTime[np.int32] = 7):
        def sequence(baz_value, bar_value):
            npu_address_patch(addr=119300, arg_idx=0, arg_plus=bar_value)
            npu_address_patch(addr=119304, arg_idx=0, arg_plus=baz_value)

        return Program(NPU2Col1(), Runtime(sequence, [baz, bar])).resolve_program()

    design = CompilableDesign(generator).specialize(**bound)
    _generate_cpp(tmp_path, str(design.generate_mlir()), fold=False)
    library = compile_dispatch_bridge(
        tmp_path, design.dispatch_params, design.dispatch_param_types
    )
    bridge = DispatchBridge(library, design.dispatch_params)
    for values in ({}, {name: 11 + i for i, name in enumerate(design.dispatch_params)}):
        _, scalars = design.split_runtime_args((), values)
        words = bridge.generate(scalars)
        expected = {"bar": 3, "baz": 7, **values}
        patches = words[4:].reshape(2, 12)
        np.testing.assert_array_equal(
            patches[:, 10], [expected["bar"], expected["baz"]]
        )


@pytest.mark.parametrize("compile_kwargs", [{}, {"bound": 8}])
def test_defaulted_compile_param_does_not_consume_dispatch_argument(compile_kwargs):
    def generator(
        a: In,
        bound: CompileTime[int] = 4,
        *,
        scale: DispatchTime[np.int32] = 1,  # pyright: ignore[reportArgumentType]
    ):
        pass

    design = CompilableDesign(generator, compile_kwargs=compile_kwargs)
    tensor = object()
    tensors, scalars = design.split_runtime_args((tensor,), {"scale": 7})
    assert tensors == [tensor]
    assert scalars == {"scale": 7}


def _source(offset=0, *, arg_idx=0, device="npu1_1col"):
    buffers = ", ".join(f"%a{i}: memref<8xi32>" for i in range(arg_idx + 1))
    return f"""module {{
      aie.device({device}) {{
        aie.runtime_sequence @seq({buffers}, %param: i32, %n: index) {{
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %i = %c0 to %n step %c1 {{
            aiex.npu.address_patch(%param : i32) {{addr = {119300 + offset} : ui32, arg_idx = {arg_idx} : i32}}
          }}
        }}
      }}
    }}"""


def _generate_cpp(kernel_dir, source, *, fold=True):
    mlir_path = kernel_dir / "aie.mlir"
    mlir_path.write_text(source)
    _run_aiecc(
        str(mlir_path),
        [
            "--get-npu-cpp",
            f"--npu-cpp-name={kernel_dir / 'dispatch_gen.cpp'}",
            "--npu-cpp-emit-dispatch-shim",
            "--get=npu_lowered.mlir",
            f"--output-dir={kernel_dir}",
            f"--tmpdir={kernel_dir / 'aiecc.prj'}",
            f"--fold-ddr-addr-offset={'true' if fold else 'false'}",
        ],
    )


def _compile(kernel_dir, source=None, *, fold=True):
    _generate_cpp(kernel_dir, _source() if source is None else source, fold=fold)
    return compile_dispatch_bridge(kernel_dir, ["param", "n"], [np.int32, np.uintp])


def _words(bridge, n=2):
    return bridge.generate({"param": 16, "n": n})


def test_generated_bridge_matches_static_binary(tmp_path):
    path = _compile(tmp_path)
    bridge = DispatchBridge(path, ["param", "n"])
    first = _words(bridge, 1)
    larger = _words(bridge, 3)
    assert larger.size > first.size > 0
    for n, words in [(1, first), (3, larger)]:
        static = (
            _source()
            .replace(
                "%a0: memref<8xi32>, %param: i32, %n: index",
                "%a0: memref<8xi32>",
            )
            .replace(
                "%c0 =",
                f"%param = arith.constant 16 : i32\n%n = arith.constant {n} : index\n%c0 =",
            )
        )
        static_dir = tmp_path / f"static-{n}"
        static_dir.mkdir()
        mlir_path = static_dir / "aie.mlir"
        mlir_path.write_text(static)
        binary_path = static_dir / "insts.bin"
        _run_aiecc(
            str(mlir_path),
            [
                "--get-npu-insts",
                f"--npu-insts-name={binary_path}",
                f"--output-dir={static_dir}",
                f"--tmpdir={static_dir / 'aiecc.prj'}",
            ],
        )
        np.testing.assert_array_equal(words, np.fromfile(binary_path, dtype="<u4"))


def test_rebuild_preserves_loaded_and_unloaded_generations(tmp_path):
    old_path = _compile(tmp_path)
    old_bridge = DispatchBridge(old_path, ["param", "n"])
    old_words = _words(old_bridge)
    new_path = _compile(tmp_path, _source(offset=4))
    assert old_path != new_path
    assert old_path.name == f"dispatch-{_manifest._digest(old_path)}{old_path.suffix}"
    new_bridge = DispatchBridge(new_path, ["param", "n"])
    assert not np.array_equal(old_words, _words(new_bridge))
    np.testing.assert_array_equal(old_words, _words(old_bridge))
    np.testing.assert_array_equal(
        old_words, _words(DispatchBridge(old_path, ["param", "n"]))
    )
    assert not list(tmp_path.glob("dispatch.staging.*"))


def test_identical_rebuild_does_not_replace_mapped_generation(tmp_path):
    path = _compile(tmp_path)
    bridge = DispatchBridge(path, ["param", "n"])
    before = path.stat()
    # PE linker timestamps have one-second resolution.
    time.sleep(1.1)
    assert _compile(tmp_path) == path
    after = path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert _words(bridge).size > 0
    assert not list(tmp_path.glob("dispatch.staging.*"))


def test_failed_compile_cleans_linker_companions(tmp_path, monkeypatch):
    import subprocess
    from aie.utils.compile.jit import _dispatch_compile

    path = _compile(tmp_path)
    before = path.read_bytes()

    def fail_compile(command, **kwargs):
        staging = Path(command[command.index("-o") + 1])
        for suffix in (".dll", ".lib", ".exp"):
            staging.with_suffix(suffix).write_bytes(b"partial link")
        raise subprocess.CalledProcessError(1, command, stderr="link failed")

    monkeypatch.setattr(
        _dispatch_compile,
        "host_shared_lib_cmd",
        lambda src, out, **kwargs: ["compiler", str(src), "-o", str(out)],
    )
    monkeypatch.setattr(_dispatch_compile.subprocess, "run", fail_compile)
    with pytest.raises(DispatchCompileError, match="link failed"):
        compile_dispatch_bridge(tmp_path, ["param", "n"], [np.int32, np.uintp])
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("dispatch.staging.*"))


def test_abi_failure_preserves_loaded_generation(tmp_path):
    path = _compile(tmp_path)
    bridge = DispatchBridge(path, ["param", "n"])
    expected = _words(bridge)
    with pytest.raises(DispatchCompileError, match="declared as int32"):
        _compile(
            tmp_path,
            _source()
            .replace("%param: i32", "%param: i64")
            .replace("%param : i32", "%param : i64"),
        )
    np.testing.assert_array_equal(expected, _words(bridge))
    assert not list(tmp_path.glob("dispatch.staging.*"))


@pytest.mark.parametrize("device", ["npu1_1col", "npu2"])
@pytest.mark.parametrize("arg_idx", [0, 4, 5, 6])
def test_fold_ddr_addr_offset_reaches_translation(tmp_path, device, arg_idx):
    source = _source(arg_idx=arg_idx, device=device)
    folded = DispatchBridge(_compile(tmp_path, source, fold=True), ["param", "n"])
    unfolded = DispatchBridge(_compile(tmp_path, source, fold=False), ["param", "n"])
    for n in (1, 3):
        folded_words, unfolded_words = _words(folded, n), _words(unfolded, n)
        # Four header words, then 12-word DDR patches: argidx at 8, argplus at 10.
        patches = unfolded_words[4:].reshape(n, 12)
        np.testing.assert_array_equal(patches[:, 8], arg_idx)
        np.testing.assert_array_equal(patches[:, 10], 16)
        np.testing.assert_array_equal(patches[:, 11], 0)
        expected = unfolded_words.copy()
        # Firmware translates buffers 0..4; only later buffers need folding.
        if arg_idx >= 5:
            expected[4:].reshape(n, 12)[:, 10] += np.uint32(0x80000000)
        np.testing.assert_array_equal(folded_words, expected)


def test_aiecc_lowers_dynamic_dma_tasks(tmp_path):
    source = (
        Path(__file__).parents[1] / "Targets/NPU/aie_npu_to_cpp_rolled_loop.mlir"
    ).read_text()
    _generate_cpp(tmp_path, source)
    path = compile_dispatch_bridge(tmp_path, ["n"], [np.uintp])
    lowered = (tmp_path / "npu_lowered.mlir").read_text()
    assert "dma_configure_task" not in lowered
    assert "aiex.npu.blockwrite_values" in lowered
    assert "scf.for" in lowered
    bridge = DispatchBridge(path, ["n"])
    assert bridge.generate({"n": 3}).size > bridge.generate({"n": 1}).size


@pytest.mark.parametrize("missing", ["dispatch_gen.cpp", "npu_lowered.mlir"])
def test_bridge_requires_compiler_outputs(tmp_path, missing):
    for name in ("dispatch_gen.cpp", "npu_lowered.mlir"):
        if name != missing:
            (tmp_path / name).write_text("")
    with pytest.raises(DispatchCompileError, match="expected aiecc"):
        compile_dispatch_bridge(tmp_path, ["n"], [np.uintp])


def test_python_bridge_rejects_unprovided_pdi_resources():
    with Context():
        module = Module.parse(
            "module { aie.device(npu2) { aie.runtime_sequence @seq(%n: index) "
            "{ aiex.npu.load_pdi {id = 1 : i32} } } }"
        )
        with pytest.raises(DispatchCompileError, match="cannot supply load_pdi"):
            _check_runtime_sequence_abi(module, ["n"], [np.uintp])


@pytest.mark.parametrize("count", [0, 2])
def test_exactly_one_runtime_sequence_required(count):
    sequences = "\n".join(
        f"aie.runtime_sequence @seq{i}(%n: index) {{}}" for i in range(count)
    )
    with Context():
        module = Module.parse(f"module {{ aie.device(npu1_1col) {{ {sequences} }} }}")
        with pytest.raises(DispatchCompileError, match="exactly one runtime_sequence"):
            _check_runtime_sequence_abi(module, ["n"], [np.uintp])


@pytest.mark.parametrize("scalar_type", ["f32", "i7", "vector<4xi32>"])
def test_unsupported_runtime_scalar_rejected(scalar_type):
    with Context():
        module = Module.parse(
            f"module {{ aie.device(npu1_1col) {{ aie.runtime_sequence @seq(%p: {scalar_type}) {{}} }} }}"
        )
        with pytest.raises(DispatchCompileError, match="Unsupported dispatch scalar"):
            _check_runtime_sequence_abi(module, ["p"], [np.int32])


@pytest.mark.parametrize(
    "arguments,names,types",
    [
        ("%a: memref<8xi32>", [], []),
        ("%a: memref<*xi32>, %p: i32", ["p"], [np.int32]),
        ("%p: i32, %a: memref<8xi32>, %n: index", ["p", "n"], [np.int32, np.uintp]),
        ("%p: i32, %n: index", ["p", "n"], [np.int32, np.uintp]),
    ],
)
def test_scalar_and_memref_argument_order(arguments, names, types):
    with Context():
        module = Module.parse(
            f"module {{ aie.device(npu1_1col) {{ aie.runtime_sequence @seq({arguments}) {{}} }} }}"
        )
        _check_runtime_sequence_abi(module, names, types)
