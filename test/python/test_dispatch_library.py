# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Compiler-only integration tests using real MLIR and the host C++ compiler."""

from pathlib import Path

import numpy as np
import pytest
from aie.dialects.aie import translate_npu_to_binary
from aie.ir import Context, Module
from aie.passmanager import PassManager
from aie.utils.compile.jit import _manifest
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.compile.jit._dispatch_compile import (
    DispatchCompileError,
    _check_runtime_sequence_abi,
    compile_dispatch_bridge,
)
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import CompileTime, DispatchTime, In


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
    (tmp_path / "input_with_addresses.mlir").write_text(str(design.generate_mlir()))
    library = compile_dispatch_bridge(
        tmp_path, design.dispatch_params, False, design.dispatch_param_types
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


def _compile(kernel_dir, source=None, *, fold=True):
    (kernel_dir / "input_with_addresses.mlir").write_text(
        _source() if source is None else source
    )
    return compile_dispatch_bridge(
        kernel_dir, ["param", "n"], fold, [np.int32, np.uintp]
    )


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
        with Context():
            module = Module.parse(static)
            PassManager.parse("builtin.module(aie-npu-dma-lowering)").run(
                module.operation
            )
            expected = translate_npu_to_binary(module.operation)
        np.testing.assert_array_equal(words, np.asarray(expected, dtype=np.uint32))


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
    assert _compile(tmp_path) == path
    after = path.stat()
    assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
    assert _words(bridge).size > 0


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


def test_registered_pipeline_lowers_dynamic_dma_tasks(tmp_path):
    source = (
        Path(__file__).parents[1] / "Targets/NPU/aie_npu_to_cpp_rolled_loop.mlir"
    ).read_text()
    (tmp_path / "input_with_addresses.mlir").write_text(source)
    path = compile_dispatch_bridge(tmp_path, ["n"], True, [np.uintp])
    lowered = (tmp_path / "dispatch_lowered.mlir").read_text()
    assert "dma_configure_task" not in lowered
    assert "aiex.npu.blockwrite_values" in lowered
    assert "scf.for" in lowered
    bridge = DispatchBridge(path, ["n"])
    assert bridge.generate({"n": 3}).size > bridge.generate({"n": 1}).size


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


def test_translation_binding_defaults_and_failure():
    from aie.dialects.aie import translate_npu_to_cpp

    with Context():
        cpp = translate_npu_to_cpp(Module.parse(_source()).operation)
        assert "dispatch_generate" not in cpp
        with pytest.raises(RuntimeError, match="translate"):
            translate_npu_to_cpp(
                Module.parse("module {}").operation, emit_dispatch_shim=True
            )
