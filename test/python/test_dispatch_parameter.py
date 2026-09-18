# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Dispatch identities and scope diagnostics, using real IR without hardware."""

import operator

import numpy as np
import pytest
from aie.helpers.util import np_ndarray_type_to_memref_type
from aie.ir import BlockArgument, IntegerAttr
from aie.iron import Program, Runtime, Worker
from aie.iron.device import NPU2Col1
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.compile.jit.markers import DispatchTime


@pytest.mark.parametrize("other_type", [np.int32, np.int64, np.uint32])
@pytest.mark.parametrize("bound", [{}, {"bar": 3}, {"baz": 7}, {"bar": 3, "baz": 7}])
def test_reordered_dispatch_identity(other_type, bound, npu2_device):
    observed = {}
    tensor_type = np.ndarray[(16,), np.dtype[np.int32]]

    def gen(*, bar: DispatchTime[np.int32], baz: DispatchTime[other_type]):
        def seq(baz_value, a, bar_value, b):
            observed.update(
                a=BlockArgument(a.op).arg_number, b=BlockArgument(b.op).arg_number
            )
            for name, value in (("bar", bar_value), ("baz", baz_value)):
                observed[name] = (
                    IntegerAttr(value.owner.attributes["value"]).value
                    if name in bound
                    else BlockArgument(value).arg_number
                )

        return Program(
            NPU2Col1(), Runtime(seq, [baz, tensor_type, bar, tensor_type])
        ).resolve_program()

    design = CompilableDesign(gen).specialize(**bound)
    assert design.generate_mlir().operation.verify()
    # Scalar ABI stays in signature order, even with interleaved memrefs.
    dynamic = [name for name in ("bar", "baz") if name not in bound]
    block_names = [name for name in ("baz", "a", "bar", "b") if name not in bound]
    scalar_slots = [i for i, name in enumerate(block_names) if name in dynamic]
    for name, slot in zip(dynamic, scalar_slots):
        assert observed[name] == slot
    for name, value in bound.items():
        assert observed[name] == value
    assert observed["a"] < observed["b"]


@pytest.mark.parametrize(
    "use",
    [
        bool,
        int,
        operator.index,
        float,
        lambda x: range(x),
        lambda x: x + 1,
        lambda x: 1 + x,
        lambda x: x * 2,
        lambda x: x // 2,
        lambda x: x < 3,
        lambda x: x == 3,
        lambda x: x & 1,
        lambda x: np.int32(1) + x,
        lambda x: np.array(x),
        lambda x: np.dtype(x),
        lambda x: np.empty((x,), dtype=np.int32),
        lambda x: np_ndarray_type_to_memref_type(np.ndarray[(x,), np.dtype[np.int32]]),
        lambda x: np_ndarray_type_to_memref_type(np.ndarray[x, np.dtype[np.int32]]),
        lambda x: np_ndarray_type_to_memref_type(np.ndarray[(16,), np.dtype[x]]),
        lambda x: Worker(lambda value: None, [x]),
        lambda x: Worker(lambda values: None, [[x]]),
        lambda x: Worker(lambda: None, stack_size=x),
        lambda x: Worker(lambda: None, while_true=x),
    ],
)
def test_dispatch_generation_time_misuse(use, npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        use(count)

    with pytest.raises(
        TypeError, match="DispatchTime parameter 'count'.*generation-time"
    ):
        CompilableDesign(gen).generate_mlir()


@pytest.mark.parametrize("replacement", [None, np.int32])
def test_omitted_or_replaced_parameter(replacement, npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        args = [] if replacement is None else [replacement]
        return Program(NPU2Col1(), Runtime(lambda *args: None, args)).resolve_program()

    with pytest.raises(TypeError, match="'count' was not bound"):
        CompilableDesign(gen).generate_mlir()


def test_duplicate_parameter_is_rejected(npu2_device):
    def gen(*, bar: DispatchTime[np.int32], baz: DispatchTime[np.int32]):
        return Program(
            NPU2Col1(), Runtime(lambda *args: None, [bar, bar])
        ).resolve_program()

    with pytest.raises(TypeError, match="exactly once"):
        CompilableDesign(gen).generate_mlir()


def test_nested_parameter_is_rejected(npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        Runtime(lambda *args: None, [count, [count]])

    with pytest.raises(TypeError, match="direct Runtime fn_args entry"):
        CompilableDesign(gen).generate_mlir()


def test_bare_scalar_cannot_bypass_dispatch_identity(npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        Runtime(lambda *args: None, [np.int32, count])

    with pytest.raises(TypeError, match="bare scalar types"):
        CompilableDesign(gen).generate_mlir()


def test_parameter_cannot_bind_multiple_sequences(npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        Program(NPU2Col1(), Runtime(lambda value: None, [count])).resolve_program()
        return Program(
            NPU2Col1(), Runtime(lambda value: None, [count])
        ).resolve_program()

    with pytest.raises(TypeError, match="exactly once"):
        CompilableDesign(gen).generate_mlir()


def test_parameters_cannot_be_split_across_sequences(npu2_device):
    def gen(*, bar: DispatchTime[np.int32], baz: DispatchTime[np.int32]):
        Program(NPU2Col1(), Runtime(lambda value: None, [bar])).resolve_program()
        return Program(NPU2Col1(), Runtime(lambda value: None, [baz])).resolve_program()

    with pytest.raises(TypeError, match="one Runtime sequence"):
        CompilableDesign(gen).generate_mlir()


def test_alias_preserves_identity(npu2_device):
    observed = []

    def gen(*, bar: DispatchTime[np.int32], baz: DispatchTime[np.int32]):
        alias = baz

        def seq(baz_value, bar_value):
            observed.extend(
                [
                    BlockArgument(baz_value).arg_number,
                    BlockArgument(bar_value).arg_number,
                ]
            )

        return Program(NPU2Col1(), Runtime(seq, [alias, bar])).resolve_program()

    assert CompilableDesign(gen).generate_mlir().operation.verify()
    assert observed == [1, 0]


def test_parameters_from_different_generations_cannot_mix(npu2_device):
    retained = []

    def first(*, count: DispatchTime[np.int32]):
        retained.append(count)
        return Program(
            NPU2Col1(), Runtime(lambda value: None, [count])
        ).resolve_program()

    CompilableDesign(first).generate_mlir()

    def second(*, count: DispatchTime[np.int32]):
        Runtime(lambda *args: None, [retained[0], count])

    with pytest.raises(TypeError, match="different designs"):
        CompilableDesign(second).generate_mlir()


def test_low_level_runtime_scalar_types_keep_positional_binding(npu2_device):
    observed = []

    def gen():
        def seq(first, second, folded):
            observed.extend(
                [
                    (BlockArgument(first).arg_number, str(first.type)),
                    (BlockArgument(second).arg_number, str(second.type)),
                    IntegerAttr(folded.owner.attributes["value"]).value,
                ]
            )

        return Program(
            NPU2Col1(), Runtime(seq, [np.int64, np.int32, np.int32(5)])
        ).resolve_program()

    assert CompilableDesign(gen).generate_mlir().operation.verify()
    assert observed == [(0, "i64"), (1, "i32"), 5]


def test_specialization_allows_compile_time_use(npu2_device):
    def gen(*, count: DispatchTime[np.int32]):
        assert count == 3
        assert list(range(count)) == [0, 1, 2]
        assert bool(count)
        return Program(NPU2Col1(), Runtime(lambda: None)).resolve_program()

    assert CompilableDesign(gen).specialize(count=3).generate_mlir().operation.verify()
