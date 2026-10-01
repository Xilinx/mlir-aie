# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ...dialects import arith, emitc  # pyright: ignore[reportMissingImports]
from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    NpuRequireOp,
)
from ...extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
    index_cast,
)
from ...ir import (  # pyright: ignore[reportMissingImports]
    IndexType,
    IntegerAttr,
    IntegerType,
    OpView,
    Value,
)


def as_signless(v, width: int, what: str = "value"):
    """Bring an integer operand to the signless `i{width}` an op takes.

    Python ints pass through. An index Value is cast, a narrower signed one
    sign-extended, and an unsigned one (a `DispatchTime[np.uint*]` scalar)
    zero-extended with `emitc.cast`, since arith takes only signless integers;
    an unsigned constant folds to its Python int. A wider Value is guarded at
    dispatch with `aiex.npu.require` to fit `width` bits before it is
    truncated, so an out-of-range value refuses the dispatch instead of
    wrapping. `what` names the operand in that guard's message.
    """
    if not isinstance(v, Value):
        return v
    target = IntegerType.get_signless(width)
    if isinstance(v.type, IndexType):
        v = index_cast(v, to=IntegerType.get_signless(64))
    elif v.type.is_unsigned:
        bits = v.type.width
        owner = v.owner
        if isinstance(owner, OpView) and owner.operation.name == "emitc.constant":
            value = IntegerAttr(owner.attributes["value"]).value & ((1 << bits) - 1)
            if value >> width:
                raise ValueError(f"{what} {value} does not fit in {width} bits")
            return value
        v = emitc.CastOp(IntegerType.get_signless(max(bits, width)), v).result
    bits = v.type.width
    if bits < width:
        return arith.extsi(target, v)
    if bits == width:
        return v
    fits = arith.cmpi(arith.CmpIPredicate.ule, v, constant((1 << width) - 1, v.type))
    NpuRequireOp(fits, f"a runtime {what} does not fit in {width} bits")
    return arith.trunci(target, v)
