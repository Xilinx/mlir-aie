# ./python/aie/dialects/aie/__init__.py -*- Python -*-

# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from functools import wraps

from ._AIE_enum_gen import *
from ._AIE_ops_gen import *
from .._mlir_libs._aie import *
from ..dialects import arith
from ..dialects import memref
from ..dialects.func import *
from ..dialects.scf import *
from ..ir import *
from ..util import constant


class AddI(arith.AddIOp):
    """Specialize AddIOp class constructor to take python integers"""

    def __init__(self, lhs, rhs):
        if isinstance(lhs, int):
            lhs = constant(lhs)
        if isinstance(rhs, int):
            rhs = constant(rhs)
        super().__init__(lhs=lhs, rhs=rhs)


# Wrapper for func FuncOp with "private" visibility.
class privateFunc(FuncOp):
    """Specialize FuncOp class constructor to take python integers"""

    def __init__(self, name, inputs, outputs=[], visibility="private"):
        super().__init__(
            name=name, type=FunctionType.get(inputs, outputs), visibility=visibility
        )


# Wrapper for func FuncOp with "private" visibility.
class publicFunc(FuncOp):
    """Specialize FuncOp class constructor to take python integers"""

    def __init__(self, name, callbackFunc, inputs, outputs=[], visibility="public"):
        super().__init__(
            name=name,
            type=FunctionType.get(inputs, outputs),
            visibility=visibility,
            body_builder=callbackFunc,
        )


# Wrapper for func CallOp.
class Call(CallOp):
    """Specialize CallOp class constructor to take python integers"""

    def __init__(self, calleeOrResults, inputs=[], input_types=[]):
        attrInputs = []
        for i in inputs:
            if isinstance(i, int):
                attrInputs.append(constant(i))
            else:
                attrInputs.append(i)
        if isinstance(calleeOrResults, FuncOp):
            super().__init__(
                calleeOrResults=calleeOrResults, argumentsOrCallee=attrInputs
            )
        else:
            super().__init__(
                calleeOrResults=input_types,
                argumentsOrCallee=FlatSymbolRefAttr.get(calleeOrResults),
                arguments=attrInputs,
            )


class Load(memref.LoadOp):
    """Specialize LoadOp class constructor to take python integers"""

    def __init__(self, mem, indices):
        valueIndices = []
        if isinstance(indices, list):
            for i in indices:
                valueIndices.append(constant(i, index=True))
        else:
            valueIndices.append(constant(indices, index=True))
        super().__init__(memref=mem, indices=valueIndices)


class Store(memref.StoreOp):
    """Specialize StoreOp class constructor to take python integers"""

    def __init__(self, val, mem, indices):
        if isinstance(val, int):
            intVal = constant(val)
        else:
            intVal = val
        valueIndices = []
        if isinstance(indices, list):
            for i in indices:
                valueIndices.append(constant(i, index=True))
        else:
            valueIndices.append(constant(indices, index=True))
        super().__init__(value=intVal, memref=mem, indices=valueIndices)


def op_region_builder(op, op_region, terminator=None):
    def builder_wrapper(body_builder):
        # add a block with block args having types ...
        if len(op_region.blocks) == 0:
            sig = inspect.signature(body_builder)
            types = [p.annotation for p in sig.parameters.values()]
            if not (
                len(types) == len(sig.parameters)
                and all(isinstance(t, Type) for t in types)
            ):
                raise ValueError(
                    f"for {body_builder=} either missing a type annotation or type annotation isn't a mlir type: {sig}"
                )
            op_region.blocks.append(*types)
        with InsertionPoint(op_region.blocks[0]):
            results = body_builder()
            if terminator is not None:
                res = []
                if isinstance(results, (tuple, list)):
                    res.extend(results)
                elif results is not None:
                    res.append(results)
                terminator(res)

        return op

    return builder_wrapper


def region_op(op_constructor, terminator=None):
    # the decorator itself
    def op_decorator(*args, **kwargs):
        op = op_constructor(*args, **kwargs)
        op_region = op.regions[0]

        return op_region_builder(op, op_region, terminator)

    # this is like make_maybe_no_args_decorator but a little different because the decorators here
    # are already wrapped (or something like that)
    @wraps(op_decorator)
    def maybe_no_args(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return op_decorator()(args[0])
        else:
            return op_decorator(*args, **kwargs)

    return maybe_no_args


from typing import List


def dim_tuple_attr_builder(wrap, stepsize):
    return Attribute.parse(f"#AIE.DimTuple<{wrap}, {stepsize}>")


@register_attribute_builder("AIE_DimTupleArrayAttr")
def dim_tuple_array_attr_builder(tups: List[tuple], context=None):
    tups = list(map(lambda t: dim_tuple_attr_builder(*t), tups))
    return Attribute.parse(
        f'#AIE<DimTupleArray[{", ".join(map(str, tups))}]>', context=context
    )


@register_attribute_builder("AIE_DimTupleArrayArrayAttr")
def dim_tuple_array_array_attr_builder(tup_arrs: List[List[tuple]], context=None):
    tup_arrs = list(map(dim_tuple_array_attr_builder, tup_arrs))
    return Attribute.parse(
        f'#AIE<DimTupleArrayArray[{", ".join(map(str, tup_arrs))}]>', context=context
    )


@register_attribute_builder("AIEI1Attr")
def _i1Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(1, context=context), x)


@register_attribute_builder("AIEI8Attr")
def _i8Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(8, context=context), x)


@register_attribute_builder("AIEI16Attr")
def _i16Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(16, context=context), x)


@register_attribute_builder("AIEI32Attr")
def _i32Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), x)


@register_attribute_builder("AIEI64Attr")
def _i64Attr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), x)


#### AIE Wrappers ####


# Create and print ModuleOp.
def constructAndPrintInModule(f):
    with Context() as ctx, Location.unknown():
        register_dialect(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)


Device = DeviceOp


# Create an aie tile on specified (col, row).
class Tile(TileOp):
    """Specialize TileOp class constructor to take python integers"""

    def __init__(self, col, row):
        idx_ty = IndexType.get()
        super().__init__(result=idx_ty, col=col, row=row)


# Create an aie core on specified aie tile.
class Core(CoreOp):
    """Specialize CoreOp class constructor to take python integers"""

    def __init__(self, tile, link_with=None):
        idx_ty = IndexType.get()
        if link_with != None:
            super().__init__(result=idx_ty, tile=tile, link_with=link_with)
        else:
            super().__init__(result=idx_ty, tile=tile)


# Create an aie buffer of (size x datatype) on given tile.
# size examples: [256], [256, 256], [256, 256,]
class Buffer(BufferOp):
    """Specialize BufferOp class constructor to take python integers"""

    def __init__(self, tile, size, datatype, name=None):
        memRef_ty = MemRefType.get(size, datatype)
        if name is None:
            super().__init__(buffer=memRef_ty, tile=tile)
        else:
            super().__init__(buffer=memRef_ty, tile=tile, sym_name=name)


# Create an aie external buffer of (size x datatype).
# size examples: [256], [256, 256], [256, 256,]
class ExternalBuffer(ExternalBufferOp):
    """Specialize ExternalBufferOp class constructor to take python integers"""

    def __init__(self, size, datatype, name=None):
        memRef_ty = MemRefType.get(size, datatype)
        if name is None:
            super().__init__(buffer=memRef_ty)
        super().__init__(buffer=memRef_ty, sym_name=name)


# Create an aie objectFifo between specified tiles, with given depth and memref datatype.
# depth examples: 2, [2,2,7]
class OrderedObjectBuffer(ObjectFifoCreateOp):
    """Specialize ObjectFifoCreateOp class constructor to take python integers"""

    def __init__(
        self,
        name,
        tile0,
        tile1,
        depth,
        datatype,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    ):
        if dimensionsFromStreamPerConsumer is None:
            dimensionsFromStreamPerConsumer = []
        if dimensionsToStream is None:
            dimensionsToStream = []
        int_ty = IntegerType.get_signless(32)
        if isinstance(depth, int):
            int_depth = IntegerAttr.get(int_ty, depth)
        else:
            int_depths = []
            for d in depth:
                int_depths.append(IntegerAttr.get(int_ty, d))
            int_depth = ArrayAttr.get(int_depths)
        of_Ty = ObjectFifoType.get(datatype)
        super().__init__(
            sym_name=name,
            producerTile=tile0,
            consumerTiles=tile1,
            elemNumber=int_depth,
            elem_type=TypeAttr.get(of_Ty),
            dimensionsToStream=dimensionsToStream,
            dimensionsFromStreamPerConsumer=dimensionsFromStreamPerConsumer,
        )


# Create an aie objectFifo link op between specified input and output arrays of objFifos.
class Link(ObjectFifoLinkOp):
    """Specialize ObjectFifoLinkOp class constructor to take python integers"""

    def __init__(self, ofIns, ofOuts):
        ofIns_sym = []
        ofOuts_sym = []
        for o in ofIns:
            ofIns_sym.append(FlatSymbolRefAttr.get(o))
        for o in ofOuts:
            ofOuts_sym.append(FlatSymbolRefAttr.get(o))
        super().__init__(fifoIns=ofIns_sym, fifoOuts=ofOuts_sym)


# Create an aie objectFifo acquire op of given number of elements with given memref datatype,
# from objFifo with given name.
class Acquire(ObjectFifoAcquireOp):
    """Specialize ObjectFifoAcquireOp class constructor to take python integers"""

    def __init__(self, port, of_name, num_elem, datatype):
        subview_Ty = ObjectFifoSubviewType.get(datatype)
        self.datatype = datatype
        super().__init__(
            subview=subview_Ty, port=port, objFifo_name=of_name, size=num_elem
        )

    def acquiredElem(self):
        objects = []
        if self.size.value == 1:
            return SubviewAccess(self.datatype, self.subview, self.size.value - 1)
        for i in range(self.size.value):
            objects.append(SubviewAccess(self.datatype, self.subview, i))
        return objects


SubviewAccess = ObjectFifoSubviewAccessOp

Release = ObjectFifoReleaseOp


# Create a flow between source and destination tile ports.
class Flow(FlowOp):
    """Specialize FlowOp class constructor to take python integers"""

    def __init__(
        self, source, source_port, source_channel, dest, dest_port, dest_channel
    ):
        super().__init__(
            source=source,
            sourceBundle=source_port,
            sourceChannel=source_channel,
            dest=dest,
            destBundle=dest_port,
            destChannel=dest_channel,
        )


# Create a packet flow between source and destination tile ports.
class PacketFlow(PacketFlowOp):
    """Specialize PacketFlowOp class constructor to take python integers"""

    def __init__(
        self, pkt_id, source, source_port, source_channel, dest, dest_port, dest_channel
    ):
        super().__init__(ID=pkt_id)
        bb = Block.create_at_start(self.ports)
        with InsertionPoint(bb):
            src = PacketSourceOp(source, source_port, source_channel)
            dest = PacketDestOp(dest, dest_port, dest_channel)
            end = EndOp()


#### Global Wrappers ####
core = region_op(Core, terminator=lambda *args: EndOp())
device = region_op(Device)
