# ./python/aie/dialects/aie/__init__.py -*- Python -*-
import inspect

# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import List, Optional, Union, Tuple

from ._aie_enum_gen import *
from ._aie_ops_gen import *
from ._aie_ops_gen import _Dialect
from .func import CallOp, FuncOp
from .._mlir_libs import get_dialect_registry
from .._mlir_libs._aie import *
from .._mlir_libs._aie import (
    ObjectFifoType,
    translate_aie_vec_to_cpp,
    ObjectFifoSubviewType,
)

from ..extras import types as T
from ..extras.dialects.ext.arith import constant
from ..extras.meta import region_op
from ..extras.util import Successor, get_user_code_loc, region_adder
from ..ir import (
    ArrayAttr,
    Attribute,
    Block,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    _typeAttr,
    TypeAttr,
    _i32ArrayAttr,
)
from ._ods_common import _cext

# Comes from _aie
register_dialect(get_dialect_registry())

assert _cext.globals._check_dialect_module_loaded("aie")


def external_func(name, inputs, outputs=None, visibility="private"):
    if outputs is None:
        outputs = []
    return FuncOp(
        name=name, type=FunctionType.get(inputs, outputs), visibility=visibility
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


def bd_dim_layout(size, stride):
    return Attribute.parse(f"#aie.bd_dim_layout<{size=}, {stride=}>")


@register_attribute_builder("BDDimLayoutArrayAttr")
def bd_dim_layout_array_attr_builder(
    tups: List[Union[Attribute, Tuple[int]]], context=None
):
    if isinstance(tups, list) and all(isinstance(t, tuple) for t in tups):
        tups = list(map(lambda t: bd_dim_layout(*t), tups))
    return Attribute.parse(
        f'#aie<bd_dim_layout_arr[{", ".join(map(str, tups))}]>', context=context
    )


@register_attribute_builder("BDDimLayoutArrayArrayAttr")
def bd_dim_layout_array_array_attr_builder(tup_arrs: List[List[tuple]], context=None):
    tup_arrs = list(map(bd_dim_layout_array_attr_builder, tup_arrs))
    return Attribute.parse(
        f'#aie<bd_dim_layout_arr_arr[{", ".join(map(str, tup_arrs))}]>', context=context
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


@register_attribute_builder("AIE_ObjectFifo_Depth")
def _objectFifo_depth_attr(x, context):
    if isinstance(x, list):
        return _i32ArrayAttr(x, context)
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), x)


#### AIE Wrappers ####

Device = DeviceOp


class Core(CoreOp):
    # Until https://github.com/llvm/llvm-project/pull/73620 gets figured out.
    def __init__(self, tile, link_with=None):
        super().__init__(result=T.index(), tile=tile, link_with=link_with)


# Create an aie buffer of (size x datatype) on given tile.
# size examples: [256], [256, 256], [256, 256,]
class Buffer(BufferOp):
    def __init__(self, tile, size, datatype, name=None):
        super().__init__(buffer=T.memref(*size, datatype), tile=tile, sym_name=name)


# Create an aie external buffer of (size x datatype).
# size examples: [256], [256, 256], [256, 256,]
class ExternalBuffer(ExternalBufferOp):
    def __init__(self, size, datatype, name=None):
        super().__init__(buffer=T.memref(*size, datatype), sym_name=name)


# Create an aie objectFifo between specified tiles, with given depth and memref datatype.
# depth examples: 2, [2,2,7]
class OrderedObjectBuffer(ObjectFifoCreateOp):
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


# Create an aie objectFifo acquire op of given number of elements with given memref datatype,
# from objFifo with given name.
class ObjectFifoAcquireOp(ObjectFifoAcquireOp):
    def __init__(self, port, of_name, num_elem, datatype):
        subview_t = ObjectFifoSubviewType.get(datatype)
        self.datatype = datatype
        super().__init__(subview_t, port, of_name, num_elem)

    def acquired_elem(self):
        objects = []
        if self.size.value == 1:
            return ObjectFifoSubviewAccessOp(
                self.datatype, self.subview, self.size.value - 1
            )
        for i in range(self.size.value):
            objects.append(ObjectFifoSubviewAccessOp(self.datatype, self.subview, i))
        return objects


def acquire(port, of_name, num_elem, datatype):
    return ObjectFifoAcquireOp(port, of_name, num_elem, datatype)


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


core = region_op(Core, terminator=lambda *_: EndOp())
device = region_op(Device)
mem = region_op(
    lambda tile, *, loc=None, ip=None: MemOp(T.index(), tile, loc=loc, ip=ip)
)
shim_dma = region_op(
    lambda tile, *, loc=None, ip=None: ShimDMAOp(T.index(), tile, loc=loc, ip=ip)
)
memtile_dma = region_op(
    lambda tile, *, loc=None, ip=None: MemTileDMAOp(T.index(), tile, loc=loc, ip=ip)
)


@region_op
def dma(channel_dir, channel_index, *, num_blocks=1, loop=None, loc=None, ip=None):
    return DMAOp(
        valid=T.bool(),
        channelDir=channel_dir,
        channelIndex=channel_index,
        num_bds=num_blocks,
        loop=loop,
        loc=loc,
        ip=ip,
    )


@region_adder()
def another_bd(dma_op):
    for r in dma_op.regions:
        if len(r.blocks) == 0:
            r.blocks.append()
        if len(r.blocks[0].operations) == 0:
            return r

    raise Exception("couldn't find empty region to add to.")


@_cext.register_operation(_Dialect, replace=True)
class DMAStartOp(DMAStartOp):
    def __init__(
        self,
        channel_dir,
        channel_index,
        *,
        dest: Optional[Union[Successor, Block]] = None,
        chain: Optional[Union[Successor, Block]] = None,
        loc=None,
        ip=None,
    ):
        if isinstance(dest, Successor):
            dest = dest.block
        if isinstance(chain, Successor):
            chain = chain.block
        if dest is None:
            dest = InsertionPoint.current.block
        if chain is None:
            chain = InsertionPoint.current.block
        super().__init__(channel_dir, channel_index, dest, chain, loc=loc, ip=ip)

    @property
    def dest(self):
        return Successor(self, [], self.successors[0], 0)

    @property
    def chain(self):
        return Successor(self, [], self.successors[1], 1)


def dma_start(
    channel_dir,
    channel_index,
    *,
    dest: Optional[Union[Successor, Block]] = None,
    chain: Optional[Union[Successor, Block]] = None,
    loc=None,
    ip=None,
):
    op = DMAStartOp(channel_dir, channel_index, dest=dest, chain=chain, loc=loc, ip=ip)
    return op.dest, op.chain


@_cext.register_operation(_Dialect, replace=True)
class NextBDOp(NextBDOp):
    def __init__(
        self, dest: Optional[Union[Successor, Block]] = None, *, loc=None, ip=None
    ):
        if isinstance(dest, Successor):
            dest = dest.block
        if dest is None:
            dest = InsertionPoint.current.block
        if loc is None:
            loc = get_user_code_loc()
        super().__init__(dest, loc=loc, ip=ip)

    @property
    def dest(self):
        return Successor(self, [], self.successors[0], 0)


def next_bd(dest: Optional[Union[Successor, Block]] = None, loc=None, ip=None):
    return NextBDOp(dest, loc=loc, ip=ip).dest


_buffer = buffer


def _get_sym_name(previous_frame, check_func_call):
    try:
        with open(inspect.getfile(previous_frame)) as src_file:
            src_lines = src_file.readlines()
            src_line = src_lines[previous_frame.f_lineno - 1].strip()
            ident, func_call = map(lambda x: x.strip(), src_line.split("=", maxsplit=1))
            assert check_func_call in func_call
        return ident
    except:
        return None


def buffer(buffer, tile, *, sym_name=None, address=None, loc=None, ip=None):
    return _buffer(
        buffer,
        tile,
        sym_name=sym_name or _get_sym_name(inspect.currentframe().f_back, "buffer"),
        address=address,
        loc=loc,
        ip=ip,
    )


_lock = lock


def lock(tile, *, lock_id=None, init=None, sym_name=None, loc=None, ip=None):
    return _lock(
        tile,
        lock_id=lock_id,
        init=init,
        sym_name=sym_name or _get_sym_name(inspect.currentframe().f_back, "lock"),
        loc=loc,
        ip=ip,
    )
