# ./python/aie/dialects/aie/__init__.py -*- Python -*-

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
class call(CallOp):
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


# Create an aie objectFifo between specified tiles, with given depth and memref datatype.
# depth examples: 2, [2,2,7]
class objectfifo(ObjectFifoCreateOp):
    def __init__(
        self,
        name,
        producerTile,
        consumerTiles,
        depth,
        datatype,
        dimensionsToStream=None,
        dimensionsFromStreamPerConsumer=None,
    ):
        self.datatype = datatype
        if not isinstance(consumerTiles, List):
            consumerTiles = [consumerTiles]
        if dimensionsFromStreamPerConsumer is None:
            dimensionsFromStreamPerConsumer = []
        if dimensionsToStream is None:
            dimensionsToStream = []
        int_ty = IntegerType.get_signless(32)
        of_Ty = TypeAttr.get(ObjectFifoType.get(datatype))
        super().__init__(
            sym_name=name,
            producerTile=producerTile,
            consumerTiles=consumerTiles,
            elemNumber=depth,
            elemType=of_Ty,
            dimensionsToStream=dimensionsToStream,
            dimensionsFromStreamPerConsumer=dimensionsFromStreamPerConsumer,
        )

    def acquire(self, num_elem):
        subview_t = ObjectFifoSubviewType.get(self.datatype)
        acq = ObjectFifoAcquireOp(subview_t, self.sym_name.value, num_elem)

        objects = []
        if acq.size.value == 1:
            return ObjectFifoSubviewAccessOp(
                self.datatype, acq.subview, acq.size.value - 1
            )
        for i in range(acq.size.value):
            objects.append(ObjectFifoSubviewAccessOp(self.datatype, acq.subview, i))
        return objects

    def release(self, num_elem):
        return objectfifo_release(self.sym_name.value, num_elem)


# Create an aie objectFifo_link between input and output objectFifos.
class objectfifo_link(ObjectFifoLinkOp):
    """Specialize ObjectFifoLinkOp class constructor to take python variables"""
    def __init__(
        self,
        fifoIns,
        fifoOuts,
    ):
        if not isinstance(fifoIns, List):
            fifoIns = [fifoIns]
        if not isinstance(fifoOuts, List):
            fifoOuts = [fifoOuts]
        fifoInRefs = map(
            lambda i: i if isinstance(i, str) else i.sym_name.value, fifoIns
        )
        fifoOutRefs = map(
            lambda i: i if isinstance(i, str) else i.sym_name.value, fifoOuts
        )
        super().__init__(
            fifoIns=fifoInRefs,
            fifoOuts=fifoOutRefs,
        )


# Create a packet flow between source and destination tile ports.
class packet_flow(PacketFlowOp):
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
def dma(channel_dir, channel_index, *, num_bds=1, loop=None, loc=None, ip=None):
    return DMAOp(
        valid=T.bool(),
        channelDir=channel_dir,
        channelIndex=channel_index,
        num_bds=num_bds,
        loop=loop,
        loc=loc,
        ip=ip,
    )


@region_adder()
def another_bd(dma_op):
    for r in dma_op.regions:
        if len(r.blocks[0].operations) == 0:
            return r


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
