# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from dataclasses import dataclass
import inspect
from typing import List, Optional, Tuple, Union

import numpy as np

from ._aie_enum_gen import *
from ._aie_ops_gen import *
from ._aie_ops_gen import _Dialect
from ._ods_common import _cext
from .func import CallOp, FuncOp
from .._mlir_libs import get_dialect_registry

# noinspection PyUnresolvedReferences
from .._mlir_libs._aie import (
    ObjectFifoSubviewType,
    ObjectFifoType,
    aie_llvm_link,
    generate_bcf,
    generate_cdo,
    generate_xaie,
    npu_instgen,
    register_dialect,
    translate_aie_vec_to_cpp,
    translate_mlir_to_llvmir,
)
from ..extras import types as T
from ..extras.dialects.ext.arith import constant

# noinspection PyUnresolvedReferences
from ..extras.dialects.ext import memref
from ..extras.meta import region_op

# this is inside the aie-python-extras (shared) namespace package
from ..extras.util import (
    Successor,
    _get_sym_name,
    find_ops,
    find_parent_of_type,
    get_user_code_loc,
    region_adder,
)

from ..ir import (
    ArrayAttr,
    Attribute,
    Block,
    DenseElementsAttr,
    DictAttr,
    FlatSymbolRefAttr,
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    TypeAttr,
    UnitAttr,
    _i32ArrayAttr,
)

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


# Create an aie buffer of (shape x datatype) on given tile.
# shape examples: [256], [256, 256], [256, 256,]
class Buffer(BufferOp):
    def __init__(
        self, tile, shape, datatype, name=None, initial_value=None, loc=None, ip=None
    ):
        if initial_value is not None:
            assert isinstance(initial_value, np.ndarray)
            initial_value = DenseElementsAttr.get(
                initial_value,
                type=datatype,
                context=None,
            )
        super().__init__(
            buffer=T.memref(*shape, datatype),
            tile=tile,
            sym_name=name,
            initial_value=initial_value,
            loc=loc,
            ip=ip,
        )


# Create an aie external buffer of (shape x datatype).
# shape examples: [256], [256, 256], [256, 256,]
class ExternalBuffer(ExternalBufferOp):
    def __init__(self, shape, datatype, name=None, loc=None, ip=None):
        super().__init__(
            buffer=T.memref(*shape, datatype),
            sym_name=name,
            loc=loc,
            ip=ip,
        )


# Create an aie objectFifo between specified tiles, with given depth and memref datatype.
# depth examples: 2, [2,2,7]
class object_fifo(ObjectFifoCreateOp):
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

    def acquire(self, port, num_elem):
        subview_t = ObjectFifoSubviewType.get(self.datatype)
        acq = ObjectFifoAcquireOp(subview_t, port, self.sym_name.value, num_elem)

        objects = []
        if acq.size.value == 1:
            return ObjectFifoSubviewAccessOp(
                self.datatype, acq.subview, acq.size.value - 1
            )
        for i in range(acq.size.value):
            objects.append(ObjectFifoSubviewAccessOp(self.datatype, acq.subview, i))
        return objects

    def release(self, port, num_elem):
        return objectfifo_release(port, self.sym_name.value, num_elem)


# Create an aie objectFifo_link between input and output objectFifos.
class object_fifo_link(ObjectFifoLinkOp):
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
class packetflow(PacketFlowOp):
    """Specialize PacketFlowOp class constructor to take python integers"""

    def __init__(
        self,
        pkt_id,
        source,
        source_port,
        source_channel,
        dest,
        dest_port,
        dest_channel,
        keep_pkt_header: Optional[bool] = None,
    ):
        super().__init__(ID=pkt_id, keep_pkt_header=keep_pkt_header)
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
switchbox = region_op(
    lambda tile, *, loc=None, ip=None: SwitchboxOp(T.index(), tile, loc=loc, ip=ip)
)
shim_mux = region_op(
    lambda tile, *, loc=None, ip=None: ShimMuxOp(T.index(), tile, loc=loc, ip=ip)
)


@region_op
def dma(
    channel_dir,
    channel_index,
    *,
    num_blocks=1,
    loop=None,
    repeat_count=None,
    sym_name=None,
    loc=None,
    ip=None,
):
    if isinstance(channel_index, IntegerAttr):
        channel_index = channel_index.value
    return DMAOp(
        valid=T.bool(),
        channel_dir=channel_dir,
        channel_index=channel_index,
        num_bds=num_blocks,
        loop=loop,
        repeat_count=repeat_count,
        sym_name=sym_name,
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
        repeat_count: Optional[int] = None,
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
        super().__init__(
            channel_dir,
            channel_index,
            dest,
            chain,
            repeat_count=repeat_count,
            loc=loc,
            ip=ip,
        )

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


def buffer(tile, shape, dtype, name=None, initial_value=None, loc=None, ip=None):
    if name is not None and not name:
        name = _get_sym_name(inspect.currentframe().f_back, "aie\\.buffer|buffer")
    return Buffer(
        tile,
        shape,
        dtype,
        name=name,
        initial_value=initial_value,
        loc=loc,
        ip=ip,
    ).result


def external_buffer(shape, dtype, name=None, loc=None, ip=None):
    return ExternalBuffer(
        shape,
        dtype,
        name=name,
        loc=loc,
        ip=ip,
    ).result


_lock = lock


def lock(
    tile, *, lock_id=None, init=None, sym_name=None, annot=None, loc=None, ip=None
):
    if sym_name is not None and not sym_name:
        sym_name = _get_sym_name(inspect.currentframe().f_back, "aie\\.lock|lock")
    l = _lock(
        tile,
        lock_id=lock_id,
        init=init,
        sym_name=sym_name,
        loc=loc,
        ip=ip,
    )
    if annot is not None:
        l.owner.attributes["annot"] = DictAttr.get({annot: UnitAttr.get()})
    return l


@_cext.register_operation(_Dialect, replace=True)
class FlowOp(FlowOp):
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"


def flow(
    source,
    source_bundle=None,
    source_channel=None,
    dest=None,
    dest_bundle=None,
    dest_channel=None,
):
    assert dest is not None
    if source_bundle is None:
        source_bundle = WireBundle.DMA
    if source_channel is None:
        source_channel = 0
    if dest_bundle is None:
        dest_bundle = WireBundle.DMA
    if dest_channel is None:
        dest_channel = 0
    return FlowOp(
        source, source_bundle, source_channel, dest, dest_bundle, dest_channel
    )


def find_matching_flows(
    tiles,
    filter_source=False,
    filter_dest=False,
    source_annot=None,
    dest_annot=None,
    device=None,
):
    assert not (filter_source and filter_dest), "Can only filter by source XOR dest"
    if device is None:
        device = find_parent_of_type(lambda op: isinstance(op, DeviceOp))

    def _cb(op):
        if isinstance(op, FlowOp):
            if filter_source and op.source.owner.opview not in tiles:
                return False
            if filter_dest and op.dest.owner.opview not in tiles:
                return False

            return (
                op.source.owner.opview in tiles
                or op.dest.owner.opview in tiles
                and (
                    (
                        "source_annot" in op.attributes
                        and source_annot in op.attributes["source_annot"]
                    )
                    if source_annot is not None
                    else True
                )
                and (
                    (
                        "dest_annot" in op.attributes
                        and dest_annot in op.attributes["dest_annot"]
                    )
                    if dest_annot is not None
                    else True
                )
            )

    return sorted(
        find_ops(device, _cb),
        key=lambda a: (
            int(a.source.owner.opview.col),
            int(a.source.owner.opview.row),
            int(a.source_bundle),
            int(a.source_channel),
            int(a.dest.owner.opview.col),
            int(a.dest.owner.opview.row),
            int(a.dest_bundle),
            int(a.dest_channel),
        ),
    )


def find_matching_locks(tiles, sym_name=None, annot=None, device=None):
    if device is None:
        device = find_parent_of_type(lambda op: isinstance(op, DeviceOp))

    def _cb(op):
        if isinstance(op, LockOp):
            return (
                op.tile.owner.opview in tiles
                and (sym_name == str(op.sym_name) if sym_name is not None else True)
                and (
                    ("annot" in op.attributes and annot in op.attributes["annot"])
                    if annot is not None
                    else True
                )
            )

    return sorted(
        [o.result for o in find_ops(device, _cb)],
        key=lambda a: (
            int(a.owner.opview.tile.owner.opview.col),
            int(a.owner.opview.tile.owner.opview.row),
            a.get_name(),
        ),
    )


def find_matching_buffers(tiles, sym_name=None, annot=None, device=None):
    if device is None:
        device = find_parent_of_type(lambda op: isinstance(op, DeviceOp))

    def _cb(op):
        if isinstance(op, BufferOp):
            return (
                op.tile.owner.opview in tiles
                and (sym_name == str(op.sym_name) if sym_name is not None else True)
                and (
                    ("annot" in op.attributes and annot in op.attributes["annot"])
                    if annot is not None
                    else True
                )
            )

    return sorted(
        [o.result for o in find_ops(device, _cb)],
        key=lambda a: (
            int(a.owner.opview.tile.owner.opview.col),
            int(a.owner.opview.tile.owner.opview.row),
            a.get_name(),
        ),
    )


@dataclass
class Neighbors:
    north: TileOp = None
    west: TileOp = None
    south: TileOp = None


def find_neighbors(tile, device=None, logical=True):
    if device is None:
        device = find_parent_of_type(lambda op: isinstance(op, DeviceOp))

    assert int(device.device) == int(AIEDevice.npu), "only npu supported"

    neighbors = {}
    col, row = map(int, (tile.col, tile.row))
    if col > 0 and row > 0 and not (col, row) == (1, 1):
        neighbors[col - 1, row] = "west"

    if logical:
        # can connect/talk/dma access
        if row >= 3:
            neighbors[col, row - 1] = "south"
        if 2 <= row < 5:
            neighbors[col, row + 1] = "north"
    else:
        # physical ie actually on the lattice nearby
        if row >= 1:
            neighbors[col, row - 1] = "south"
        if 0 < row < 5:
            neighbors[col, row + 1] = "north"

    neighbors_ = {"north": None, "west": None, "south": None}

    for n in find_ops(
        device,
        lambda op: isinstance(op, TileOp) and (int(op.col), int(op.row)) in neighbors,
    ):
        neighbors_[neighbors[int(n.col), int(n.row)]] = n

    return Neighbors(**neighbors_)


@_cext.register_operation(_Dialect, replace=True)
class TileOp(TileOp):
    def __str__(self):
        return f"tile(col={self.col.value}, row={self.row.value})"

    def __repr__(self):
        return str(self.operation)

    def __lt__(self, other):
        return tuple(map(int, (self.col, self.row))) < tuple(
            map(int, (other.col, other.row))
        )

    def __eq__(self, other):
        return tuple(map(int, (self.col, self.row))) == tuple(
            map(int, (other.col, other.row))
        )

    def __hash__(self):
        return hash((self.col, self.row))

    def flows(
        self, filter_source=False, filter_dest=False, source_annot=None, dest_annot=None
    ):
        return find_matching_flows(
            [self],
            filter_source=filter_source,
            filter_dest=filter_dest,
            source_annot=None,
            dest_annot=None,
        )

    def locks(self, sym_name=None, annot=None, device=None):
        return find_matching_locks(
            [self], sym_name=sym_name, annot=annot, device=device
        )


def tile(col, row, *, loc=None, ip=None):
    return TileOp(col=col, row=row, loc=loc, ip=ip)
