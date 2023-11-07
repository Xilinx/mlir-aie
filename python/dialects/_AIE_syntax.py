from ._AIE_ops_gen import *
from ._AIE_enum_gen import *
from ._AIE_util import *
from .._mlir_libs._aieMlir import register_dialect, ObjectFifoType, ObjectFifoSubviewType


# Helper function that returns the index of a named objectFifo port.
def objectFifoPortToIndex(port):
    if (port == "Produce"):
        port_index = 0
    elif (port == "Consume"):
        port_index = 1
    else:
        port_index = -1
    return port_index


# Helper function that returns the index of a named WireBundle.
def wireBundleToIndex(port):
    if (port == "Core"):
        port_index = 0
    elif (port == "DMA"):
        port_index = 1
    elif (port == "FIFO"):
        port_index = 2
    elif (port == "South"):
        port_index = 3
    elif (port == "West"):
        port_index = 4
    elif (port == "North"):
        port_index = 5
    elif (port == "East"):
        port_index = 6
    elif (port == "PLIO"):
        port_index = 7
    elif (port == "NOC"):
        port_index = 8
    elif (port == "Trace"):
        port_index = 9
    else:
        port_index = -1
    return port_index


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
        if (link_with != None):
            super().__init__(result=idx_ty, tile=tile, link_with=link_with)
        else :    
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
    def __init__(self, name, tile0, tile1, depth, datatype, dimensionsToStream = [], dimensionsFromStreamPerConsumer = []):
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
            dimensionsFromStreamPerConsumer=dimensionsFromStreamPerConsumer
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
        ofIns_array = ArrayAttr.get(ofIns_sym)
        ofOuts_array = ArrayAttr.get(ofOuts_sym)
        super().__init__(fifoIns=ofIns_array, fifoOuts=ofOuts_array)


# Create an aie objectFifo acquire op of given number of elements with given memref datatype,
# from objFifo with given name.
class Acquire(ObjectFifoAcquireOp):
    """Specialize ObjectFifoAcquireOp class constructor to take python integers"""
    def __init__(self, of_name, port, num_elem, datatype):
        of_sym = FlatSymbolRefAttr.get(of_name)
        int_ty = IntegerType.get_signless(32)
        int_port = IntegerAttr.get(int_ty, objectFifoPortToIndex(port))
        subview_Ty = ObjectFifoSubviewType.get(datatype)
        self.datatype = datatype
        super().__init__(subview=subview_Ty, port=int_port, objFifo_name=of_sym, size=num_elem)

    def acquiredElem(self):
        objects = []
        if self.size.value == 1:
            return SubviewAccess(self.subview, self.size.value - 1, self.datatype)
        for i in range(self.size.value):
            objects.append(SubviewAccess(self.subview, i, self.datatype))
        return objects


# Create an aie objectFifo access op on given subview with given memref datatype,
# at given index.
class SubviewAccess(ObjectFifoSubviewAccessOp):
    """Rename ObjectFifoSubviewAccessOp class"""
    def __init__(self, subview, index, datatype):
        super().__init__(output=datatype, subview=subview, index=index)


# Create an aie objectFifo release op of given number of elements from objFifo with given name.
class Release(ObjectFifoReleaseOp):
    """Specialize ObjectFifoReleaseOp class constructor to take python integers"""
    def __init__(self, of_name, port, num_elem):
        of_sym = FlatSymbolRefAttr.get(of_name)
        int_ty = IntegerType.get_signless(32)
        int_port = IntegerAttr.get(int_ty, objectFifoPortToIndex(port))
        super().__init__(port=int_port, objFifo_name=of_sym, size=num_elem)


# Create a flow between source and destination tile ports.
class Flow(FlowOp):
    """Specialize FlowOp class constructor to take python integers"""
    def __init__(self, source, source_port, source_channel, dest, dest_port, dest_channel):
        int_ty = IntegerType.get_signless(32)
        sourceBundle = IntegerAttr.get(int_ty, wireBundleToIndex(source_port))
        destBundle = IntegerAttr.get(int_ty, wireBundleToIndex(dest_port))
        super().__init__(
            source=source, 
            sourceBundle=sourceBundle, 
            sourceChannel=source_channel, 
            dest=dest, 
            destBundle=destBundle, 
            destChannel=dest_channel
        )


# Create a packet flow between source and destination tile ports.
class PacketFlow(PacketFlowOp):
    """Specialize PacketFlowOp class constructor to take python integers"""
    def __init__(self, pkt_id, source, source_port, source_channel, dest, dest_port, dest_channel):
        int8_ty = IntegerType.get_signless(8)
        int_ty = IntegerType.get_signless(32)
        int8_pkt_id = IntegerAttr.get(int8_ty, pkt_id)
        sourceBundle = IntegerAttr.get(int_ty, wireBundleToIndex(source_port))
        destBundle = IntegerAttr.get(int_ty, wireBundleToIndex(dest_port))
        super().__init__(ID=int8_pkt_id)
        bb = Block.create_at_start(self.ports)
        with InsertionPoint(bb):
            src = PacketSourceOp(source, sourceBundle, source_channel)
            dest = PacketDestOp(dest, destBundle, dest_channel)
            end = EndOp()


#### Global Wrappers ####
core = region_op(Core, terminator=lambda *args: EndOp())
device = region_op(Device)
forLoop = region_op(For, terminator=lambda *args: YieldOp([]))
