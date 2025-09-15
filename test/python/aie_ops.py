# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy as np

# RUN: %PYTHON %s | FileCheck %s


from aie.dialects.aie import (
    AIEDevice,
    Core,
    Device,
    MemOp,
    ObjectFifoPort,
    buffer,
    external_buffer,
    bd_dim_layout,
    end,
    object_fifo,
    object_fifo_link,
    tile,
    cascade_flow,
    WireBundle,
    packetflow,
    get_target_model,
    dma_bd,
)
from aie.ir import InsertionPoint, Block
from aie.extras.context import mlir_mod_ctx
from aie.extras import types as T
from util import construct_and_print_module


# CHECK-LABEL: tileOp
# CHECK: aie.tile(0, 0)
@construct_and_print_module
def tileOp():
    t = tile(col=0, row=0)


# CHECK-LABEL: tileOpAllocationScheme
# CHECK: aie.tile(2, 2) {allocation_scheme = "basic-sequential"}
@construct_and_print_module
def tileOpAllocationScheme():
    t = tile(col=2, row=2, allocation_scheme="basic-sequential")


# CHECK-LABEL: coreOp
# CHECK: %[[VAL1:.*]] = aie.tile(1, 1)
# CHECK: %[[VAL2:.*]] = aie.core(%[[VAL1]]) {
# CHECK:   aie.end
# CHECK: }
@construct_and_print_module
def coreOp():
    t = tile(col=1, row=1)
    c = Core(t)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: coreOpParameters
# CHECK: %[[VAL1:.*]] = aie.tile(1, 1)
# CHECK: %[[VAL2:.*]] = aie.core(%[[VAL1]]) {
# CHECK:   aie.end
# CHECK: } {dynamic_objfifo_lowering = false, link_with = "test.elf", stack_size = 2048 : i32}
@construct_and_print_module
def coreOpParameters():
    t = tile(col=1, row=1)
    c = Core(t, link_with="test.elf", dynamic_objfifo_lowering=False, stack_size=2048)
    bb = Block.create_at_start(c.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: memOp
# CHECK: %[[VAL1:.*]] = aie.tile(2, 2)
# CHECK: %[[VAL2:.*]] = aie.mem(%[[VAL1]]) {
# CHECK:   aie.end
# CHECK: }
@construct_and_print_module
def memOp():
    t = tile(col=2, row=2)
    m = MemOp(T.index(), t)
    assert isinstance(m.result.owner.opview, MemOp)
    bb = Block.create_at_start(m.body)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: deviceOp
# CHECK: aie.device
@construct_and_print_module
def deviceOp():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        end()


# CHECK-LABEL: bufferOp
# CHECK: %[[VAL_0:.*]] = aie.tile(0, 3)
# CHECK: %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) : memref<12xi32>
# CHECK: %[[VAL_2:.*]] = aie.buffer(%[[VAL_0]]) : memref<2x2xi32> = dense<{{\[}}[0, 1], [2, 3]]>
# CHECK: %[[VAL_3:.*]] = aie.buffer(%[[VAL_0]]) {address = 48879 : i32} : memref<42xi8>
@construct_and_print_module
def bufferOp():
    t = tile(col=0, row=3)
    b = buffer(t, np.ndarray[(12,), np.dtype[np.int32]])
    b = buffer(
        t,
        T.memref(2, 2, T.i32()),
        initial_value=np.arange(2 * 2, dtype=np.int32).reshape(2, 2),
    )
    b = buffer(t, np.ndarray[(42,), np.dtype[np.int8]], address=0xBEEF)


# CHECK-LABEL: externalBufferOp
# CHECK: %[[VAL_0:.*]] = aie.external_buffer : memref<12xi32>
# CHECK: %[[VAL_1:.*]] = aie.external_buffer {address = 209934011881080 : i64} : memref<13xi8>
@construct_and_print_module
def externalBufferOp():
    b = external_buffer(T.memref(12, T.i32()))
    c = external_buffer(T.memref(13, T.i8()), address=0xBEEF12345678)


# CHECK-LABEL: objFifo
# CHECK: %[[VAL0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @of0(%[[VAL0]] dimensionsToStream [<size = 1, stride = 2>], {%[[VAL1]] dimensionsFromStream [<size = 1, stride = 2>]}, 2 : i32) {via_DMA = true} : !aie.objectfifo<memref<4xf16>> = [dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : memref<4xf16>, dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00]> : memref<4xf16>]
@construct_and_print_module
def objFifo():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        object_fifo(
            "of0",
            tile0,
            tile1,
            2,
            np.ndarray[(4,), np.dtype[np.float16]],
            [bd_dim_layout(size=1, stride=2)],
            [[bd_dim_layout(size=1, stride=2)]],
            via_DMA=True,
            initValues=[np.arange(4, dtype=np.float16), np.arange(4, dtype=np.float16)],
        )
        end()


# CHECK-LABEL: objFifoLink
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 3)
# CHECK: %[[VAL_1:.*]] = aie.tile(6, 1)
# CHECK: %[[VAL_2:.*]] = aie.tile(7, 3)
# CHECK: aie.objectfifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo @[[VAL_4:.*]](%[[VAL_1]], {%[[VAL_2]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo.link [@[[VAL_3]]] -> [@[[VAL_4]]]([] [])
@construct_and_print_module
def objFifoLink():
    dev = Device(AIEDevice.xcve2302)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=3)
        tile1 = tile(col=6, row=1)
        tile2 = tile(col=7, row=3)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        of1 = object_fifo("of1", tile1, tile2, 2, T.memref(12, T.f16()))
        object_fifo_link(of0, of1, [], [])
        end()


# CHECK-LABEL: objFifoAcquire
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = aie.objectfifo.acquire @[[VAL_2]](Consume, 1) : !aie.objectfifosubview<memref<12xf16>>
@construct_and_print_module
def objFifoAcquire():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = of0.acquire(port=ObjectFifoPort.Consume, num_elem=1)
            end()
        end()


# CHECK-LABEL: objFifoSubviewAccess
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: %[[VAL_3:.*]] = aie.objectfifo.acquire @[[VAL_2]](Consume, 1) : !aie.objectfifosubview<memref<12xf16>>
# CHECK: %[[VAL_4:.*]] = aie.objectfifo.subview.access %[[VAL_3]][0] : !aie.objectfifosubview<memref<12xf16>> -> memref<12xf16>
@construct_and_print_module
def objFifoSubviewAccess():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo(
            "of0", tile0, tile1, 2, np.ndarray[(12,), np.dtype[np.float16]]
        )
        C = Core(tile1)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = of0.acquire(ObjectFifoPort.Consume, 1)
            end()
        end()


# CHECK-LABEL: objFifoRelease
# CHECK: %[[VAL_0:.*]] = aie.tile(6, 6)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 2)
# CHECK: aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !aie.objectfifo<memref<12xf16>>
# CHECK: aie.objectfifo.release @[[VAL_2]](Produce, 1)
@construct_and_print_module
def objFifoRelease():
    dev = Device(AIEDevice.xcvc1902)
    bb = Block.create_at_start(dev.body_region)
    with InsertionPoint(bb):
        tile0 = tile(col=6, row=6)
        tile1 = tile(col=2, row=2)
        of0 = object_fifo("of0", tile0, tile1, 2, T.memref(12, T.f16()))
        C = Core(tile0)
        bb = Block.create_at_start(C.body)
        with InsertionPoint(bb):
            acq = of0.release(ObjectFifoPort.Produce, 1)
            end()
        end()


# CHECK-LABEL: cascadeFlowOp
# CHECK: %[[VAL_0:.*]] = aie.tile(1, 3)
# CHECK: %[[VAL_1:.*]] = aie.tile(2, 3)
# CHECK: aie.cascade_flow(%[[VAL_0]], %[[VAL_1]])
@construct_and_print_module
def cascadeFlowOp():
    t0 = tile(col=1, row=3)
    t1 = tile(col=2, row=3)
    cascade_flow(t0, t1)


# CHECK-LABEL: packetFlowOp
# CHECK: %[[VAL_0:.*]] = aie.tile(1, 3)
# CHECK: aie.packet_flow(16) {
# CHECK:   aie.packet_source<%[[VAL_0]], Core : 0>
# CHECK:   aie.packet_dest<%[[VAL_0]], Core : 0>
# CHECK: } {keep_pkt_header = true}
@construct_and_print_module
def packetFlowOp():
    t0 = tile(col=1, row=3)
    packetflow(
        pkt_id=0x10,
        source=t0,
        source_port=WireBundle.Core,
        source_channel=0,
        dest=t0,
        dest_port=WireBundle.Core,
        dest_channel=0,
        keep_pkt_header=True,
    )


# CHECK-LABEL: dmaBDOp
# CHECK: %[[VAL_0:.*]] = aie.tile(1, 3)
# CHECK: %[[VAL_1:.*]] = aie.buffer(%[[VAL_0]]) : memref<12xi32>
# CHECK: aie.dma_bd(%[[VAL_1]] : memref<12xi32>) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 4>}
@construct_and_print_module
def dmaBDOp():
    t0 = tile(col=1, row=3)
    b = buffer(t0, np.ndarray[(12,), np.dtype[np.int32]])
    dma_bd(b, packet=(0, 4))


# CHECK-LABEL: test_module_context
# CHECK: module {
# CHECK:   %tile_1_1 = aie.tile(1, 1)
# CHECK:   %core_1_1 = aie.core(%tile_1_1) {
# CHECK:     aie.end
# CHECK:   }
# CHECK: }
def test_module_context():
    print("test_module_context")

    with mlir_mod_ctx() as ctx:
        t = tile(col=1, row=1)
        c = Core(t)
        bb = Block.create_at_start(c.body)
        with InsertionPoint(bb):
            end()

    print(ctx.module)


test_module_context()


# CHECK-LABEL: test_target_model
# CHECK: xcvc1902 rows 9
# CHECK: xcvc1902 cols 50
# CHECK: xcvc1902 npu False
# CHECK: npu1 rows 6
# CHECK: npu1 cols 4
# CHECK: npu1 npu True
# CHECK: npu1_1col rows 6
# CHECK: npu1_1col cols 1
# CHECK: npu1_1col npu True
def test_target_model():
    print("test_target_model")
    for d in AIEDevice:
        tm = get_target_model(d)
        print(f"{d} rows {tm.rows()}")
        print(f"{d} cols {tm.columns()}")
        print(f"{d} npu {tm.is_npu()}")


test_target_model()
