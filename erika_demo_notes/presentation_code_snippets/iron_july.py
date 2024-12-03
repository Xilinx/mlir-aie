# https://github.com/Xilinx/mlir-aie/tree/b40e1792b5a4c38ac4addcffd5ca4b6f06ac1ac6
# https://github.com/Xilinx/mlir-aie/blob/b40e1792b5a4c38ac4addcffd5ca4b6f06ac1ac6/programming_examples/basic/matrix_scalar_add/aie2.py

# fmt: off
import sys
from aie.dialects.aie import tile, object_fifo, for_, core, T, AIEDevice, device, ObjectFifoPort, yield_, runtime_sequence, npu_dma_memcpy_nd, npu_sync
from aie.dialects.aiex import *
from aie.dialects.scf import *
from aie.extras.dialects.ext import memref, arith
from aie.extras.context import mlir_mod_ctx


# Size of the entire image to process
IMAGE_HEIGHT = 16
IMAGE_WIDTH = 128
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT

# Size of the tile to process
TILE_HEIGHT = 8
TILE_WIDTH = 16
TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

with mlir_mod_ctx() as ctx:
  @device(AIEDevice.npu1_1col)
  def device_body():
    # Types, tile declarations, and AIE data movement with object fifos
    tensor_ty = T.memref(TILE_SIZE, T.i32())
    ShimTile = tile(0, 0)
    ComputeTile2 = tile(0, 2)
    of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tensor_ty)
    of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tensor_ty)

    @core(ComputeTile2) # Set up compute tile 2
    def core_body():
      for _ in for_(sys.maxsize): # Effective while(1)
        elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
        elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
        for i in for_(TILE_SIZE):
          v0 = memref.load(elem_in, [i])
          v1 = arith.addi(v0, arith.constant(1, T.i32()))
          memref.store(v1, elem_out, [i])
          yield_([])
        of_in.release(ObjectFifoPort.Consume, 1)
        of_out.release(ObjectFifoPort.Produce, 1)
        yield_([])

    @runtime_sequence(tensor_ty, tensor_ty)
    def sequence(inTensor, outTensor):
      npu_dma_memcpy_nd(metadata="out", bd_id=0, mem=outTensor, 
            sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[1, 1, IMAGE_WIDTH, 1])
      npu_dma_memcpy_nd(metadata="in", bd_id=1, mem=inTensor, 
            sizes=[1, 1, TILE_HEIGHT, TILE_WIDTH], strides=[1, 1, IMAGE_WIDTH, 1])
      npu_sync(column=0, row=0, direction=0, channel=0)

  print(ctx.module) # Emit MLIR strings
# fmt: on
