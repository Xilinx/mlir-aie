"""Probe: runtime-indexed DMA gather (host-fed offset).

Goal: prove that a DMA can fetch row `idx` from a table resident in DDR, where
`idx` is a RUNTIME value (not compile-time) -- the core primitive for on-chip
embed gather (token id -> embedding row) and, later, len+value tokenizer gather.

This first version is HOST-FED: the index arrives as a 1-element runtime arg,
and the gather DMA's source offset is driven by that runtime value (idx*ROW).
A passthrough compute tile copies the gathered row to the output; the host
checks out == table[idx].

If npu_dma_memcpy_nd's dynamic-offset path lowers to a runtime-patched BD, this
validates the mechanism. Env: LLAMA_GATHER_NROWS, LLAMA_GATHER_ROW.
"""

import os as _os
import sys

import numpy as np
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_

NROWS = int(_os.environ.get("LLAMA_GATHER_NROWS", "8"))
ROW = int(_os.environ.get("LLAMA_GATHER_ROW", "64"))
dtype = np.int32


def design():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            table_ty = np.ndarray[(NROWS * ROW,), np.dtype[dtype]]
            row_ty = np.ndarray[(ROW,), np.dtype[dtype]]
            idx_ty = np.ndarray[(1,), np.dtype[np.int32]]

            ShimTile = tile(0, 0)
            ComputeTile = tile(0, 2)
            fifo_in = object_fifo("fifo_in", ShimTile, ComputeTile, 2, row_ty)
            fifo_out = object_fifo("fifo_out", ComputeTile, ShimTile, 2, row_ty)

            @core(ComputeTile)
            def core_body():
                for _ in range_(0, 0xFFFFFFFF):
                    ec = fifo_out.acquire(ObjectFifoPort.Produce, 1)
                    ea = fifo_in.acquire(ObjectFifoPort.Consume, 1)
                    for i in range_(ROW):
                        ec[i] = ea[i]
                    fifo_in.release(ObjectFifoPort.Consume, 1)
                    fifo_out.release(ObjectFifoPort.Produce, 1)

            # idx arrives as a runtime arg; the gather offset = idx[0]*ROW.
            @runtime_sequence(table_ty, idx_ty, row_ty)
            def sequence(table, idx, out):
                # dynamic offset: read idx[0] as an SSA value -> offset elems.
                from aie.dialects import memref, arith
                from aie.ir import IndexType, IntegerType

                idxt = IndexType.get()
                i64 = IntegerType.get_signless(64)
                c0 = arith.constant(idxt, 0)
                idx_v = memref.load(idx, [c0])  # i32
                idx_64 = arith.extsi(i64, idx_v)  # i32 -> i64
                row_c = arith.constant(i64, ROW)
                off = arith.muli(idx_64, row_c)  # i64 element offset
                npu_dma_memcpy_nd(
                    metadata=fifo_in,
                    bd_id=1,
                    mem=table,
                    offsets=[0, 0, 0, off],
                    sizes=[1, 1, 1, ROW],
                    strides=[0, 0, 0, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=fifo_out,
                    bd_id=0,
                    mem=out,
                    sizes=[1, 1, 1, ROW],
                    strides=[0, 0, 0, 1],
                )
                dma_wait(fifo_out)

    print(ctx.module)


if __name__ == "__main__":
    design()
