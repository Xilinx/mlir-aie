"""Silicon-level probe for AIE-ML BD-level Enable_Compression.

Single-tile passthrough on Phoenix npu1 (shim -> compute(0,2) -> shim).
Four configs flip the compute-tile DMA compression registers via
npu_maskwrite32 from the runtime sequence; everything else is the
standard passthrough_dmas_placed.py pattern.

BD layout on tile (0,2) after objectfifo lowering (verified via
aie-opt --aie-objectFifo-stateful-transform --aie-assign-bd-ids):
  BD 0,1 -> S2MM ch 0 (incoming from shim)
  BD 2,3 -> MM2S ch 0 (outgoing to shim)

Register addresses (tile-relative, AIE-ML compute-tile MEM module,
from third_party/aie-rt/driver/src/global/xaiemlgbl_params.h):
  DMA_BD0_1 = 0x1D004, BD1_1 = 0x1D024, BD2_1 = 0x1D044, BD3_1 = 0x1D064
    bit 31 = Enable_Compression
  DMA_S2MM_0_CTRL = 0x1DE00, bit 4 = Decompression_Enable
  DMA_MM2S_0_CTRL = 0x1DE10, bit 4 = Compression_Enable    (symmetric)
"""

import sys

import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

N = 4096
IN_N = None  # bytes the shim MM2S BD should send (in int32 units); defaults to N
OUT_N = None  # bytes the shim S2MM BD should expect (in int32 units); defaults to N
LINE = 1024
COL = 0
ROW = 2

BD_S2MM = [0, 1]
BD_MM2S = [2, 3]
BD1_BASE = 0x1D004
BD_STRIDE = 0x20
S2MM0_CTRL = 0x1DE00
MM2S0_CTRL = 0x1DE10
COMPRESS_BIT = 0x80000000
CHAN_BIT = 0x10  # bit 4 in both *_CTRL regs

CONFIGS = ("base", "cmp_only", "dcmp_only", "both")


def emit(config: str, in_n: int, out_n: int):
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; pick from {CONFIGS}")

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
            vec_ty = np.ndarray[(N,), np.dtype[np.int32]]

            shim = tile(COL, 0)
            ct = tile(COL, ROW)

            of_in = object_fifo("in", shim, ct, 2, line_ty)
            of_out = object_fifo("out", ct, shim, 2, line_ty)
            object_fifo_link(of_in, of_out)

            @core(ct)
            def core_body():
                for _ in range_(sys.maxsize):
                    pass

            @runtime_sequence(vec_ty, vec_ty, vec_ty)
            def sequence(A, B, C):
                if config in ("cmp_only", "both"):
                    for bd in BD_MM2S:
                        npu_maskwrite32(
                            column=COL,
                            row=ROW,
                            address=BD1_BASE + bd * BD_STRIDE,
                            value=COMPRESS_BIT,
                            mask=COMPRESS_BIT,
                        )
                    npu_maskwrite32(
                        column=COL,
                        row=ROW,
                        address=MM2S0_CTRL,
                        value=CHAN_BIT,
                        mask=CHAN_BIT,
                    )
                if config in ("dcmp_only", "both"):
                    for bd in BD_S2MM:
                        npu_maskwrite32(
                            column=COL,
                            row=ROW,
                            address=BD1_BASE + bd * BD_STRIDE,
                            value=COMPRESS_BIT,
                            mask=COMPRESS_BIT,
                        )
                    npu_maskwrite32(
                        column=COL,
                        row=ROW,
                        address=S2MM0_CTRL,
                        value=CHAN_BIT,
                        mask=CHAN_BIT,
                    )

                in_task = shim_dma_single_bd_task(
                    of_in, A, sizes=[1, 1, 1, in_n], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, out_n], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    print(ctx.module)


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "base"
    in_n = int(sys.argv[2]) if len(sys.argv) > 2 else N
    out_n = int(sys.argv[3]) if len(sys.argv) > 3 else N
    emit(cfg, in_n, out_n)
