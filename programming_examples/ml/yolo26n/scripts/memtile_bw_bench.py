"""Memtile -> compute-tile weight-DMA bandwidth microbenchmark.

Goal: measure sustained MM2S bandwidth from a memtile to a compute tile under
realistic chunked-DMA + multi-sample-dispatch conditions, with and without a
second concurrent stream on a different (memtile, compute-tile) pair.

Design (single stream, BENCH_MODE=single):
  - memtile (5,1) holds a 256 KB int8 buffer (the "weights").
  - compute tile (5,3) -- same physical placement as m8's m_0_split tile --
    holds a small recv buffer, and a worker that acquires/releases each chunk
    N_CHUNKS times per outer iter. The worker does ZERO compute (no kernel),
    so wall time should be DMA-bound.
  - StaticWeightStream's `repeat_count = N_OUTER` means the memtile-side BD
    chain fires N_OUTER * N_CHUNKS times per dispatch.
  - Multi-sample dispatch via TensorAccessPattern: N_SAMPLES samples per
    `rt.run`, each carrying 1 unit of dummy IO that gates the kernel start.
  - `result.npu_time` gives total elapsed; bandwidth = total_bytes / time.

Design (dual stream, BENCH_MODE=dual):
  - Adds a second stream memtile (4,1) -> compute (4,5), same payload size and
    chunking. Both compute workers run concurrently per sample. Per-stream BW
    drops if memtile MM2S is shared / arbitrated under contention.

Build:
  make memtile_bw_bench           # builds xclbin + insts
  make run_memtile_bw_bench       # runs single + dual + prints MB/s

Or manually:
  python3 scripts/memtile_bw_bench.py --mode single > build/aie_bw_single.mlir
  cd build && aiecc.py --aie-generate-xclbin ... aie_bw_single.mlir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern

from lowlevel_dma import StaticWeightStream  # noqa: E402

# ---- Knobs (also overridable via env in the runner script) ----
PAYLOAD_BYTES = 256 * 1024  # 256 KB per dispatch per stream
N_CHUNKS = 8                # chunk count (memtile BD-chain depth = 1, but
                            # consumer acquires N_CHUNKS times per outer iter)
N_OUTER = 1                 # repeat_count on StaticWeightStream per sample;
                            # total chunks per sample = N_OUTER * N_CHUNKS
N_SAMPLES = 100             # samples per dispatch (TAP replay). Must be <=255
                            # (HW repeat_count cap).


def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def _i32(shape):
    return np.ndarray[shape, np.dtype[np.int32]]


def build(mode: str) -> str:
    """Build the bench MLIR.

    mode: "single" or "dual"
    Returns resolved MLIR as a string.
    """
    assert mode in ("single", "dual"), mode
    assert PAYLOAD_BYTES % N_CHUNKS == 0
    chunk_sz = PAYLOAD_BYTES // N_CHUNKS

    # Deterministic dummy payload (content irrelevant — kernel does nothing).
    data_a = (np.arange(PAYLOAD_BYTES, dtype=np.int32) & 0x7F).astype(np.int8)

    # Stream A: memtile (5,1) -> compute (5,3)
    t_compute_a = Tile(5, 3)
    t_memtile_a = Tile(5, 1)

    # Dummy activation input fifo per compute tile. Carries 1 i32 word per
    # sample. The fifo's sole purpose is to gate the worker loop -- one
    # iteration per shim-fill so the worker executes N_SAMPLES times. Total
    # DMA on this fifo is trivial (4 bytes/sample) so it does NOT inflate
    # the BW measurement.
    act_in_a = ObjectFifo(_i32((1,)), depth=2, name="bw_act_in_a")
    # Dummy output fifo so the worker has something to drain to the shim and
    # the runtime sequence has a `wait=True` drain to close the loop.
    act_out_a = ObjectFifo(_i32((1,)), depth=2, via_DMA=True, name="bw_act_out_a")

    ws_a = StaticWeightStream(
        obj_type=_i8((PAYLOAD_BYTES,)),
        initial_value=data_a,
        name="bw_wts_a",
        recv_type=_i8((chunk_sz,)),
        repeat_count=N_OUTER,
        memtile_placement=t_memtile_a,
        compute_placement=t_compute_a,
        mem_lock_id=0,
        comp_lock_id=0,
        mm2s_channel=0,
        s2mm_channel=0,
    )

    def worker_a_fn(act_in_c, ws, act_out_p):
        # One outer iter per sample. Each outer iter acquires N_OUTER
        # times the N_CHUNKS chunks. Kernel-less: just acquire/release
        # so DMA arrival is the only gating factor.
        gate = act_in_c.acquire(1)
        for _ in range_(N_OUTER):
            for _ in range_(N_CHUNKS):
                _ = ws.acquire(1)
                ws.release(1)
        out = act_out_p.acquire(1)
        out[0] = gate[0]
        act_in_c.release(1)
        act_out_p.release(1)

    workers = [
        Worker(
            worker_a_fn,
            fn_args=[act_in_a.cons(), ws_a, act_out_a.prod()],
            tile=t_compute_a,
            dynamic_objfifo_lowering=True,
        )
    ]

    # ---- Optional second stream ----
    act_in_b = None
    act_out_b = None
    if mode == "dual":
        t_compute_b = Tile(4, 5)  # same column as m8 cv1 / cv2
        t_memtile_b = Tile(4, 1)
        data_b = (np.arange(PAYLOAD_BYTES, dtype=np.int32) & 0x5A).astype(np.int8)
        act_in_b = ObjectFifo(_i32((1,)), depth=2, name="bw_act_in_b")
        act_out_b = ObjectFifo(_i32((1,)), depth=2, via_DMA=True, name="bw_act_out_b")
        ws_b = StaticWeightStream(
            obj_type=_i8((PAYLOAD_BYTES,)),
            initial_value=data_b,
            name="bw_wts_b",
            recv_type=_i8((chunk_sz,)),
            repeat_count=N_OUTER,
            memtile_placement=t_memtile_b,
            compute_placement=t_compute_b,
            mem_lock_id=0,
            comp_lock_id=0,
            mm2s_channel=0,
            s2mm_channel=0,
        )

        def worker_b_fn(act_in_c, ws, act_out_p):
            gate = act_in_c.acquire(1)
            for _ in range_(N_OUTER):
                for _ in range_(N_CHUNKS):
                    _ = ws.acquire(1)
                    ws.release(1)
            out = act_out_p.acquire(1)
            out[0] = gate[0]
            act_in_c.release(1)
            act_out_p.release(1)

        workers.append(
            Worker(
                worker_b_fn,
                fn_args=[act_in_b.cons(), ws, act_out_b.prod()],
                tile=t_compute_b,
                dynamic_objfifo_lowering=True,
            )
        )

    # ---- Runtime sequence: N_SAMPLES dispatch via TAP replay ----
    rt = Runtime()
    # Total IO is trivial -- 1 i32 in + 1 i32 out per sample per stream.
    in_a_ty = _i32((N_SAMPLES,))
    out_a_ty = _i32((N_SAMPLES,))
    in_b_ty = _i32((N_SAMPLES,))
    out_b_ty = _i32((N_SAMPLES,))

    # Shim tile slots: input on col 0 row 0, output on col 7 row 0.
    # Use distinct shim cols per stream to avoid same-column shim contention.
    shim_in_a = Tile(0, 0)
    shim_out_a = Tile(7, 0)
    shim_in_b = Tile(1, 0)
    shim_out_b = Tile(6, 0)

    if mode == "single":
        with rt.sequence(in_a_ty, out_a_ty) as (in_a, out_a):
            rt.start(*workers)
            tg = rt.task_group()
            in_tap = TensorAccessPattern(
                (N_SAMPLES,), offset=0, sizes=[N_SAMPLES, 1, 1, 1],
                strides=[1, 0, 0, 1],
            )
            out_tap = TensorAccessPattern(
                (N_SAMPLES,), offset=0, sizes=[N_SAMPLES, 1, 1, 1],
                strides=[1, 0, 0, 1],
            )
            rt.fill(act_in_a.prod(), in_a, tap=in_tap, tile=shim_in_a, task_group=tg)
            rt.drain(act_out_a.cons(), out_a, tap=out_tap, wait=True,
                     tile=shim_out_a, task_group=tg)
            rt.finish_task_group(tg)
    else:
        with rt.sequence(in_a_ty, out_a_ty, in_b_ty, out_b_ty) as (
            in_a, out_a, in_b, out_b
        ):
            rt.start(*workers)
            tg = rt.task_group()
            in_tap = TensorAccessPattern(
                (N_SAMPLES,), offset=0, sizes=[N_SAMPLES, 1, 1, 1],
                strides=[1, 0, 0, 1],
            )
            out_tap = TensorAccessPattern(
                (N_SAMPLES,), offset=0, sizes=[N_SAMPLES, 1, 1, 1],
                strides=[1, 0, 0, 1],
            )
            rt.fill(act_in_a.prod(), in_a, tap=in_tap, tile=shim_in_a, task_group=tg)
            rt.fill(act_in_b.prod(), in_b, tap=in_tap, tile=shim_in_b, task_group=tg)
            rt.drain(act_out_a.cons(), out_a, tap=out_tap, wait=True,
                     tile=shim_out_a, task_group=tg)
            rt.drain(act_out_b.cons(), out_b, tap=out_tap, wait=True,
                     tile=shim_out_b, task_group=tg)
            rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "dual"], required=True)
    args = p.parse_args()
    print(build(args.mode))


if __name__ == "__main__":
    main()
