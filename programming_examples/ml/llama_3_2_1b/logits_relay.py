"""Hand-written memtile DMA relay for the M3a lm_head->sampler logits bridge.

Why hand-written (not ObjectFifo): the bridge needs a 256 KB HALF of fp32 logits
RESIDENT on a memtile, written once by the lm_head GEMM (a compute tile), then
re-read THREE times by the sampler in CHUNK-sized windows. The IRON ObjectFifo
lowering can't express this:
  - a single HALF buffer (needed for HW repeat_count, 1 BD) forces the consumer
    to acquire 256 KB, which overflows the 64 KB compute-tile L1;
  - a multi-buffer chunk fifo (depth=CHUNKS_PER_HALF) overflows the memtile's
    24-BD-per-channel budget (and x repeat).
So we drop to placed-dialect ops (the yolo26n/mobilenet lowlevel_dma.py pattern),
which lets one MM2S BD stream the resident HALF with a hardware repeat_count
while the consumer S2MM pulls CHUNK windows.

Geometry per relay (one per logits half):
  GEMM compute tile --MM2S--> [memtile HALF buffer] --MM2S(repeat=R)--> sampler
                       (fill)        resident             (replay, chunked)

Locks:
  fill_done : released by the GEMM-side mem DMA after the whole HALF is written;
              acquired by the replay MM2S before its first send.
  Standard prod/cons lock pair on the sampler S2MM recv buffer gates chunk flow.

This is a Resolvable: pass instances as Worker fn_args; Program.resolve() walks
them. The GEMM worker calls .produce_chunk() handles; the sampler worker calls
.acquire()/.release() to pull chunks.
"""

import numpy as np

from aie.iron.resolvable import Resolvable


class LogitsHalfRelay(Resolvable):
    """One half of the logits memtile relay (GEMM-written, replayed R times).

    Args:
      half_elems: fp32 elements in the half (e.g. V//2 = 64128).
      chunk_elems: consumer read granularity (e.g. CHUNK_N = 2004).
      repeat_count: how many times the sampler re-reads the whole half (3).
      memtile_placement / gemm_placement / sampler_placement: Tile objects.
      gemm_chunk_elems: producer write granularity (the GEMM writes the half in
        these pieces; defaults to chunk_elems).
      *_channel / *_lock_id: explicit DMA channel + lock IDs (avoid collisions).
    """

    def __init__(
        self,
        half_elems: int,
        chunk_elems: int,
        repeat_count: int,
        memtile_placement,
        gemm_placement,
        sampler_placement,
        name: str,
        gemm_chunk_elems: int | None = None,
        gemm_mm2s_channel: int = 0,
        fill_s2mm_channel: int = 0,
        replay_mm2s_channel: int = 1,
        sampler_s2mm_channel: int = 0,
        mem_lock_id: int = 0,
        gemm_lock_id: int = 0,
        sampler_lock_id: int = 0,
    ):
        self._half = half_elems
        self._chunk = chunk_elems
        self._gemm_chunk = gemm_chunk_elems or chunk_elems
        self._R = repeat_count
        self._memtile = memtile_placement
        self._gemm = gemm_placement
        self._sampler = sampler_placement
        self._name = name
        self._gemm_mm2s_ch = gemm_mm2s_channel
        self._fill_s2mm_ch = fill_s2mm_channel
        self._replay_mm2s_ch = replay_mm2s_channel
        self._sampler_s2mm_ch = sampler_s2mm_channel
        self._mem_lock_id = mem_lock_id
        self._gemm_lock_id = gemm_lock_id
        self._sampler_lock_id = sampler_lock_id

        # Set by resolve().
        self._gemm_send_buf = None
        self._gemm_prod_lock = None
        self._gemm_cons_lock = None
        self._recv_buf = None
        self._samp_prod_lock = None
        self._samp_cons_lock = None

    def tiles(self) -> list:
        return [self._memtile, self._gemm, self._sampler]

    # ---- GEMM producer side: write one chunk into the send buffer ----
    def gemm_acquire(self):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._gemm_cons_lock, LockAction.AcquireGreaterEqual)
        return self._gemm_send_buf

    def gemm_release(self):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._gemm_prod_lock, LockAction.Release)

    # ---- sampler consumer side: pull one CHUNK window ----
    def acquire(self, n: int = 1):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._samp_cons_lock, LockAction.AcquireGreaterEqual)
        return self._recv_buf

    def release(self, n: int = 1):
        from aie.dialects.aie import use_lock, LockAction

        use_lock(self._samp_prod_lock, LockAction.Release)

    def resolve(self, loc=None, ip=None) -> None:
        from aie.dialects.aie import (
            buffer,
            lock,
            flow,
            memtile_dma,
            mem,
            dma_start,
            dma_bd,
            next_bd,
            use_lock,
            DMAChannelDir,
            LockAction,
            WireBundle,
            EndOp,
        )

        f32 = np.float32
        half_ty = np.ndarray[(self._half,), np.dtype[f32]]
        chunk_ty = np.ndarray[(self._chunk,), np.dtype[f32]]
        gchunk_ty = np.ndarray[(self._gemm_chunk,), np.dtype[f32]]

        mem_op = self._memtile.op
        gemm_op = self._gemm.op
        samp_op = self._sampler.op

        n_gemm_chunks = self._half // self._gemm_chunk
        n_cons_chunks = self._half // self._chunk

        # --- Memtile: the resident HALF buffer ---
        half_buf = buffer(mem_op, half_ty, f"{self._name}_half")
        # fill_done: 0 until the GEMM has written the whole half; the replay
        # MM2S waits on it. The GEMM-side fill S2MM releases it n_gemm_chunks
        # times (once per chunk); replay acquires n_gemm_chunks to confirm full.
        fill_lock = lock(mem_op, lock_id=self._mem_lock_id, init=0)
        # replay_done: lets the fill S2MM know the memtile is free again (single
        # dispatch -> not strictly needed; init high so fill never blocks).
        free_lock = lock(mem_op, lock_id=self._mem_lock_id + 1, init=1)

        # --- GEMM compute tile: small send buffer (one chunk) ---
        gemm_prod_lock = lock(gemm_op, lock_id=self._gemm_lock_id, init=1)
        gemm_cons_lock = lock(gemm_op, lock_id=self._gemm_lock_id + 1, init=1)
        gemm_send_buf = buffer(gemm_op, gchunk_ty, f"{self._name}_gsend")
        self._gemm_send_buf = gemm_send_buf
        self._gemm_prod_lock = gemm_prod_lock
        self._gemm_cons_lock = gemm_cons_lock

        # --- Sampler compute tile: small recv buffer (one chunk) ---
        samp_prod_lock = lock(samp_op, lock_id=self._sampler_lock_id, init=1)
        samp_cons_lock = lock(samp_op, lock_id=self._sampler_lock_id + 1, init=0)
        recv_buf = buffer(samp_op, chunk_ty, f"{self._name}_recv")
        self._recv_buf = recv_buf
        self._samp_prod_lock = samp_prod_lock
        self._samp_cons_lock = samp_cons_lock

        # --- Flows: GEMM MM2S -> memtile S2MM (fill); memtile MM2S -> sampler
        #     S2MM (replay) ---
        flow(gemm_op, WireBundle.DMA, self._gemm_mm2s_ch,
             mem_op, WireBundle.DMA, self._fill_s2mm_ch)
        flow(mem_op, WireBundle.DMA, self._replay_mm2s_ch,
             samp_op, WireBundle.DMA, self._sampler_s2mm_ch)

        # --- GEMM compute tile DMA: MM2S send each chunk ---
        @mem(gemm_op)
        def _gdma(block):
            dma_start(DMAChannelDir.MM2S, self._gemm_mm2s_ch,
                      dest=block[1], chain=block[2])
            with block[1]:
                use_lock(gemm_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(gemm_send_buf)
                use_lock(gemm_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()

        # --- Memtile DMA: fill S2MM (gather chunks into half_buf) +
        #     replay MM2S (stream the resident half, HW repeat R, chunked) ---
        @memtile_dma(mem_op)
        def _mtdma(block):
            # S2MM fill: write incoming GEMM chunks into half_buf at successive
            # offsets; release fill_lock once per chunk.
            s2 = dma_start(DMAChannelDir.S2MM, self._fill_s2mm_ch,
                           dest=block[1], chain=block[2])
            with block[1]:
                use_lock(free_lock, LockAction.AcquireGreaterEqual)
                dma_bd(half_buf)  # whole-half S2MM; GEMM streams n_gemm_chunks
                use_lock(fill_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                # MM2S replay: send the whole resident half, HW repeat R-1 extra.
                dma_start(DMAChannelDir.MM2S, self._replay_mm2s_ch,
                          dest=block[3], chain=block[4], repeat_count=self._R - 1)
                with block[3]:
                    use_lock(fill_lock, LockAction.AcquireGreaterEqual)
                    dma_bd(half_buf)
                    next_bd(block[3])
                with block[4]:
                    EndOp()

        # --- Sampler compute tile DMA: S2MM pull each CHUNK window ---
        @mem(samp_op)
        def _sdma(block):
            dma_start(DMAChannelDir.S2MM, self._sampler_s2mm_ch,
                      dest=block[1], chain=block[2])
            with block[1]:
                use_lock(samp_prod_lock, LockAction.AcquireGreaterEqual)
                dma_bd(recv_buf)
                use_lock(samp_cons_lock, LockAction.Release)
                next_bd(block[1])
            with block[2]:
                EndOp()
