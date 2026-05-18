"""IRON-style DMA compression probe on Phoenix npu1.

Single-tile passthrough (shim -> compute(0,2) -> shim) on Phoenix npu1.
The data path is DMA-only via `ObjectFifo.forward(tile=compute_tile)`.
The AIE-ML compute-tile DMA `Enable_Compression` and `*compression_Enable`
register pair is flipped from the host runtime sequence via
`aiex.npu.npu_maskwrite32`, surfaced through `Runtime.inline_ops`.

BD layout on tile (0,2) after IRON lowering:
  BD 0,1 -> S2MM ch 0 (incoming from shim)
  BD 2,3 -> MM2S ch 0 (outgoing to shim)

Configs (all with arange input, all complete in milliseconds):
  base       passthrough, no compression                       in=out=4096
  cmp_only   compression on MM2S only (output side)            in=4096, out=2944
  dcmp_only  decompression on S2MM only (input side)           in=2944, out=4096
  both       compress on output + decompress on input          in=4096, out=2944

The asymmetric configs use a precomputed ratio (~1.39x for arange 0..N-1
on Phoenix) so the shim BD lengths match the compressed byte count and
neither side hangs on a BD-length mismatch.
"""

import os
import sys

import numpy as np

import aie.iron as iron
from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.helpers.dialects.func import func
from aie.iron.device import Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.dialects._aie_enum_gen import AIETileType
from aie.dialects.aiex import npu_maskwrite32

N = 4096
LINE_SIZE = 1024
RATIOED_N = 2944  # empirical compressed byte count for arange(N) on Phoenix
RATIOED_PER_LINE = RATIOED_N // (N // LINE_SIZE)  # = 736; per-BD compressed

COL = 0
COMPUTE_ROW = 2
MEMTILE_ROW = 1

# Compute-tile (AIE-ML compute tile, row 2+) DMA register layout. The BD
# compression bit lives in BD?_1 (word 1, bit 31). Address stride is 0x20
# bytes between consecutive BD registers.
CT_BD1_BASE = 0x1D004
CT_S2MM0_CTRL = 0x1DE00
CT_MM2S0_CTRL = 0x1DE10

# Memtile (AIE-ML mem tile, row 1) DMA register layout. The BD compression
# bit lives in BD?_4 (word 4, bit 31). Same 0x20 stride.
MT_BD4_BASE = 0xA0010
MT_S2MM0_CTRL = 0xA0600
MT_MM2S0_CTRL = 0xA0630

BD_STRIDE = 0x20
COMPRESS_BIT = 0x80000000  # BD?_X bit 31 (X=1 on compute, X=4 on memtile)
CHAN_BIT = 0x10            # *_CTRL bit 4 (same on compute and memtile)
BD_S2MM = (0, 1)
BD_MM2S = (2, 3)

HOST_CONFIGS = ("base", "cmp_only", "dcmp_only", "both")
CORE_CONFIGS = ("core_cmp_only", "core_dcmp_only", "core_both")
MEMTILE_CONFIGS = ("memtile_base", "memtile_cmp_only", "memtile_dcmp_only", "memtile_both")
# Two-tile chains via shim -> CT(0,2) -> {memtile(0,1) | CT(0,3)} -> shim.
#   lossless_roundtrip       : compress on CT(0,2) MM2S + decompress on memtile
#                              S2MM — proves the cross-tile roundtrip is
#                              invertible on arbitrary input
#   multi_base               : passthrough through CT(0,2) + CT(0,3), no
#                              compression (sanity check that the topology
#                              works)
#   multi_cmp_only           : compress on CT(0,2) MM2S only; consumer BDs on
#                              CT(0,3) hand-sized to RATIOED_PER_LINE. Built
#                              via low-level aie.dialect (IRON's link API
#                              doesn't expose per-side BD sizing). Runnable
#                              asymmetric proof that compression engages on
#                              the inter-tile link.
#   multi_lossless_roundtrip : compress on CT(0,2) MM2S + decompress on CT(0,3)
#                              S2MM — full roundtrip on CT-to-CT link.
ROUNDTRIP_CONFIGS = (
    "lossless_roundtrip",
    "multi_base",
    "multi_cmp_only",
    "multi_lossless_roundtrip",
)
CONFIGS = HOST_CONFIGS + CORE_CONFIGS + MEMTILE_CONFIGS + ROUNDTRIP_CONFIGS

COMPUTE_ROW_2 = 3  # second compute tile for the multi-tile configs

# Address of Core_Processor_Bus enable (must be 1 before the core can do
# st.tm to the processor bus). Pattern from PR #2348's
# test/npu-xrt/tile_mapped_read/aie.mlir:61.
CORE_PROCESSOR_BUS_EN = 0x32038

_KERNEL_CC = os.path.join(os.path.dirname(__file__), "kernel.cc")

line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]


# Module-level @func — used by the `lossless_roundtrip` config's CT Worker.
# Has to live here (not inside the design function) because the @func
# decorator emits via the surrounding MLIR module context at module-load
# time and won't have one if defined inside a regular function.
@func
def passthrough_line(src: line_ty, dst: line_ty, n: np.int32):
    for i in range_(n):
        dst[i] = src[i]


def _maskwrite_compress(row, bd_base, bds, ctrl_addr):
    for bd in bds:
        npu_maskwrite32(
            column=COL,
            row=row,
            address=bd_base + bd * BD_STRIDE,
            value=COMPRESS_BIT,
            mask=COMPRESS_BIT,
        )
    npu_maskwrite32(
        column=COL,
        row=row,
        address=ctrl_addr,
        value=CHAN_BIT,
        mask=CHAN_BIT,
    )


def _linear_tap(n_elems):
    return TensorAccessPattern((1, N), 0, [1, 1, 1, n_elems], [0, 0, 0, 1])


def _build_multi_cmp_only():
    """Low-level dialect construction for multi_cmp_only.

    Asymmetric inter-tile compression: CT(0,2) MM2S compresses, CT(0,3)
    S2MM does NOT decompress. This means the wire carries fewer bytes
    per BD than the consumer expects. To avoid a stall, the consumer
    BD (and the shim S2MM BD that drains CT(0,3)) must be hand-sized to
    the empirically-measured per-BD compressed byte count.

    Topology: shim -> CT(0,2) -> CT(0,3) -> shim. All BDs explicit; no
    object_fifo. The CT(0,3) S2MM/MM2S buffers are sized RATIOED_PER_LINE
    so the BD length matches what the compressor actually emits.
    """
    from aie.dialects.aie import (
        device, tile, buffer, lock, mem, dma_start, dma_bd, next_bd,
        use_lock, flow, end as aie_end, core, AIEDevice, DMAChannelDir,
        LockAction, WireBundle, shim_dma_allocation,
    )
    from aie.extras.context import mlir_mod_ctx
    from aie.dialects.aiex import (
        runtime_sequence, shim_dma_single_bd_task, dma_start_task,
        dma_await_task,
    )

    raw_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]
    comp_ty = np.ndarray[(RATIOED_PER_LINE,), np.dtype[np.int32]]
    vec_ty_in = np.ndarray[(N,), np.dtype[np.int32]]
    vec_ty_out = np.ndarray[(RATIOED_N,), np.dtype[np.int32]]
    n_iters = N // LINE_SIZE

    with mlir_mod_ctx() as ctx:
        @device(AIEDevice.npu1_1col)
        def _dev():
            shim = tile(COL, 0)
            ct2 = tile(COL, COMPUTE_ROW)
            ct3 = tile(COL, COMPUTE_ROW_2)

            # Ping-pong buffers + locks on each compute tile
            ct2_buf0 = buffer(ct2, raw_ty, name="ct2_buf0")
            ct2_buf1 = buffer(ct2, raw_ty, name="ct2_buf1")
            ct2_full = lock(ct2, init=0, sym_name="ct2_full")
            ct2_empty = lock(ct2, init=2, sym_name="ct2_empty")

            ct3_buf0 = buffer(ct3, comp_ty, name="ct3_buf0")
            ct3_buf1 = buffer(ct3, comp_ty, name="ct3_buf1")
            ct3_full = lock(ct3, init=0, sym_name="ct3_full")
            ct3_empty = lock(ct3, init=2, sym_name="ct3_empty")

            # Flows
            flow(shim, WireBundle.DMA, 0, ct2, WireBundle.DMA, 0)
            flow(ct2, WireBundle.DMA, 0, ct3, WireBundle.DMA, 0)
            flow(ct3, WireBundle.DMA, 0, shim, WireBundle.DMA, 0)

            # Shim DMA alloc declarations for the runtime sequence symbols.
            shim_dma_allocation("in_alloc", shim, DMAChannelDir.MM2S, 0)
            shim_dma_allocation("out_alloc", shim, DMAChannelDir.S2MM, 0)

            # Cores must exist on used compute tiles (infinite spinners; data
            # path is DMA-only).
            @core(ct2)
            def _ct2_core():
                for _ in range_(0x7FFFFFFF):
                    pass

            @core(ct3)
            def _ct3_core():
                for _ in range_(0x7FFFFFFF):
                    pass

            # CT(0,2) mem block: S2MM receives raw 1024-int lines from shim
            # (cycling 2 BDs); MM2S compresses and emits.
            @mem(ct2)
            def _ct2_mem(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ct2_empty, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct2_buf0)
                    use_lock(ct2_full, LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(ct2_empty, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct2_buf1)
                    use_lock(ct2_full, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ct2_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct2_buf0)
                    use_lock(ct2_empty, LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ct2_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct2_buf1)
                    use_lock(ct2_empty, LockAction.Release, value=1)
                    next_bd(block[4])
                with block[6]:
                    aie_end()

            # CT(0,3) mem block: S2MM receives RATIOED_PER_LINE int worth of
            # bytes (the compressed stream from CT(0,2)); MM2S forwards to shim.
            @mem(ct3)
            def _ct3_mem(block):
                dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
                with block[1]:
                    use_lock(ct3_empty, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct3_buf0)
                    use_lock(ct3_full, LockAction.Release, value=1)
                    next_bd(block[2])
                with block[2]:
                    use_lock(ct3_empty, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct3_buf1)
                    use_lock(ct3_full, LockAction.Release, value=1)
                    next_bd(block[1])
                with block[3]:
                    dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
                with block[4]:
                    use_lock(ct3_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct3_buf0)
                    use_lock(ct3_empty, LockAction.Release, value=1)
                    next_bd(block[5])
                with block[5]:
                    use_lock(ct3_full, LockAction.AcquireGreaterEqual, value=1)
                    dma_bd(ct3_buf1)
                    use_lock(ct3_empty, LockAction.Release, value=1)
                    next_bd(block[4])
                with block[6]:
                    aie_end()

            @runtime_sequence(vec_ty_in, vec_ty_out)
            def _seq(a_in, c_out):
                # Compress on CT(0,2) MM2S only — no decompress anywhere.
                _maskwrite_compress(COMPUTE_ROW, CT_BD1_BASE, BD_MM2S, CT_MM2S0_CTRL)
                in_task = shim_dma_single_bd_task(
                    "in_alloc", a_in, sizes=[1, 1, 1, N], issue_token=True
                )
                out_task = shim_dma_single_bd_task(
                    "out_alloc", c_out, sizes=[1, 1, 1, RATIOED_N], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(in_task, out_task)

    return ctx.module


def dma_compression(in_tensor, out_tensor, config: str = "base", dev=None):
    """Build the IRON program for one compression config and return its MLIR module."""
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; pick from {CONFIGS}")

    if config == "multi_cmp_only":
        return _build_multi_cmp_only()

    # `@func passthrough_line` caches its FuncOp across calls, but each
    # iron.jit dispatch builds in a fresh MLIR context. Reset so the
    # cached op from a previous config doesn't leak into the new module.
    passthrough_line._func_op = None

    n = int(np.size(in_tensor))
    assert n == N and int(np.size(out_tensor)) == N, (
        f"in/out tensors must both be {N} elements, got "
        f"in={np.size(in_tensor)} out={np.size(out_tensor)}"
    )

    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]

    # The cross-tile lossless roundtrip is a 3-tile chain:
    # shim -> CT(0,2) -> memtile(0,1) -> shim. CT runs a Worker that copies
    # of_a -> of_b (so of_b can be the source of memtile's link without
    # being a target of CT's link — IRON rejects a fifo being in two
    # ObjectFifoLinkOps). CT MM2S compresses on the of_b leg; memtile S2MM
    # decompresses on receive. Compressed bytes live on the wire between
    # CT and memtile; both shim ends see raw int32s, so no shim TAPs.
    if config in ROUNDTRIP_CONFIGS:
        compute_tile = Tile(COL, COMPUTE_ROW, tile_type=AIETileType.CoreTile)
        if config == "lossless_roundtrip":
            link_consumer = Tile(COL, MEMTILE_ROW, tile_type=AIETileType.MemTile)
            consumer_row = MEMTILE_ROW
            consumer_bd_base = MT_BD4_BASE
            consumer_s2mm_ctrl = MT_S2MM0_CTRL
        else:  # multi_*
            link_consumer = Tile(COL, COMPUTE_ROW_2, tile_type=AIETileType.CoreTile)
            consumer_row = COMPUTE_ROW_2
            consumer_bd_base = CT_BD1_BASE
            consumer_s2mm_ctrl = CT_S2MM0_CTRL

        # Decide which compression bits to flip on the CT->consumer leg:
        #   *_base               : neither
        #   *_cmp_only           : producer (CT MM2S) only
        #   *_lossless_roundtrip : producer (CT MM2S) + consumer (S2MM decompress)
        engage_compress = config in ("multi_cmp_only", "lossless_roundtrip",
                                     "multi_lossless_roundtrip")
        engage_decompress = config in ("lossless_roundtrip",
                                       "multi_lossless_roundtrip")
        # Consumer-side TAP: when only compress is on, the consumer DMA on
        # shim receives the compressed byte stream (shorter than raw), so
        # size the shim S2MM to RATIOED_N to avoid a BD-length hang.
        out_tap_rt = _linear_tap(RATIOED_N) if engage_compress and not engage_decompress else None

        of_a = ObjectFifo(line_ty, name="a_shim_to_ct")
        of_b = ObjectFifo(line_ty, name="b_ct_to_consumer")
        of_c = of_b.cons().forward(tile=link_consumer, name="c_consumer_to_shim")

        def ct_core(of_in, of_out, copy_fn):
            for _ in range_(N // LINE_SIZE):
                elem_in = of_in.acquire(1)
                elem_out = of_out.acquire(1)
                copy_fn(elem_in, elem_out, LINE_SIZE)
                of_in.release(1)
                of_out.release(1)

        ct_worker = Worker(
            ct_core,
            [of_a.cons(), of_b.prod(), passthrough_line],
            tile=compute_tile,
        )

        rt = Runtime()
        with rt.sequence(vec_ty, vec_ty) as (a_in, c_out):
            def configure_compression_roundtrip():
                if engage_compress:
                    # CT(0,2) MM2S compress (sends compressed bytes to consumer)
                    _maskwrite_compress(
                        COMPUTE_ROW, CT_BD1_BASE, BD_MM2S, CT_MM2S0_CTRL
                    )
                if engage_decompress:
                    # Consumer tile S2MM decompress (receives compressed bytes)
                    _maskwrite_compress(
                        consumer_row, consumer_bd_base, BD_S2MM, consumer_s2mm_ctrl
                    )

            if engage_compress or engage_decompress:
                rt.inline_ops(configure_compression_roundtrip, [])
            rt.start(ct_worker)
            rt.fill(of_a.prod(), a_in)
            rt.drain(of_c.cons(), c_out, tap=out_tap_rt, wait=True)
        return Program(
            dev if dev is not None else iron.get_current_device(), rt
        ).resolve_program()

    is_memtile = config in MEMTILE_CONFIGS
    if is_memtile:
        link_tile = Tile(COL, MEMTILE_ROW, tile_type=AIETileType.MemTile)
        link_row = MEMTILE_ROW
        link_bd_base = MT_BD4_BASE
        link_s2mm_ctrl = MT_S2MM0_CTRL
        link_mm2s_ctrl = MT_MM2S0_CTRL
    else:
        link_tile = Tile(COL, COMPUTE_ROW, tile_type=AIETileType.CoreTile)
        link_row = COMPUTE_ROW
        link_bd_base = CT_BD1_BASE
        link_s2mm_ctrl = CT_S2MM0_CTRL
        link_mm2s_ctrl = CT_MM2S0_CTRL

    of_in = ObjectFifo(line_ty, name="in")
    of_out = of_in.cons().forward(tile=link_tile, name="out")

    # Core-side configs need a Worker on the compute tile that calls the
    # peano `write_tm` kernel functions to flip the compression registers
    # from within the core itself. The DMA-link still moves the data; the
    # Worker just configures + spins. Pattern is the peano-side companion
    # to PR #2348's chess-based test/npu-xrt/tile_mapped_read/.
    # Both ExternalFunctions share `object_file_name="kernel.o"` so kernel.cc
    # compiles once and both func.func @... {link_with = "kernel.o"} decls
    # resolve to the same object — no duplicate-symbol link error.
    enables = []
    if config == "core_cmp_only":
        enables = [ExternalFunction("enable_mm2s_compression", "kernel.o",
                                    source_file=_KERNEL_CC, arg_types=[])]
    elif config == "core_dcmp_only":
        enables = [ExternalFunction("enable_s2mm_decompression", "kernel.o",
                                    source_file=_KERNEL_CC, arg_types=[])]
    elif config == "core_both":
        enables = [
            ExternalFunction("enable_mm2s_compression", "kernel.o",
                             source_file=_KERNEL_CC, arg_types=[]),
            ExternalFunction("enable_s2mm_decompression", "kernel.o",
                             source_file=_KERNEL_CC, arg_types=[]),
        ]
    core_worker = None
    if enables:
        def core_body(*enable_fns):
            for fn in enable_fns:
                fn()
            for _ in range_(sys.maxsize):
                pass
        core_worker = Worker(core_body, list(enables), tile=link_tile)

    # Sizing for the shim BDs. Configs with compression on MM2S (output)
    # need a ratioed out_n to match the compressor's emitted byte count.
    # Configs with decompression on S2MM (input) need a ratioed in_n.
    # Suffix (after optional "core_"/"memtile_" prefix) determines which side
    # of the link has compression engaged: "cmp_only" = MM2S out, "dcmp_only"
    # = S2MM in, "both" = both.
    suffix = config.split("_", 1)[1] if config.startswith(("core_", "memtile_")) else config
    has_mm2s_cmp = suffix in ("cmp_only", "both")
    has_s2mm_dcmp = suffix in ("dcmp_only", "both")
    in_tap = _linear_tap(RATIOED_N) if (has_s2mm_dcmp and not has_mm2s_cmp) else None
    out_tap = _linear_tap(RATIOED_N) if has_mm2s_cmp else None

    is_host_compression = config in HOST_CONFIGS or config in MEMTILE_CONFIGS
    base_config = config in ("base", "memtile_base")

    rt = Runtime()
    with rt.sequence(vec_ty, vec_ty) as (a_in, c_out):
        if is_host_compression and not base_config:
            def configure_compression_host():
                if has_mm2s_cmp:
                    _maskwrite_compress(link_row, link_bd_base, BD_MM2S, link_mm2s_ctrl)
                if has_s2mm_dcmp:
                    _maskwrite_compress(link_row, link_bd_base, BD_S2MM, link_s2mm_ctrl)

            rt.inline_ops(configure_compression_host, [])
        elif config in CORE_CONFIGS:
            # Enable the processor bus on the compute tile so st.tm from
            # inside the core can reach the DMA registers (otherwise the
            # core hangs on the first write_tm).
            def enable_processor_bus():
                npu_maskwrite32(
                    column=COL,
                    row=COMPUTE_ROW,
                    address=CORE_PROCESSOR_BUS_EN,
                    value=0x1,
                    mask=0x1,
                )

            rt.inline_ops(enable_processor_bus, [])
            rt.start(core_worker)

        rt.fill(of_in.prod(), a_in, tap=in_tap)
        rt.drain(of_out.cons(), c_out, tap=out_tap, wait=True)

    return Program(dev if dev is not None else iron.get_current_device(), rt).resolve_program()
