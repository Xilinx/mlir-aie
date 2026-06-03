"""IRON DMA compression probes for AIE-ML (npu1) and AIE2P (npu2).

`dma_compression(in_tensor, out_tensor, config=...)` builds the MLIR module
for one of 16 configs across 5 families: host-driven (`base`/`cmp_only`/...),
core-driven via peano `write_tm` (`core_*`), memtile (`memtile_*`),
cross-tile chains (`lossless_roundtrip`/`multi_*`), and a `regdump`
write_tm+read_tm self-test. See README.md for the per-config table.

RATIOED_N=2944 is the empirical compressed byte count for arange(N); used
to size asymmetric BDs so neither side hangs on a length mismatch.
"""

import os
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.helpers.dialects.func import func
from aie.iron.device import Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.dialects._aie_enum_gen import AIETileType
from aie.dialects.aie import (
    device,
    tile,
    buffer,
    lock,
    mem,
    dma_start,
    dma_bd,
    next_bd,
    use_lock,
    flow,
    end as aie_end,
    core,
    AIEDevice,
    DMAChannelDir,
    LockAction,
    WireBundle,
    shim_dma_allocation,
)
from aie.dialects.aiex import (
    npu_maskwrite32,
    runtime_sequence,
    shim_dma_single_bd_task,
    dma_start_task,
    dma_await_task,
)
from aie.extras.context import mlir_mod_ctx

N = 4096
LINE_SIZE = 1024
RATIOED_N = 2944  # empirical compressed byte count for arange(N) on Phoenix
RATIOED_PER_LINE = RATIOED_N // (N // LINE_SIZE)  # = 736; per-BD compressed

COL = 0
COMPUTE_ROW = 2
MEMTILE_ROW = 1

# Compute-tile DMA register layout (compression bit in BD?_1, bit 31).
CT_BD1_BASE = 0x1D004
CT_S2MM0_CTRL = 0x1DE00
CT_MM2S0_CTRL = 0x1DE10

# Memtile DMA register layout (compression bit in BD?_4, bit 31).
MT_BD4_BASE = 0xA0010
MT_S2MM0_CTRL = 0xA0600
MT_MM2S0_CTRL = 0xA0630

BD_STRIDE = 0x20
COMPRESS_BIT = 0x80000000  # BD?_X bit 31 (X=1 on compute, X=4 on memtile)
CHAN_BIT = 0x10  # *_CTRL bit 4 (same on compute and memtile)
BD_S2MM = (0, 1)
BD_MM2S = (2, 3)

HOST_CONFIGS = ("base", "cmp_only", "dcmp_only", "both")
CORE_CONFIGS = ("core_cmp_only", "core_dcmp_only", "core_both")
MEMTILE_CONFIGS = (
    "memtile_base",
    "memtile_cmp_only",
    "memtile_dcmp_only",
    "memtile_both",
)
# Two-tile chains: shim -> CT(0,2) -> {memtile(0,1) | CT(0,3)} -> shim.
ROUNDTRIP_CONFIGS = (
    "lossless_roundtrip",
    "multi_base",
    "multi_cmp_only",
    "multi_lossless_roundtrip",
)
# Core-side write_tm + read_tm self-test (issue #2346 readback side).
REGDUMP_CONFIGS = ("regdump",)
CONFIGS = (
    HOST_CONFIGS + CORE_CONFIGS + MEMTILE_CONFIGS + ROUNDTRIP_CONFIGS + REGDUMP_CONFIGS
)

COMPUTE_ROW_2 = 3  # second compute tile for the multi-tile configs

# Must be 1 before the core can do st.tm / lda.tm to the processor bus.
CORE_PROCESSOR_BUS_EN = 0x32038

_KERNEL_CC = os.path.join(os.path.dirname(__file__), "kernel.cc")

line_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]


# Module-level — @func needs an active MLIR context at decoration time.
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
    """Asymmetric inter-tile compression: CT(0,2) MM2S compresses, CT(0,3)
    S2MM does NOT decompress, so the CT(0,3) and shim S2MM BDs are
    hand-sized to RATIOED_PER_LINE to avoid a length-mismatch stall.
    Built from low-level aie dialect because IRON's link API doesn't
    expose per-side BD sizing.
    """

    raw_ty = np.ndarray[(LINE_SIZE,), np.dtype[np.int32]]
    comp_ty = np.ndarray[(RATIOED_PER_LINE,), np.dtype[np.int32]]
    vec_ty_in = np.ndarray[(N,), np.dtype[np.int32]]
    # Host-facing out memref is full N so the JIT tensor-size check accepts
    # the test's N-element out_tensor; the shim S2MM BD below still writes
    # only RATIOED_N ints (the compressed stream length), leaving the tail
    # at SENTINEL — which the test scores as "untouched".
    vec_ty_out = np.ndarray[(N,), np.dtype[np.int32]]

    def _emit_passthrough_mem(t, buf0, buf1, full_lock, empty_lock):
        """2-BD ping-pong S2MM ch0 -> MM2S ch0 on tile `t` (pure DMA)."""

        @mem(t)
        def _m(block):
            dma_start(DMAChannelDir.S2MM, 0, dest=block[1], chain=block[3])
            with block[1]:
                use_lock(empty_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf0)
                use_lock(full_lock, LockAction.Release, value=1)
                next_bd(block[2])
            with block[2]:
                use_lock(empty_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf1)
                use_lock(full_lock, LockAction.Release, value=1)
                next_bd(block[1])
            with block[3]:
                dma_start(DMAChannelDir.MM2S, 0, dest=block[4], chain=block[6])
            with block[4]:
                use_lock(full_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf0)
                use_lock(empty_lock, LockAction.Release, value=1)
                next_bd(block[5])
            with block[5]:
                use_lock(full_lock, LockAction.AcquireGreaterEqual, value=1)
                dma_bd(buf1)
                use_lock(empty_lock, LockAction.Release, value=1)
                next_bd(block[4])
            with block[6]:
                aie_end()

    resolved = iron.get_current_device().resolve()
    aie_dev = (
        AIEDevice.npu2_1col
        if resolved in (AIEDevice.npu2, AIEDevice.npu2_1col)
        else AIEDevice.npu1_1col
    )

    with mlir_mod_ctx() as ctx:

        @device(aie_dev)
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

            # CT(0,2): S2MM receives raw lines from shim, MM2S compresses
            # and emits to CT(0,3). Buffers sized LINE_SIZE (raw).
            _emit_passthrough_mem(ct2, ct2_buf0, ct2_buf1, ct2_full, ct2_empty)
            # CT(0,3): S2MM receives RATIOED_PER_LINE ints' worth from the
            # wire (the compressed stream from CT(0,2)); MM2S forwards to
            # shim. Buffers sized RATIOED_PER_LINE — the key trick that
            # avoids a BD-length stall with compress-on / decompress-off.
            _emit_passthrough_mem(ct3, ct3_buf0, ct3_buf1, ct3_full, ct3_empty)

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


def _build_regdump():
    """Core-side write_tm + read_tm self-test. Host enables the processor bus;
    kernel writes COMPRESS_BIT to each BD?_1 and reads it back. Driver
    asserts each post-write read equals COMPRESS_BIT."""
    compute_tile = Tile(COL, COMPUTE_ROW, tile_type=AIETileType.CoreTile)
    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_out = ObjectFifo(line_ty, name="regs_out")

    dump_fn = ExternalFunction(
        "dump_compress_regs",
        "kernel.o",
        source_file=_KERNEL_CC,
        arg_types=[line_ty],
    )

    def regdump_core(of_out_h, dump):
        for _ in range_(N // LINE_SIZE):
            elem = of_out_h.acquire(1)
            dump(elem)
            of_out_h.release(1)

    worker = Worker(regdump_core, [of_out.prod(), dump_fn], tile=compute_tile)

    rt = Runtime()
    with rt.sequence(vec_ty, vec_ty) as (a_in, c_out):

        def enable_processor_bus():
            npu_maskwrite32(
                column=COL,
                row=COMPUTE_ROW,
                address=CORE_PROCESSOR_BUS_EN,
                value=0x1,
                mask=0x1,
            )

        rt.inline_ops(enable_processor_bus, [])
        rt.start(worker)
        rt.drain(of_out.cons(), c_out, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def dma_compression(
    in_tensor: In,
    out_tensor: Out,
    *,
    config: Compile[str] = "base",
):
    """Build the IRON program for one compression config and return its MLIR module."""
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; pick from {CONFIGS}")

    if config == "multi_cmp_only":
        return _build_multi_cmp_only()

    if config == "regdump":
        return _build_regdump()

    # @func caches its emitted FuncOp on the decorator instance, but each
    # @iron.jit invocation builds in its own MLIR context. Reset the cache
    # so passthrough_line emits a fresh FuncOp into THIS context (otherwise
    # the second config in a multi-config sweep tries to reference a
    # FuncOp owned by a now-dead context — KeyError at func call site).
    passthrough_line._func_op = None

    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]

    # Cross-tile chains: CT(0,2) runs a copy Worker (not a link, because
    # IRON rejects a fifo participating in two ObjectFifoLinkOps); the
    # consumer tile forwards back to shim. Compression engages on the
    # CT -> consumer leg; both shim ends see raw int32s when both
    # directions are enabled.
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

        engage_compress = config in (
            "multi_cmp_only",
            "lossless_roundtrip",
            "multi_lossless_roundtrip",
        )
        engage_decompress = config in ("lossless_roundtrip", "multi_lossless_roundtrip")
        # Asymmetric compress-only: ratio-size shim S2MM to match the
        # compressed stream length.
        out_tap_rt = (
            _linear_tap(RATIOED_N)
            if engage_compress and not engage_decompress
            else None
        )

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
        return Program(iron.get_current_device(), rt).resolve_program()

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

    # `core_*` configs spawn a Worker on the link tile that calls the peano
    # `write_tm` kernel functions to flip compression registers from inside
    # the core. The DMA passthrough still moves the data; the Worker only
    # configures and then spins.
    enables = []
    if config == "core_cmp_only":
        enables = [
            ExternalFunction(
                "enable_mm2s_compression",
                "kernel.o",
                source_file=_KERNEL_CC,
                arg_types=[],
            )
        ]
    elif config == "core_dcmp_only":
        enables = [
            ExternalFunction(
                "enable_s2mm_decompression",
                "kernel.o",
                source_file=_KERNEL_CC,
                arg_types=[],
            )
        ]
    elif config == "core_both":
        enables = [
            ExternalFunction(
                "enable_mm2s_compression",
                "kernel.o",
                source_file=_KERNEL_CC,
                arg_types=[],
            ),
            ExternalFunction(
                "enable_s2mm_decompression",
                "kernel.o",
                source_file=_KERNEL_CC,
                arg_types=[],
            ),
        ]
    core_worker = None
    if enables:

        def core_body(*enable_fns):
            for fn in enable_fns:
                fn()
            for _ in range_(sys.maxsize):
                pass

        core_worker = Worker(core_body, list(enables), tile=link_tile)

    # Strip "core_"/"memtile_" prefix to get the cmp/dcmp/both suffix.
    suffix = (
        config.split("_", 1)[1] if config.startswith(("core_", "memtile_")) else config
    )
    has_mm2s_cmp = suffix in ("cmp_only", "both")
    has_s2mm_dcmp = suffix in ("dcmp_only", "both")
    # Ratio-size each shim BD whose channel is doing (de)compression.
    in_tap = _linear_tap(RATIOED_N) if has_s2mm_dcmp else None
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

    return Program(iron.get_current_device(), rt).resolve_program()
