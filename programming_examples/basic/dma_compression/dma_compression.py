# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

import aie.iron as iron
import numpy as np
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
    DMAChannelDir,
)
from aie.dialects.aiex import npu_maskwrite32
from aie.helpers.dialects.func import func
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    CompileTime,
    DmaChannel,
    ExternalFunction,
    Flow,
    In,
    Lock,
    ObjectFifo,
    Out,
    Program,
    Release,
    Runtime,
    TileDma,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

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
        dst[i] = src[i]  # pyright: ignore[reportCallIssue, reportArgumentType]


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


def _pingpong_dma(t, line_ty, into, out, name):
    """2-BD ping-pong S2MM -> MM2S on tile `t` (pure DMA). BD ids are pinned
    to BD_S2MM / BD_MM2S so `_maskwrite_compress` hits the right BDs."""
    bufs = [Buffer(type=line_ty, tile=t, name=f"{name}_buf{i}") for i in range(2)]
    full = Lock(tile=t, init=0, name=f"{name}_full")
    empty = Lock(tile=t, init=2, name=f"{name}_empty")

    def chain(direction, flow, acq, rel, bd_ids):
        return DmaChannel(
            direction=direction,
            channel=flow.endpoint(t),
            bds=[
                Bd(buffer=b, bd_id=i, acquires=[Acquire(acq)], releases=[Release(rel)])
                for b, i in zip(bufs, bd_ids)
            ],
        )

    dma = TileDma(
        tile=t,
        channels=[
            chain(DMAChannelDir.S2MM, into, empty, full, BD_S2MM),
            chain(DMAChannelDir.MM2S, out, full, empty, BD_MM2S),
        ],
    )
    return dma, [full, empty]


def _build_multi_cmp_only():
    """Asymmetric inter-tile compression: CT(0,2) MM2S compresses, CT(0,3)
    S2MM does NOT decompress, so the CT(0,3) buffers and the shim drain are
    sized to the compressed length to avoid a length-mismatch stall. An
    explicit `TileDma` per tile sizes each side's BDs independently, which a
    forwarded ObjectFifo cannot.
    """
    comp_ty = np.ndarray[(RATIOED_PER_LINE,), np.dtype[np.int32]]
    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]

    shim = Tile(COL, 0, tile_type=AIETileType.ShimNOCTile)
    ct2 = Tile(COL, COMPUTE_ROW, tile_type=AIETileType.CoreTile)
    ct3 = Tile(COL, COMPUTE_ROW_2, tile_type=AIETileType.CoreTile)

    # Channel 0 throughout: the maskwrites target the channel-0 CTRL registers.
    into = Flow(shim, ct2, src_channel=0, dst_channel=0)
    link = Flow(ct2, ct3, src_channel=0, dst_channel=0)
    out = Flow(ct3, shim, src_channel=0, dst_channel=0)

    ct2_dma, ct2_locks = _pingpong_dma(ct2, line_ty, into, link, "ct2")
    ct3_dma, ct3_locks = _pingpong_dma(ct3, comp_ty, link, out, "ct3")

    # The host out buffer is full N so the JIT size check accepts the test's
    # tensor; the drain writes only RATIOED_N ints and leaves the tail at
    # SENTINEL, which the test scores as "untouched".
    def sequence(a_in, c_out):
        _maskwrite_compress(COMPUTE_ROW, CT_BD1_BASE, BD_MM2S, CT_MM2S0_CTRL)
        into.fill(a_in)
        out.drain(c_out, tap=_linear_tap(RATIOED_N), wait=True)

    rt = Runtime(sequence, [vec_ty, vec_ty])
    for f in (into, link, out):
        rt.add_flow(f)
    for lk in ct2_locks + ct3_locks:
        rt.add_lock(lk)
    rt.add_tile_dma(ct2_dma)
    rt.add_tile_dma(ct3_dma)
    return Program(iron.get_current_device(), rt).resolve_program()


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

    def sequence(a_in, c_out, out_h):
        npu_maskwrite32(
            column=COL,
            row=COMPUTE_ROW,
            address=CORE_PROCESSOR_BUS_EN,
            value=0x1,
            mask=0x1,
        )
        out_h.drain(c_out, wait=True)

    rt = Runtime(sequence, [vec_ty, vec_ty, of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def dma_compression(
    in_tensor: In,
    out_tensor: Out,
    *,
    config: CompileTime[str] = "base",
):
    """Build the IRON program for one compression config and return its MLIR module."""
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; pick from {CONFIGS}")

    if config == "multi_cmp_only":
        return _build_multi_cmp_only()

    if config == "regdump":
        return _build_regdump()

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

        def sequence(a_in, c_out, in_h, out_h):
            if engage_compress:
                # CT(0,2) MM2S compress (sends compressed bytes to consumer)
                _maskwrite_compress(COMPUTE_ROW, CT_BD1_BASE, BD_MM2S, CT_MM2S0_CTRL)
            if engage_decompress:
                # Consumer tile S2MM decompress (receives compressed bytes)
                _maskwrite_compress(
                    consumer_row, consumer_bd_base, BD_S2MM, consumer_s2mm_ctrl
                )
            in_h.fill(a_in)
            out_h.drain(c_out, tap=out_tap_rt, wait=True)

        rt = Runtime(sequence, [vec_ty, vec_ty, of_a.prod(), of_c.cons()])
        return Program(
            iron.get_current_device(), rt, workers=[ct_worker]
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

    def sequence(a_in, c_out, in_h, out_h):
        if is_host_compression and not base_config:
            if has_mm2s_cmp:
                _maskwrite_compress(link_row, link_bd_base, BD_MM2S, link_mm2s_ctrl)
            if has_s2mm_dcmp:
                _maskwrite_compress(link_row, link_bd_base, BD_S2MM, link_s2mm_ctrl)
        elif config in CORE_CONFIGS:
            # Enable the processor bus on the compute tile so st.tm from
            # inside the core can reach the DMA registers (otherwise the
            # core hangs on the first write_tm).
            npu_maskwrite32(
                column=COL,
                row=COMPUTE_ROW,
                address=CORE_PROCESSOR_BUS_EN,
                value=0x1,
                mask=0x1,
            )

        in_h.fill(a_in, tap=in_tap)
        out_h.drain(c_out, tap=out_tap, wait=True)

    rt = Runtime(sequence, [vec_ty, vec_ty, of_in.prod(), of_out.cons()])
    workers = [core_worker] if core_worker is not None else []
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()
