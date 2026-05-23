"""Per-block standalone IRON design for yolo26n-cls.

Compile a single block to MLIR:

    python3 aie2_yolo_per_block.py m0 > /tmp/m0.mlir

Mirrors the mobilenet-py pattern of `aie2_iron_per_block.py`:
  - one builder per block (m0..m10),
  - block name → builder dispatch,
  - shim-fill -> compute -> shim-drain runtime sequence.

Data sources:
  - yolo_spec.py        : block shapes (Conv records, kinds)
  - placement.py        : tile assignments (PLACEMENT[block_name])
  - data/manifest.json  : per-op weights/bias paths, log2 scales
  - data/<block>/*.bin  : raw INT8 weights + INT32 biases (extractor output)

The kernels under kernels/ are forks of mlir-aie's mobilenet kernels with:
  (a) bias promoted to INT32 accumulator init (no separate bias-add stage),
  (b) INT8 SiLU LUT replacing the fused-ReLU output (see gen_yolo_silu_luts.py).
"""

import argparse
import json
import os
import sys

import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime
from aie.iron import Worker as _IronWorker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint

import yolo_spec
import placement
from lowlevel_dma import StaticWeightStream

# ---------------------------------------------------------------------------
# Trace gating. When env var TRACE_SIZE_PER_WORKER > 0 at MLIR-generation
# time, every Worker constructed below auto-enables HW packet tracing with
# the given per-tile byte budget. The kernels already emit event0()/event1()
# around compute; aie.iron.Runtime.enable_trace() (called in the chain/
# partial top-levels) routes the packets to DRAM. Default 0 = no trace
# (zero overhead on production builds).
#
# TRACE_EVENTS (csv of CoreEventAIE2P names) overrides the default 8 events
# (INSTR_EVENT_0/1 + 6 stall/port events) on each traced Worker — handy
# for stall-attribution traces where you want, e.g., just
# `INSTR_EVENT_0,INSTR_EVENT_1,LOCK_STALL,STREAM_STALL,MEMORY_STALL`.
# Smaller event sets also shrink per-packet payload, letting the trace BO
# capture more kernel iterations before overflow.
# ---------------------------------------------------------------------------
TRACE_SIZE_PER_WORKER = int(os.environ.get("TRACE_SIZE_PER_WORKER", "0"))


def _resolve_trace_events():
    names = [
        n.strip() for n in os.environ.get("TRACE_EVENTS", "").split(",") if n.strip()
    ]
    if not names:
        return None
    from aie.utils.trace.events import CoreEventAIE2P

    out = []
    for n in names:
        if not hasattr(CoreEventAIE2P, n):
            raise ValueError(
                f"TRACE_EVENTS: {n!r} not in CoreEventAIE2P; available: "
                f"{[m for m in dir(CoreEventAIE2P) if m.isupper()][:20]}..."
            )
        out.append(getattr(CoreEventAIE2P, n))
    return out


TRACE_EVENTS = _resolve_trace_events()


class Worker(_IronWorker):
    def __init__(self, *args, **kwargs):
        if TRACE_SIZE_PER_WORKER > 0 and kwargs.get("trace") is None:
            kwargs["trace"] = TRACE_SIZE_PER_WORKER
        if TRACE_EVENTS is not None and kwargs.get("trace_events") is None:
            kwargs["trace_events"] = TRACE_EVENTS
        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def _i32(shape):
    return np.ndarray[shape, np.dtype[np.int32]]


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load_manifest():
    with open(os.path.join(DATA_DIR, "manifest.json")) as f:
        return json.load(f)


def _load_bin(manifest_path: str, dtype, expected_size: int) -> np.ndarray:
    """Read a raw-bytes weight/bias file. `manifest_path` is relative to repo
    root (e.g. 'data/model.0/...'); we strip the leading 'data/' to make it
    relative to DATA_DIR."""
    rel = (
        manifest_path[len("data/") :]
        if manifest_path.startswith("data/")
        else manifest_path
    )
    arr = np.fromfile(os.path.join(DATA_DIR, rel), dtype=dtype)
    if arr.size != expected_size:
        raise ValueError(
            f"{manifest_path}: expected {expected_size} {dtype.__name__} elements, got {arr.size}"
        )
    return arr


def _op_meta(manifest, manifest_name: str) -> dict:
    for L in manifest["layers"]:
        if L["name"] == manifest_name:
            return L
    raise KeyError(f"manifest has no op named {manifest_name!r}")


# ---------------------------------------------------------------------------
# m0 — stem: 3x3 stride-2 INT8 conv, 3 -> 16 channels.
# Mirrors mobilenet `init` (aie2_mobilenet_iron.py:90-200). Three deltas:
#   1. Input channels = 3 (RGB), padded to 8 on host for vectorization.
#      Weights also need padding 3->8 in I dim (zero-fill new slots). Manifest
#      currently stores OIYX_raw at (16,3,3,3) = 432 i8 elements. Caller of
#      this builder is responsible for emitting padded (16,8,3,3) = 1152 i8
#      and a matching padded INT8 input — same pre-processing trick as
#      mobilenet's init.
#   2. Bias: manifest has bias_int32 (INT32, pre-shifted by bias_pre_shift).
#      Kernel uses it as accumulator init when weight_index==0 (folds the
#      bias-add into the conv's first MAC). Bias passed as a separate Buffer.
#   3. Activation: SiLU LUT, not ReLU. Kernel `.o` must implement INT8 SiLU
#      at the requant epilogue.
# ---------------------------------------------------------------------------
def _build_m0(act_in, manifest):
    blk = yolo_spec.block("m0")
    layer = blk.layers[0]
    in_w, in_h, in_c_raw = layer.in_shape  # (512, 512, 3)
    out_w, out_h, out_c = layer.out_shape  # (256, 256, 16)
    assert in_c_raw == 3 and out_c == 16, f"unexpected m0 shape: {layer}"

    # Vectorized input channel count (pad 3 -> 8). Both the host activation
    # buffer and the weights are expected pre-padded; see docstring above.
    in_c = 8
    wts_padded_sz = out_c * in_c * 3 * 3  # 16 * 8 * 9 = 1152

    op_meta = _op_meta(manifest, layer.manifest_name)
    right_shift = op_meta["right_shift"]  # post-MAC shift baked into requant

    # Weights: load raw (16,3,3,3) then pad I dim to 8 with zeros.
    wts_raw_sz = int(np.prod(op_meta["weights_shape"]))  # 432
    wts_raw = _load_bin(op_meta["weights_file"], np.int8, wts_raw_sz)
    wts_raw = wts_raw.reshape(16, 3, 3, 3)
    wts_padded = np.zeros((16, in_c, 3, 3), dtype=np.int8)
    wts_padded[:, :3, :, :] = wts_raw
    wts_flat = wts_padded.reshape(-1)
    assert wts_flat.size == wts_padded_sz

    # Bias: 16 INT32 values, already promoted as (bias_i8 << bias_pre_shift).
    bias_data = _load_bin(op_meta["bias_file"], np.int32, out_c)

    # SiLU LUT: 256-byte int8 LUT indexed by pre_silu+128. Generated by
    # gen_yolo_silu_luts.py from the ONNX SiLU-chain scales.
    silu_lut_path = os.path.join(DATA_DIR, "model.0", "silu_lut.bin")
    silu_lut_data = np.fromfile(silu_lut_path, dtype=np.int8)
    assert (
        silu_lut_data.size == 256
    ), f"m0 silu_lut should be 256 bytes, got {silu_lut_data.size}"

    # Static buffers on the compute tile.
    wts_buf = Buffer(_i8((wts_padded_sz,)), initial_value=wts_flat)
    bias_buf = Buffer(_i32((out_c,)), initial_value=bias_data)
    silu_lut_buf = Buffer(_i8((256,)), initial_value=silu_lut_data)

    # Activation FIFOs. Output is INT8 (SiLU output is signed, unlike ReLU/uint8).
    # via_DMA on the block-output fifo: the producer always emits to a
    # DMA-routed stream so standalone (shim drain) and chain (next block's
    # input) share the same transport, regardless of consumer placement.
    act_out = ObjectFifo(_i8((out_w, 1, out_c)), depth=2, via_DMA=True)

    # Kernel: forked from mlir-aie/aie_kernels/aie2p/init_conv2dk3.cc with
    # bias-as-accumulator-init and ReLU→INT8 SiLU LUT epilogue.
    k_m0 = Kernel(
        "yolo_m0_conv2dk3_stride2_silu_bias_i8_i8",
        "yolo_m0_conv2dk3_silu_bias.o",
        [
            _i8((in_w, 1, in_c)),  # row[t-1] (top neighbor row)
            _i8((in_w, 1, in_c)),  # row[t]   (current row)
            _i8((in_w, 1, in_c)),  # row[t+1] (bottom neighbor row)
            _i8((wts_padded_sz,)),  # weights (OIYX padded I=8)
            _i32((out_c,)),  # bias INT32 (accum init)
            _i8((256,)),  # silu_lut (int8 LUT, pre_silu+128)
            _i8((out_w, 1, out_c)),  # output row
            np.int32,
            np.int32,
            np.int32,  # inW, inC, outC
            np.int32,
            np.int32,  # kW, kH
            np.int32,  # border (0=preamble, 1=middle)
            np.int32,  # right_shift
            np.int32,  # padding (unused for stride-2 stem)
        ],
    )

    def m0_fn(act_in_f, act_out_f, wts, bias, lut, k):
        # Two-phase sliding window for 3x3 stride-2:
        #   Preamble: rows [0, 0, 1] -> output row 0, border=0
        #   Middle:   rows [2k, 2k+1, 2k+2] -> output row k, border=1, k in [1, out_h-1]
        rows = act_in_f.acquire(2)
        row_out = act_out_f.acquire(1)
        k(
            rows[0],
            rows[0],
            rows[1],
            wts,
            bias,
            lut,
            row_out,
            in_w,
            in_c,
            out_c,
            3,
            3,
            0,
            right_shift,
            0,
        )
        act_out_f.release(1)
        act_in_f.release(1)

        for _ in range_(out_h - 1):
            rows = act_in_f.acquire(3)
            row_out = act_out_f.acquire(1)
            k(
                rows[0],
                rows[1],
                rows[2],
                wts,
                bias,
                lut,
                row_out,
                in_w,
                in_c,
                out_c,
                3,
                3,
                1,
                right_shift,
                0,
            )
            act_in_f.release(2)
            act_out_f.release(1)
        act_in_f.release(1)

    w_m0 = Worker(
        m0_fn,
        fn_args=[
            act_in.cons(depth=5),  # sliding-window of 3 needs at least 3 live slots
            act_out.prod(depth=2),
            wts_buf,
            bias_buf,
            silu_lut_buf,
            k_m0,
        ],
        tile=placement.PLACEMENT["m0"],
        # Deep-opt vec kernel holds 2 × 576 B weight pack buffers
        # (OCx2 fold) on stack; default 1 KB stack would overflow
        # silently (the symptom is a NPU dispatch timeout).
        stack_size=4096,
    )
    return act_out, [w_m0]


# ---------------------------------------------------------------------------
# Conv-stride blocks (m1, m3, m5, m7) — same shape family as m0 but with
# input channels aligned to 8, so no padding step. Two implementations:
#   - m1 (this function): one shared OIYXI8O8 .o, shapes passed as runtime
#     args; weights fit in tile L1 alongside activations.
#   - m3/m5/m7 (_build_conv_stride_block_streamed): weights exceed the 64KB
#     tile budget so a per-block kernel .o streams weight chunks from a
#     MemTile via StaticWeightStream.
# ---------------------------------------------------------------------------
def _build_conv_stride_block(block_name: str, act_in, manifest):
    blk = yolo_spec.block(block_name)
    assert blk.topology == "conv_stride" and len(blk.layers) == 1, blk
    layer = blk.layers[0]
    in_w, in_h, in_c = layer.in_shape
    out_w, out_h, out_c = layer.out_shape
    assert (
        in_c % 8 == 0
    ), f"{block_name}: in_c={in_c} not 8-aligned — needs padding pass like m0"
    assert layer.stride == 2, f"{block_name}: expected stride 2, got {layer.stride}"

    op_meta = _op_meta(manifest, layer.manifest_name)
    right_shift = op_meta["right_shift"]
    wts_sz = int(
        np.prod(op_meta["weights_shape"])
    )  # OIYXI8O8 packed for all conv_stride blocks here
    wts_data = _load_bin(op_meta["weights_file"], np.int8, wts_sz)
    bias_data = _load_bin(op_meta["bias_file"], np.int32, out_c)

    # SiLU LUT: 256-byte int8 from gen_yolo_silu_luts.py.
    model_n = block_name[1:]  # 'm1' -> '1'
    silu_lut_path = os.path.join(DATA_DIR, f"model.{model_n}", "silu_lut.bin")
    silu_lut_data = np.fromfile(silu_lut_path, dtype=np.int8)
    assert (
        silu_lut_data.size == 256
    ), f"{block_name} silu_lut should be 256, got {silu_lut_data.size}"

    wts_buf = Buffer(_i8((wts_sz,)), initial_value=wts_data)
    bias_buf = Buffer(_i32((out_c,)), initial_value=bias_data)
    silu_lut_buf = Buffer(_i8((256,)), initial_value=silu_lut_data)
    # via_DMA on the block-output fifo so standalone (shim drain) and chain
    # (next block's input) share the same DMA-routed transport; see m0 above.
    act_out = ObjectFifo(_i8((out_w, 1, out_c)), depth=2, via_DMA=True)

    # Shared OIYXI8O8 kernel across m1/m3/m5/m7 — shapes are runtime args, so
    # one .o serves all four blocks. m0 has its own kernel (OIYX raw weights).
    k = Kernel(
        "yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_i8_i8",
        "yolo_conv2dk3_stride2_silu_bias_oiyxi8o8.o",
        [
            _i8((in_w, 1, in_c)),
            _i8((in_w, 1, in_c)),
            _i8((in_w, 1, in_c)),
            _i8((wts_sz,)),
            _i32((out_c,)),
            _i8((256,)),  # silu_lut
            _i8((out_w, 1, out_c)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    def block_fn(act_in_f, act_out_f, wts, bias, lut, k):
        # Identical 2-phase sliding-window 3x3 stride-2 pattern as m0.
        rows = act_in_f.acquire(2)
        row_out = act_out_f.acquire(1)
        k(
            rows[0],
            rows[0],
            rows[1],
            wts,
            bias,
            lut,
            row_out,
            in_w,
            in_c,
            out_c,
            3,
            3,
            0,
            right_shift,
            0,
        )
        act_out_f.release(1)
        act_in_f.release(1)
        for _ in range_(out_h - 1):
            rows = act_in_f.acquire(3)
            row_out = act_out_f.acquire(1)
            k(
                rows[0],
                rows[1],
                rows[2],
                wts,
                bias,
                lut,
                row_out,
                in_w,
                in_c,
                out_c,
                3,
                3,
                1,
                right_shift,
                0,
            )
            act_in_f.release(2)
            act_out_f.release(1)
        act_in_f.release(1)

    w = Worker(
        block_fn,
        fn_args=[
            act_in.cons(depth=5),
            act_out.prod(depth=2),
            wts_buf,
            bias_buf,
            silu_lut_buf,
            k,
        ],
        tile=placement.PLACEMENT[block_name],
        # Deep-opt m1 vec uses up to 4 mmul accumulators + small a_buf
        # scratches per interior x_pair body. Stays comfortably under
        # 4 KB but the AIE2P default 1 KB stack is too small for the
        # accumulator-register spills the OCx2/2X×2OC fold introduces.
        stack_size=4096,
    )
    return act_out, [w]


# ---------------------------------------------------------------------------
# Streamed variant for m3/m5/m7 — weights exceed the AIE2P 64KB tile budget,
# so we stage them on a MemTile (one per col, row 1) and stream chunks via
# lowlevel_dma.StaticWeightStream. Same OIYXI8O8 algorithm as
# _build_conv_stride_block but the kernel takes oc_offset/oc_count and the
# worker calls it n_splits times per output row.
# ---------------------------------------------------------------------------
def _build_conv_stride_block_streamed(
    block_name: str, act_in, manifest, oc_per_chunk: int = 16, out_depth: int = 2
):
    blk = yolo_spec.block(block_name)
    assert blk.topology == "conv_stride" and len(blk.layers) == 1, blk
    layer = blk.layers[0]
    in_w, in_h, in_c = layer.in_shape
    out_w, out_h, out_c = layer.out_shape
    assert (
        in_c % 8 == 0 and out_c % 8 == 0
    ), f"{block_name}: in_c/out_c must be 8-aligned"
    assert layer.stride == 2
    assert (
        out_c % oc_per_chunk == 0
    ), f"{block_name}: out_c={out_c} not divisible by oc_per_chunk={oc_per_chunk}"
    assert (
        oc_per_chunk % 8 == 0
    ), f"oc_per_chunk={oc_per_chunk} must be 8-aligned (OIYXI8O8 packing)"

    op_meta = _op_meta(manifest, layer.manifest_name)
    right_shift = op_meta["right_shift"]
    wts_sz_total = int(np.prod(op_meta["weights_shape"]))  # OIYXI8O8 packed total
    wts_data = _load_bin(op_meta["weights_file"], np.int8, wts_sz_total)
    bias_data = _load_bin(op_meta["bias_file"], np.int32, out_c)

    n_splits = out_c // oc_per_chunk
    # Weight chunk: (oc_per_chunk/8, in_c/8, 3, 3, 8, 8) bytes — slice of the
    # OIYXI8O8-packed weights along the oc_outer dim. Since OIYXI8O8's first
    # axis is oc/8, chunk i (oc range [i*oc_per_chunk, (i+1)*oc_per_chunk))
    # is a contiguous slice of the flat weight buffer.
    wts_chunk_sz = wts_sz_total // n_splits
    assert wts_sz_total % n_splits == 0

    model_n = block_name[1:]
    silu_lut_path = os.path.join(DATA_DIR, f"model.{model_n}", "silu_lut.bin")
    silu_lut_data = np.fromfile(silu_lut_path, dtype=np.int8)
    assert silu_lut_data.size == 256

    compute_tile = placement.PLACEMENT[block_name]
    memtile = next(
        t
        for t in placement.PLACEMENT["memtile"]["available"]
        if t.col == compute_tile.col
    )

    # Split the weight tensor into n_splits per-chunk arrays. The memtile
    # ObjectFifo holds n_splits buffers preloaded via init_values; with
    # repeat_count=out_h, the BD chain replays the n_splits-buffer cycle
    # out_h times, delivering n_splits * out_h chunks per sample without
    # any host weight transfer.
    chunks = [
        np.ascontiguousarray(wts_data[i * wts_chunk_sz : (i + 1) * wts_chunk_sz])
        for i in range(n_splits)
    ]
    wts_fifo = ObjectFifo(
        _i8((wts_chunk_sz,)),
        depth=n_splits,
        name=f"{block_name}_wts",
        init_values=chunks,
    )

    # iter_count unset → BD chain self-loops infinitely via next_bd(self),
    # so a single dispatch can consume arbitrarily many samples regardless of
    # N. Lowering correctness for this case lives in
    # AIEObjectFifoStatefulTransform.cpp::createObjectFifoLocks — when a
    # static-init no-link producer has no iter_count, source-side locks are
    # skipped so the chain doesn't deadlock on its second pass.
    # The producer side has no Worker / no rt.fill; pin it to a MemTile so
    # the static buffers land there (init_values is rejected on shim).
    wts_fifo.prod().endpoint = ObjectFifoEndpoint(memtile)

    bias_buf = Buffer(_i32((out_c,)), initial_value=bias_data)
    silu_lut_buf = Buffer(_i8((256,)), initial_value=silu_lut_data)
    # via_DMA on the block-output fifo so standalone (shim drain) and chain
    # (next block's input) share the same DMA-routed transport; see m0 above.
    act_out = ObjectFifo(_i8((out_w, 1, out_c)), depth=out_depth, via_DMA=True)

    k = Kernel(
        f"yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_i8_i8_{block_name}",
        f"yolo_conv2dk3_stride2_silu_bias_oiyxi8o8_chunked_{block_name}.o",
        [
            _i8((in_w, 1, in_c)),
            _i8((in_w, 1, in_c)),
            _i8((in_w, 1, in_c)),
            _i8((wts_chunk_sz,)),  # chunk of weights (not full tensor)
            _i32((out_c,)),  # full bias (still tiny)
            _i8((256,)),  # silu_lut
            _i8((out_w, 1, out_c)),  # full output row (chunks RMW-merge into it)
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,  # border
            np.int32,  # right_shift
            np.int32,
            np.int32,  # oc_offset, oc_count
        ],
    )

    def block_fn(act_in_f, act_out_f, wts_cons, bias, lut, k):
        # Two-phase stride-2 sliding window. For each output row, call kernel
        # n_splits times — each call processes oc_per_chunk output channels.
        # Preamble (border=0):
        rows = act_in_f.acquire(2)
        row_out = act_out_f.acquire(1)
        for wi in range_(n_splits):
            wts_chunk = wts_cons.acquire(1)
            k(
                rows[0],
                rows[0],
                rows[1],
                wts_chunk,
                bias,
                lut,
                row_out,
                in_w,
                in_c,
                out_c,
                3,
                3,
                0,
                right_shift,
                wi * oc_per_chunk,
                oc_per_chunk,
            )
            wts_cons.release(1)
        act_out_f.release(1)
        act_in_f.release(1)

        # Middle (border=1):
        for _ in range_(out_h - 1):
            rows = act_in_f.acquire(3)
            row_out = act_out_f.acquire(1)
            for wi in range_(n_splits):
                wts_chunk = wts_cons.acquire(1)
                k(
                    rows[0],
                    rows[1],
                    rows[2],
                    wts_chunk,
                    bias,
                    lut,
                    row_out,
                    in_w,
                    in_c,
                    out_c,
                    3,
                    3,
                    1,
                    right_shift,
                    wi * oc_per_chunk,
                    oc_per_chunk,
                )
                wts_cons.release(1)
            act_in_f.release(2)
            act_out_f.release(1)
        act_in_f.release(1)

    w = Worker(
        block_fn,
        fn_args=[
            act_in.cons(depth=3),
            act_out.prod(depth=out_depth),
            wts_fifo.cons(depth=1),
            bias_buf,
            silu_lut_buf,
            k,
        ],
        tile=compute_tile,
    )
    return act_out, [w]


# ---------------------------------------------------------------------------
# c3k2_small (m2, m4) — yolo C2f-style block with chunk(2) + inner bottleneck.
#
# Topology (ultralytics C2f reference):
#     y0, y1 = chunk(cv1(x), 2, dim=channel)
#     y_m0   = y1 + m.0/cv2(SiLU(m.0/cv1(y1)))          # inner residual
#     out    = cv2(cat([y0, y1, y_m0], dim=channel))
#
# Channel widths (with c = out_c // 2):
#     cv1     :  c_in -> 2c   (1x1)
#     y0, y1  : each c
#     m.0/cv1 :  c -> c/2     (3x3 stride-1)
#     m.0/cv2 :  c/2 -> c     (3x3 stride-1, +SiLU, +skip-add against y1)
#     cv2     :  3c -> c_out  (1x1)
#
# Three tiles (placement.PLACEMENT[block_name]["cv1" | "m_0_inner" | "cv2"]):
#     cv1         : 1x1 conv, kernel writes y0/y1 into two separate output fifos.
#     m_0_inner   : 2-kernel worker (m.0/cv1 -> m.0/cv2-with-internal-skip-add).
#                   Pattern mirrors mobilenet's build_2layer_skip (bn0) — same
#                   sliding-window over 3x3, just with conv3x3 instead of DW.
#     cv2         : 1x1 conv with 3-way channel concat in the kernel prologue.
#
# Fan-out: the `bot` output fifo from cv1 has TWO consumers (m_0_inner worker
# AND cv2 tile's "bot" input slot). IRON supports this via multiple .cons()
# calls; the fifo depth must accommodate both consumers' lag.
#
# Kernel objects used (kernels/*.cc, compiled with per-block KERNEL_SUFFIX):
#   yolo_c3k2_small_cv1_split.cc       - 1x1 + channel chunk
#   yolo_c3k2_small_m0_cv1.cc          - 3x3 stride-1 SiLU-bias (standard)
#   yolo_c3k2_small_m0_cv2_skip.cc     - 3x3 stride-1 SiLU-bias + INT8 skip-add
#   yolo_c3k2_small_cv2_concat3.cc     - 1x1 + 3-input channel concat
# ---------------------------------------------------------------------------
def _build_c3k2_small(block_name: str, act_in, manifest):
    blk = yolo_spec.block(block_name)
    assert blk.topology == "c3k2_small" and len(blk.layers) == 4, blk

    # Layers in order: cv1, m.0/cv1, m.0/cv2, cv2
    L_cv1, L_m0_cv1, L_m0_cv2, L_cv2 = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape  # spatial+input chans for the block
    twoc = L_cv1.out_shape[2]  # cv1 output = 2c
    c = twoc // 2  # half-channel count (y0/y1/y_m0 width)
    c_half = L_m0_cv1.out_shape[2]  # = c/2 (inner bottleneck width)
    out_c = L_cv2.out_shape[2]  # block output channel count
    assert (
        twoc == 2 * c and L_cv2.in_shape[2] == 3 * c
    ), f"{block_name}: channel arithmetic broken: 2c={twoc}, 3c={L_cv2.in_shape[2]}"

    meta_cv1 = _op_meta(manifest, L_cv1.manifest_name)
    meta_m0_cv1 = _op_meta(manifest, L_m0_cv1.manifest_name)
    meta_m0_cv2 = _op_meta(manifest, L_m0_cv2.manifest_name)
    meta_cv2 = _op_meta(manifest, L_cv2.manifest_name)

    # Weight + bias buffers (one Buffer per layer for clarity; mobilenet packs
    # them into per-block chains, which we'd do once kernels are written).
    def _wts(meta):
        sz = int(np.prod(meta["weights_shape"]))
        return (
            Buffer(
                _i8((sz,)), initial_value=_load_bin(meta["weights_file"], np.int8, sz)
            ),
            sz,
        )

    def _bias(meta, oc):
        return Buffer(
            _i32((oc,)), initial_value=_load_bin(meta["bias_file"], np.int32, oc)
        )

    wts_cv1, sz_cv1 = _wts(meta_cv1)
    wts_m0_cv1, sz_m0_cv1 = _wts(meta_m0_cv1)
    wts_m0_cv2, sz_m0_cv2 = _wts(meta_m0_cv2)
    wts_cv2, sz_cv2 = _wts(meta_cv2)
    bias_cv1 = _bias(meta_cv1, twoc)
    bias_m0_cv1 = _bias(meta_m0_cv1, c_half)
    bias_m0_cv2 = _bias(meta_m0_cv2, c)
    bias_cv2 = _bias(meta_cv2, out_c)

    rs_cv1 = meta_cv1["right_shift"]
    rs_m0_cv1 = meta_m0_cv1["right_shift"]
    rs_m0_cv2 = meta_m0_cv2["right_shift"]
    rs_cv2 = meta_cv2["right_shift"]

    # Per-layer SiLU LUTs. cv2 stays linear here — we compare against the
    # pre-SiLU output tensor (/model.N/cv2/conv/Conv_output_0_QL_Output).
    model_n = block_name[1:]

    def _silu_lut(layer):
        p = os.path.join(DATA_DIR, f"model.{model_n}", layer, "silu_lut.bin")
        data = np.fromfile(p, dtype=np.int8)
        assert data.size == 256, f"{p}: expected 256 bytes, got {data.size}"
        return Buffer(_i8((256,)), initial_value=data)

    silu_cv1 = _silu_lut("cv1")
    silu_m0_cv1 = _silu_lut("m.0/cv1")
    silu_m0_cv2 = _silu_lut("m.0/cv2")
    silu_cv2 = _silu_lut("cv2")

    # ----- ObjectFifos: inter-tile streams -----
    # cv1 outputs: two half-channel streams. bot is fanned out (m0 + cv2-slot1).
    # top depth=4 because cv2 stalls until m_0_inner's preamble completes; cv1
    # gets a few rows ahead in the meantime.
    top_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=4)
    # bot is fanned out (m_0_inner peeks 3 via acquire(3), cv2 also consumes
    # at 1/iter). Producer pool must accommodate peek_size + cv2 in-flight
    # buffer to avoid the "cv2-released-but-m_0_inner-still-peeking" deadlock
    # window — the same sliding-window-AcquireGE pattern documented in
    # placement.py's DESIGN RULES (m8 stage 4 example). Empirically:
    # depth=4 hangs, depth>=5 succeeds. Picked 6 for slack.
    bot_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=6)
    # m_0_inner output stream -> cv2 slot 2
    m0_out_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=2)
    # via_DMA on the block-output fifo so standalone (shim drain) and chain
    # (next block's input) share the same DMA-routed transport; see m0 above.
    out_fifo = ObjectFifo(_i8((in_w, 1, out_c)), depth=2, via_DMA=True)

    # ----- Kernels (all .o files TBD; see header comment) -----
    k_cv1_split = Kernel(
        f"yolo_c3k2_small_cv1_split_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_small_cv1_split_{block_name}.o",
        [
            _i8((in_w, 1, in_c)),  # input row
            _i8((sz_cv1,)),
            _i32((twoc,)),
            _i8((256,)),  # silu_lut
            _i8((in_w, 1, c)),  # out_top (first c chans)
            _i8((in_w, 1, c)),  # out_bot (second c chans)
            np.int32,
            np.int32,
            np.int32,  # inW, inC, outC (=2c)
            np.int32,  # right_shift
        ],
    )
    k_m0_cv1 = Kernel(
        f"yolo_c3k2_small_m0_cv1_conv2dk3_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_small_m0_cv1_{block_name}.o",
        [
            _i8((in_w, 1, c)),  # row[t-1]
            _i8((in_w, 1, c)),  # row[t]
            _i8((in_w, 1, c)),  # row[t+1]
            _i8((sz_m0_cv1,)),
            _i32((c_half,)),
            _i8((256,)),  # silu_lut
            _i8((in_w, 1, c_half)),  # intermediate row out
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    k_m0_cv2_skip = Kernel(
        f"yolo_c3k2_small_m0_cv2_skip_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_small_m0_cv2_skip_{block_name}.o",
        [
            _i8((in_w, 1, c_half)),  # row[t-1] (intermediate)
            _i8((in_w, 1, c_half)),  # row[t]
            _i8((in_w, 1, c_half)),  # row[t+1]
            _i8((sz_m0_cv2,)),
            _i32((c,)),
            _i8((256,)),  # silu_lut
            _i8((in_w, 1, c)),  # skip input (y1 row at matching t)
            _i8((in_w, 1, c)),  # output (y1 + SiLU(cv2_out))
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,  # skip_scale (right_shift for the add path)
        ],
    )
    k_cv2_concat = Kernel(
        f"yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_small_cv2_concat3_{block_name}.o",
        [
            _i8((in_w, 1, c)),  # top slice
            _i8((in_w, 1, c)),  # bot slice
            _i8((in_w, 1, c)),  # m0 slice
            _i8((sz_cv2,)),
            _i32((out_c,)),
            _i8((256,)),  # silu_lut
            _i8((in_w, 1, out_c)),  # output (post-SiLU)
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    # ----- Workers -----
    def cv1_fn(act_in_f, wts, bias, lut, top_p, bot_p, k):
        # 1x1: one input row -> one output row, split into two halves.
        for _ in range_(in_h):
            row_in = act_in_f.acquire(1)
            row_top = top_p.acquire(1)
            row_bot = bot_p.acquire(1)
            k(row_in, wts, bias, lut, row_top, row_bot, in_w, in_c, twoc, rs_cv1)
            act_in_f.release(1)
            top_p.release(1)
            bot_p.release(1)

    w_cv1 = Worker(
        cv1_fn,
        fn_args=[
            act_in.cons(),
            wts_cv1,
            bias_cv1,
            silu_cv1,
            top_fifo.prod(),
            bot_fifo.prod(),
            k_cv1_split,
        ],
        tile=placement.PLACEMENT[block_name]["cv1"],
    )

    # Inter-kernel fifo on the m_0_inner tile (m.0/cv1 → m.0/cv2).
    # Depth=3 supports m.0/cv2's 3-row sliding window (1 ahead + current + 1 behind).
    f12 = ObjectFifo(_i8((in_w, 1, c_half)), depth=3)

    def m0_inner_fn(
        bot_c, wts1, bias1, lut1, wts2, bias2, lut2, m0_out_p, p12, c12, k1, k2
    ):
        # Two stacked 3x3 sliding windows + skip-add. Adapted from mobilenet's
        # build_3layer (where only L2 slides) — here BOTH layers slide. The
        # `bot` input is consumed by THIS worker for two purposes simultaneously:
        # m.0/cv1's 3-row window + m.0/cv2's per-output skip-add. Both pulls
        # come from `bot_c` (this worker's own consumer view of bot_fifo).
        #
        # State at start of middle iter `i` (i=1..H-3):
        #   bot_c view  : [bot[i],  bot[i+1], bot[i+2]]    (3 rows live)
        #   c12 view    : [int[i-1], int[i],  int[i+1]]    (3 rows live)
        # Per iter: acquire 1 new bot, produce 1 new intermediate via m.0/cv1,
        # produce 1 output via m.0/cv2 (using 3 ints + bot[i] as skip),
        # release 1 bot + 1 intermediate.

        def _l1(top, mid, bot, border):
            """m.0/cv1 step: 3 input rows (with border padding) → 1 intermediate row."""
            row_int = p12.acquire(1)
            k1(
                top,
                mid,
                bot,
                wts1,
                bias1,
                lut1,
                row_int,
                in_w,
                c,
                c_half,
                3,
                3,
                border,
                rs_m0_cv1,
                0,
            )
            p12.release(1)

        def _l2(top, mid, bot, border, skip_row):
            """m.0/cv2-with-skip step: 3 intermediate rows + 1 skip row → 1 output row.
            The skip-add (integer add + clip) and SiLU LUT live inside the
            kernel. `skip_scale` is reserved for future cross-scale rescaling;
            for m2/m4 all chain scales match → unused."""
            row_out = m0_out_p.acquire(1)
            k2(
                top,
                mid,
                bot,
                wts2,
                bias2,
                lut2,
                skip_row,
                row_out,
                in_w,
                c_half,
                c,
                3,
                3,
                border,
                rs_m0_cv2,
                0,
            )
            m0_out_p.release(1)

        # ── preamble: produce output[0] ─────────────────────────────────────
        # Need bot[0,1,2] for L1; m.0/cv2 uses border=top with int[0,0,1].
        bots = bot_c.acquire(3)
        _l1(bots[0], bots[0], bots[1], border=0)  # → intermediate[0]
        _l1(bots[0], bots[1], bots[2], border=1)  # → intermediate[1]
        ints = c12.acquire(2)
        _l2(ints[0], ints[0], ints[1], border=0, skip_row=bots[0])  # → output[0]
        bot_c.release(1)  # release bot[0]
        # intermediates[0,1] stay live; they're needed for output[1].

        # ── middle (in_h - 3 iters): produce output[1..in_h-3] ──────────────
        for _ in range_(in_h - 3):
            bots = bot_c.acquire(3)  # peek [i, i+1, i+2]
            _l1(bots[0], bots[1], bots[2], border=1)  # → intermediate[i+1]
            ints = c12.acquire(3)  # peek [i-1, i, i+1]
            _l2(ints[0], ints[1], ints[2], border=1, skip_row=bots[0])  # → output[i]
            bot_c.release(1)  # advance bot cursor
            c12.release(1)  # advance int cursor

        # ── postamble step 1: produce output[H-2] ───────────────────────────
        # No new bot to fetch (cursor at H-2; only bot[H-2, H-1] remain).
        # Use L1 border=bot to replicate the last input row.
        bots = bot_c.acquire(2)
        _l1(bots[0], bots[1], bots[1], border=2)  # → intermediate[H-1]
        ints = c12.acquire(3)  # [H-3, H-2, H-1]
        _l2(ints[0], ints[1], ints[2], border=1, skip_row=bots[0])  # → output[H-2]
        bot_c.release(1)
        c12.release(1)

        # ── postamble step 2: produce output[H-1] ───────────────────────────
        # No new intermediate; replicate the last intermediate via L2 border=bot.
        # acquire(1) returns the row directly (not a list); indexing into it
        # would give a 1xCxC subview — pass `bot_last` straight through.
        bot_last = bot_c.acquire(1)
        ints = c12.acquire(2)  # [H-2, H-1]
        _l2(ints[0], ints[1], ints[1], border=2, skip_row=bot_last)  # → output[H-1]
        bot_c.release(1)
        c12.release(2)

    w_m0_inner = Worker(
        m0_inner_fn,
        fn_args=[
            bot_fifo.cons(depth=4),  # one of bot's two consumers
            wts_m0_cv1,
            bias_m0_cv1,
            silu_m0_cv1,
            wts_m0_cv2,
            bias_m0_cv2,
            silu_m0_cv2,
            m0_out_fifo.prod(),
            f12.prod(),
            f12.cons(),
            k_m0_cv1,
            k_m0_cv2_skip,
        ],
        tile=placement.PLACEMENT[block_name]["m_0_inner"],
    )

    def cv2_fn(top_c, bot_c, m0_c, wts, bias, lut, out_p, k):
        # cv2 now applies SiLU LUT — output is the block's post-SiLU int8
        # (matches /model.N/cv2/act/Mul_output_0_QuantizeLinear_Output) so
        # downstream blocks see the correct activations when chained.
        for _ in range_(in_h):
            r_top = top_c.acquire(1)
            r_bot = bot_c.acquire(1)
            r_m0 = m0_c.acquire(1)
            r_out = out_p.acquire(1)
            k(r_top, r_bot, r_m0, wts, bias, lut, r_out, in_w, 3 * c, out_c, rs_cv2)
            top_c.release(1)
            bot_c.release(1)
            m0_c.release(1)
            out_p.release(1)

    w_cv2 = Worker(
        cv2_fn,
        fn_args=[
            top_fifo.cons(),
            bot_fifo.cons(depth=4),  # the OTHER bot consumer
            m0_out_fifo.cons(),
            wts_cv2,
            bias_cv2,
            silu_cv2,
            out_fifo.prod(),
            k_cv2_concat,
        ],
        tile=placement.PLACEMENT[block_name]["cv2"],
    )

    return out_fifo, [w_cv1, w_m0_inner, w_cv2]


# ---------------------------------------------------------------------------
# m10 (head): conv1x1 256->1280 (SiLU) + GAP 16x16->1x1 + Gemm 1280->2
#
# Two tiles per placement.PLACEMENT["m10"]:
#   conv_pool : fused 1x1 conv + SiLU + spatial avgpool (mobilenet post_l1 shape)
#   linear    : Gemm 1280->2 (binary classifier)
#
# Heavyweight design point: m10/conv weights are 327,680 B — way over the
# per-tile L1 budget — so they're staged on a memtile and chunked via
# StaticWeightStream, mirroring mobilenet post_l1's `post_l1_pb`. We split
# output channels into 16 chunks of 80 channels each (20,480 B/chunk; vs
# mobilenet's 8 chunks at 9,600 B — both are well under the per-tile budget).
# The chunk count is tunable when we measure tile mem usage on Linux.
#
# Gemm weights are tiny (2,560 B), so the linear tile uses a regular Buffer.
# The conv_pool output is i8 (SiLU-applied, spatially-averaged, requantized
# inside the kernel) — no intermediate uint16 wide-accumulator type since
# our binary classifier's tile budget is generous and we'd rather keep types
# uniform across the network for the IRON Python sketch.
#
# Kernels TBD:
#   yolo_m10_conv2dk1_silu_xy_pool.o   - 1x1 + SiLU LUT + 16x16 average pool
#   yolo_m10_linear_gemm.o             - Gemm 1280x2 (with bias + requant)
# ---------------------------------------------------------------------------
def _build_head(block_name, act_in, manifest):
    assert block_name == "m10"
    blk = yolo_spec.block("m10")
    L_conv, L_pool, L_lin, L_softmax = blk.layers
    in_w, in_h, in_c = L_conv.in_shape  # (16, 16, 256)
    expand_c = L_conv.out_shape[2]  # 1280
    out_c = L_lin.out_shape[0]  # 2

    meta_conv = _op_meta(manifest, L_conv.manifest_name)
    meta_lin = _op_meta(manifest, L_lin.manifest_name)
    meta_softmax = _op_meta(manifest, L_softmax.manifest_name)
    rs_conv = meta_conv["right_shift"]  # 8
    rs_lin = meta_lin["right_shift"]  # 10
    # softmax has no right_shift (LUT-driven); uses in_log2_scale to size the
    # exp table. Output scale is our choice (2^-7 for [0,1) → [0,127] INT8).
    softmax_in_log2 = meta_softmax["in_log2_scale"]  # = -3 per extractor

    # ----- conv_pool: weight staging via memtile -----
    # Mobilenet's StaticWeightStream lives in `lowlevel_dma.py` from their
    # mobilenet folder — when we stage on Linux, we'll either import that
    # module or fold it into our package. Import path noted but unused at
    # sketch time (this file already won't import on Mac).
    n_splits = 16
    conv_wts_total = expand_c * in_c  # 327,680
    conv_wts_chunk = conv_wts_total // n_splits  # 20,480
    assert conv_wts_total % n_splits == 0

    conv_wts_data = _load_bin(meta_conv["weights_file"], np.int8, conv_wts_total)
    conv_bias_data = _load_bin(meta_conv["bias_file"], np.int32, expand_c)
    conv_bias_buf = Buffer(_i32((expand_c,)), initial_value=conv_bias_data)

    # Memtile placement: in chain context cols 2-6 are all claimed by m8
    # streams ((3,1)–(6,1)) or m9 streams ((1,1) cv2, (2,1) ffn.0, (7,1)
    # cv1). Use (0,1) for m10's conv wts — fully free, just costs a
    # stream-switch hop to the fused_tile at (2,2).
    fused_tile = placement.PLACEMENT["m10"]["fused"]
    conv_wts_pb = StaticWeightStream(
        obj_type=_i8((conv_wts_total,)),
        initial_value=conv_wts_data,
        name="m10_conv_wts",
        recv_type=_i8((conv_wts_chunk,)),
        repeat_count=in_h,  # n_splits chunks per row × in_h rows
        memtile_placement=Tile(0, 1),
        compute_placement=fused_tile,
        mem_lock_id=0,
        comp_lock_id=0,
    )

    # ----- ObjectFifos -----
    # m10 fused onto one tile per placement — no inter-tile pool_out fifo;
    # pool intermediate stays in a local Buffer. Same for gemm-to-softmax.
    # block output: 2 i8 probs per sample. Padded to 4 bytes because shim
    # DMA requires 4-byte aligned transfer length. softmax kernel only
    # writes the first out_c (=2) entries; the trailing 2 are unused.
    OUT_PAD = 4
    assert out_c <= OUT_PAD
    out_fifo = ObjectFifo(_i8((OUT_PAD,)), depth=2)
    pool_scratch = Buffer(
        _i8((expand_c,)),
        initial_value=np.zeros(expand_c, dtype=np.int8),
        name="m10_pool_scratch",
    )
    gemm_scratch = Buffer(
        _i8((out_c,)),
        initial_value=np.zeros(out_c, dtype=np.int8),
        name="m10_gemm_scratch",
    )

    # ----- Kernels -----
    # Persistent i32 accumulator on the compute tile (5KB). Holds the
    # post-SiLU spatial sum across all (yi, wi) calls within one sample,
    # finalized on the last call (yi=in_h-1, wi=n_splits-1) into elem_out.
    conv_accum_buf = Buffer(
        _i32((expand_c,)),
        initial_value=np.zeros(expand_c, dtype=np.int32),
        name="m10_conv_pool_accum",
    )
    # SiLU LUT generated by gen_yolo_silu_luts.py from model.10's
    # HardSiLU chain (conv_out=2^-1, silu_out=2^-2).
    silu_lut_path = os.path.join(DATA_DIR, "model.10", "silu_lut.bin")
    silu_lut_data = np.fromfile(silu_lut_path, dtype=np.int8)
    assert silu_lut_data.size == 256, (silu_lut_path, silu_lut_data.size)
    conv_silu_lut_buf = Buffer(
        _i8((256,)), initial_value=silu_lut_data, name="m10_conv_silu_lut"
    )

    k_conv_pool = Kernel(
        "yolo_m10_conv2dk1_silu_xy_pool_i8_i8",
        "yolo_m10_conv2dk1_silu_xy_pool.o",
        [
            _i8((in_w, 1, in_c)),  # one input row
            _i8((conv_wts_chunk,)),  # one weight chunk
            _i32((expand_c,)),  # bias
            _i8((256,)),  # silu LUT
            _i32((expand_c,)),  # i32 accumulator
            _i8((expand_c,)),  # i8 final output
            np.int32,
            np.int32,
            np.int32,
            np.int32,  # in_w, in_c, expand_c, in_h
            np.int32,  # right_shift
            np.int32,
            np.int32,
            np.int32,  # yi, n_splits, wi
        ],
    )

    # Gemm: load full weights as a single Buffer since 2,560 B fits comfortably.
    lin_wts_data = _load_bin(meta_lin["weights_file"], np.int8, expand_c * out_c)
    lin_bias_data = _load_bin(meta_lin["bias_file"], np.int32, out_c)
    lin_wts_buf = Buffer(_i8((expand_c * out_c,)), initial_value=lin_wts_data)
    lin_bias_buf = Buffer(_i32((out_c,)), initial_value=lin_bias_data)

    k_gemm = Kernel(
        "yolo_m10_linear_gemm_i8_i8",
        "yolo_m10_linear_gemm.o",
        [
            _i8((expand_c,)),  # input vector (1280)
            _i8((expand_c * out_c,)),  # weights (2x1280 flat)
            _i32((out_c,)),  # bias (2)
            _i8((out_c,)),  # output (2 logits)
            np.int32,
            np.int32,  # in_dim (=1280), out_dim (=2)
            np.int32,  # right_shift
        ],
    )

    # ----- Final softmax (on the linear tile, chained after Gemm) -----
    # 2-class softmax: subtract max, exp via fp32 LUT, normalize. INT8 in,
    # INT8 out at scale 2^-7. LUT computed host-side from softmax_in_log2:
    #     softmax_exp_lut[idx] = exp((idx - 128) * 2^softmax_in_log2)
    # Same pattern as m9's softmax_row but flat (n_classes=2).
    softmax_exp_in_scale = 2.0**softmax_in_log2
    softmax_lut_data = np.array(
        [np.exp((i - 128) * softmax_exp_in_scale) for i in range(256)],
        dtype=np.float32,
    )
    _f32 = lambda shape: np.ndarray[shape, np.dtype[np.float32]]
    softmax_lut_buf = Buffer(
        _f32((256,)), initial_value=softmax_lut_data, name="m10_softmax_exp_lut"
    )

    k_softmax = Kernel(
        "yolo_m10_softmax_i8_i8",
        "yolo_m10_softmax.o",
        [
            _i8((out_c,)),  # input logits
            _f32((256,)),  # exp LUT
            _i8((OUT_PAD,)),  # output probs (scale 2^-7), padded to 4 bytes for shim
            np.int32,
            np.int32,  # n_classes, in_log2_scale
        ],
    )

    # ----- Single fused worker (conv+pool → gemm → softmax) on one tile -----
    # In chain context only (2,2) is free for m10, so we collapse the
    # original 2-tile (conv_pool + linear) design into one worker that
    # runs the full pipeline sequentially per sample. No inter-tile
    # fifo overhead; same total wall time since the work is serial.
    def fused_fn(
        act_in_f,
        wts_pb,
        conv_bias,
        conv_silu,
        accum,
        pool,
        lin_wts,
        lin_bias,
        gemm_out,
        sm_lut,
        out_p,
        k_conv,
        k_gemm_,
        k_softmax_,
    ):
        # Phase 1: conv + SiLU + GAP via the streamed conv_pool kernel.
        for yi in range_(in_h):
            elem_in = act_in_f.acquire(1)
            for wi in range_(n_splits):
                wts_chunk = wts_pb.acquire(1)
                k_conv(
                    elem_in,
                    wts_chunk,
                    conv_bias,
                    conv_silu,
                    accum,
                    pool,
                    in_w,
                    in_c,
                    expand_c,
                    in_h,
                    rs_conv,
                    yi,
                    n_splits,
                    wi,
                )
                wts_pb.release(1)
            act_in_f.release(1)
        # Phase 2: Gemm 1280→2 → softmax → output.
        out_row = out_p.acquire(1)
        k_gemm_(pool, lin_wts, lin_bias, gemm_out, expand_c, out_c, rs_lin)
        k_softmax_(gemm_out, sm_lut, out_row, out_c, softmax_in_log2)
        out_p.release(1)

    w_fused = Worker(
        fused_fn,
        fn_args=[
            act_in.cons(),
            conv_wts_pb,
            conv_bias_buf,
            conv_silu_lut_buf,
            conv_accum_buf,
            pool_scratch,
            lin_wts_buf,
            lin_bias_buf,
            gemm_scratch,
            softmax_lut_buf,
            out_fifo.prod(),
            k_conv_pool,
            k_gemm,
            k_softmax,
        ],
        tile=fused_tile,
        dynamic_objfifo_lowering=True,
    )

    return out_fifo, [w_fused]


# ---------------------------------------------------------------------------
# c3k2_heavy (m6, m8) — outer C2f-style block whose inner module is a C3k
# (not just a Bottleneck), with the C3k containing two sequential Bottlenecks.
#
# Topology (ultralytics C3k2(c3k=True) + C3k):
#     # outer C2f-style
#     y0, y1 = chunk(cv1(x), 2, dim=ch)
#     y_m0   = m.0_C3k(y1)
#     out    = cv2(cat([y0, y1, y_m0], dim=ch))
#
#     # m.0 is C3k:
#     a = m.0/cv1(y1)                       # parallel branch A
#     b = m.0/cv2(y1)                       # parallel branch B (same input as A!)
#     a = m.0/m/m.0(a)                      # inner Bottleneck #0 (e=1.0, no narrowing)
#     a = m.0/m/m.1(a)                      # inner Bottleneck #1
#     m_0_out = m.0/cv3(cat([a, b], dim=ch))
#
# Channel widths (c = out_c // 2; c' = c // 2; m6 -> c=64, c'=32; m8 -> c=128, c'=64):
#     cv1         : c_in -> 2c
#     y0, y1      : each c
#     m.0/cv1     : c   -> c'           (parallel branch A)
#     m.0/cv2     : c   -> c'           (parallel branch B)
#     m.0/m/m.{0,1} cv1+cv2 : c' -> c'  (3x3, uniform; +SiLU; +internal residual)
#     m.0/cv3     : 2c' = c -> c        (1x1 fuse of cat([a_final, b]))
#     cv2         : 3c -> out_c         (1x1 fuse of cat([y0, y1, m_0_out]))
#
# Five tiles per placement.PLACEMENT[block_name]:
#     cv1, m_0_split, inner_pair_0, inner_pair_1, cv3_cv2
#
# y1's fifo fan-out: 3 consumers — m_0_split (computes both parallel branches),
# AND cv3_cv2 (cv2 input slot 1). Bumped to depth 6.
#
# Kernels TBD:
#   yolo_<blk>_cv1_split.o                - already needed for c3k2_small too
#   yolo_<blk>_m_0_split.o                - 1x1 with 2 parallel-branch outputs
#   yolo_<blk>_inner_pair_<i>_cv1.o       - 3x3 SiLU+bias (standard inner)
#   yolo_<blk>_inner_pair_<i>_cv2_skip.o  - 3x3 SiLU+bias + skip-add (against pair input)
#   yolo_<blk>_cv3_concat2.o              - 1x1 + 2-input concat
#   yolo_<blk>_cv2_concat3.o              - 1x1 + 3-input concat (already needed for c3k2_small)
# ---------------------------------------------------------------------------
def _build_c3k2_heavy(block_name: str, act_in, manifest):
    blk = yolo_spec.block(block_name)
    assert blk.topology == "c3k2_heavy" and len(blk.layers) == 9, blk
    (
        L_cv1,
        L_m0_cv1,
        L_m0_cv2,
        L_p0_cv1,
        L_p0_cv2,
        L_p1_cv1,
        L_p1_cv2,
        L_m0_cv3,
        L_cv2,
    ) = blk.layers

    in_w, in_h, in_c = L_cv1.in_shape
    twoc = L_cv1.out_shape[2]  # 2c
    c = twoc // 2  # cv1 chunk width
    cp = L_m0_cv1.out_shape[2]  # c' (e.g. 32 for m6, 64 for m8)
    assert L_m0_cv2.out_shape[2] == cp
    assert L_m0_cv1.in_shape[2] == c and L_m0_cv2.in_shape[2] == c  # both take y1=c
    assert L_p0_cv1.in_shape[2] == cp and L_p0_cv2.out_shape[2] == cp  # uniform inside
    assert L_m0_cv3.in_shape[2] == 2 * cp == c  # cat(a, b) feeds cv3
    out_c = L_cv2.out_shape[2]
    assert L_cv2.in_shape[2] == 3 * c

    # ----- Per-layer metadata helpers (same shape as c3k2_small section) -----
    def _wts_for(meta):
        sz = int(np.prod(meta["weights_shape"]))
        return (
            Buffer(
                _i8((sz,)), initial_value=_load_bin(meta["weights_file"], np.int8, sz)
            ),
            sz,
        )

    def _bias_for(meta, oc):
        return Buffer(
            _i32((oc,)), initial_value=_load_bin(meta["bias_file"], np.int32, oc)
        )

    m_cv1 = _op_meta(manifest, L_cv1.manifest_name)
    m_m0c1 = _op_meta(manifest, L_m0_cv1.manifest_name)
    m_m0c2 = _op_meta(manifest, L_m0_cv2.manifest_name)
    m_p0c1 = _op_meta(manifest, L_p0_cv1.manifest_name)
    m_p0c2 = _op_meta(manifest, L_p0_cv2.manifest_name)
    m_p1c1 = _op_meta(manifest, L_p1_cv1.manifest_name)
    m_p1c2 = _op_meta(manifest, L_p1_cv2.manifest_name)
    m_m0c3 = _op_meta(manifest, L_m0_cv3.manifest_name)
    m_cv2 = _op_meta(manifest, L_cv2.manifest_name)

    # Per-layer SiLU LUTs (9 per heavy block) — extracted by
    # gen_yolo_silu_luts.py to data/model.N/<layer>/silu_lut.bin.
    model_n_str = block_name[1:]

    def _silu_lut(layer):
        p = os.path.join(DATA_DIR, f"model.{model_n_str}", layer, "silu_lut.bin")
        data = np.fromfile(p, dtype=np.int8)
        assert data.size == 256, f"{p}: expected 256 bytes, got {data.size}"
        return Buffer(_i8((256,)), initial_value=data)

    silu_cv1 = _silu_lut("cv1")
    silu_m0_cv1 = _silu_lut("m.0/cv1")
    silu_m0_cv2 = _silu_lut("m.0/cv2")
    silu_p0_cv1 = _silu_lut("m.0/m/m.0/cv1")
    silu_p0_cv2 = _silu_lut("m.0/m/m.0/cv2")
    silu_p1_cv1 = _silu_lut("m.0/m/m.1/cv1")
    silu_p1_cv2 = _silu_lut("m.0/m/m.1/cv2")
    silu_m0_cv3 = _silu_lut("m.0/cv3")
    silu_cv2 = _silu_lut("cv2")

    # ----- ObjectFifos (inter-tile) -----
    # Depths sized to each consumer's max-acquire (sliding windows need
    # peek-3 for 3x3 conv); kept tight against per-tile 16-BD budget.
    # split_b is the exception: cv3 can't consume split_b until inner_1_out
    # is ready (which needs both inner pairs to fully process all rows),
    # but m_0_split produces split_a + split_b in lockstep. A small split_b
    # ping-pong back-pressures m_0_split → starves inner pairs → deadlock.
    # in_h slots lets m_0_split run to completion ahead of cv3.
    # Streaming decision needs to be known before fifo allocation so the
    # m8 path can declare its separate bot_to_cv2 fifo (no fanout).
    _stream_outer = block_name == "m8"
    n_cv1_chunks = 8 if _stream_outer else 1
    n_cv2_chunks = 8 if _stream_outer else 1

    top_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=4)
    # bot_fifo is fanned out to TWO consumers (m_0_split's 3x3 sliding window
    # AcquireGE(3) AND cv3_cv2's 1-row read for the concat). Each .cons() asks
    # for depth=6; the producer also needs depth=6 so it can stay 6 rows
    # ahead while one consumer waits on the other -- otherwise the trailing
    # consumer's window starves. Same shape as the c3k2_small fix
    # (commit 79343963, 4 -> 6) and depends on the IRON ObjectFifo depth-
    # collapse fix (upstream PR #3096) for the per-handle depth to actually
    # reach the lowering instead of being auto-minimized to ping-pong.
    bot_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=6)
    # m8 (_stream_outer): cv1 writes bot to two separate fifos to avoid
    # fanout — bot_fifo feeds m_0_split via DMA, bot_to_cv2_fifo feeds cv2
    # via shared mem from cv1's adjacent tile. Eliminates the fanout that
    # was forcing both consumers onto DMA channels and busting cv2's
    # S2MM budget.
    bot_to_cv2_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=4) if _stream_outer else None
    split_a = ObjectFifo(_i8((in_w, 1, cp)), depth=4)
    split_b = ObjectFifo(_i8((in_w, 1, cp)), depth=in_h)
    inner_0_out = ObjectFifo(_i8((in_w, 1, cp)), depth=4)
    inner_1_out = ObjectFifo(_i8((in_w, 1, cp)), depth=2)
    out_fifo = ObjectFifo(_i8((in_w, 1, out_c)), depth=2, via_DMA=True)

    # m6/m8 cross-scale skip-add ratios: s_y/s_add = 0.5, s_cv2/s_add = 1.0
    # for both inner pairs in both blocks. Pre-rshift integer formula:
    #   out = banker_srs(y * 1 + cv2silu * 2, 1)
    SKIP_Y_MULT, SKIP_CV2_MULT, SKIP_RSH = 1, 2, 1

    # Streaming decision already set above (needed before fifo decls). m8
    # has cv1 (~65 KB) and cv2 (~98 KB) over the per-tile L1 budget; chunk
    # them via StaticWeightStream from same-column memtiles. m6 fits both
    # in static Buffers (cv1 ~16 KB, cv2 ~24 KB).

    # ----- Kernels -----
    # cv1 + channel split.
    sz_cv1 = twoc * in_c  # OIYXI8O8 or OIYX, raw bytes
    data_cv1 = _load_bin(m_cv1["weights_file"], np.int8, sz_cv1)
    bias_cv1 = _bias_for(m_cv1, twoc)
    if _stream_outer:
        chunk_sz_cv1 = sz_cv1 // n_cv1_chunks
        cv1_tile = placement.PLACEMENT[block_name]["cv1"]
        wts_cv1 = StaticWeightStream(
            obj_type=_i8((sz_cv1,)),
            initial_value=data_cv1,
            name=f"{block_name}_cv1_wts",
            recv_type=_i8((chunk_sz_cv1,)),
            repeat_count=in_h,
            memtile_placement=Tile(cv1_tile.col, 1),
            compute_placement=cv1_tile,
            mem_lock_id=0,
            comp_lock_id=0,
        )
        k_cv1_split = Kernel(
            f"yolo_c3k2_small_cv1_split_streamed_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_small_cv1_split_streamed_{block_name}.o",
            [
                _i8((in_w, 1, in_c)),
                _i8((chunk_sz_cv1,)),
                _i32((twoc,)),
                _i8((256,)),  # silu_lut
                _i8((in_w, 1, c)),  # out_top
                _i8((in_w, 1, c)),  # out_bot_a (for m_0_split)
                _i8((in_w, 1, c)),  # out_bot_b (for cv2)
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
            # in_w, in_c, twoc, n_chunks, chunk_idx, right_shift
        )
    else:
        wts_cv1 = Buffer(_i8((sz_cv1,)), initial_value=data_cv1)
        k_cv1_split = Kernel(
            f"yolo_c3k2_small_cv1_split_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_small_cv1_split_{block_name}.o",
            [
                _i8((in_w, 1, in_c)),
                _i8((sz_cv1,)),
                _i32((twoc,)),
                _i8((256,)),  # silu_lut
                _i8((in_w, 1, c)),
                _i8((in_w, 1, c)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

    # m.0/cv1 + m.0/cv2 fused on m_0_split tile — single kernel, one input row,
    # two outputs (parallel branches with separate weights/bias).
    wts_m0c1, sz_m0c1 = _wts_for(m_m0c1)
    bias_m0c1 = _bias_for(m_m0c1, cp)
    wts_m0c2, sz_m0c2 = _wts_for(m_m0c2)
    bias_m0c2 = _bias_for(m_m0c2, cp)
    rs_m0c1 = m_m0c1["right_shift"]
    rs_m0c2 = m_m0c2["right_shift"]
    k_m0_split = Kernel(
        f"yolo_c3k2_heavy_m_0_split_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_heavy_m_0_split_{block_name}.o",
        [
            _i8((in_w, 1, c)),  # input row (y1)
            _i8((sz_m0c1,)),
            _i32((cp,)),
            _i8((256,)),  # branch A wts+bias+lut
            _i8((sz_m0c2,)),
            _i32((cp,)),
            _i8((256,)),  # branch B wts+bias+lut
            _i8((in_w, 1, cp)),  # output A
            _i8((in_w, 1, cp)),  # output B
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],  # inW, inC, outC_A, outC_B, rs_a, rs_b
    )

    # Bottleneck-pair helper: 2-kernel sliding window + internal skip-add, same
    # shape as c3k2_small's m_0_inner but with uniform widths (c' -> c' -> c').
    # Both inner pairs share the same shape signature → one Kernel per block
    # serves both. For m8 the 3x3 weights are too big for a single tile (36KB
    # each × 2 = 73KB > 64KB L1), so each pair-conv kernel runs streamed-OC
    # against per-chunk weights from memtile (n_pair_chunks chunks per row).
    sz_p_cv1 = int(np.prod(m_p0c1["weights_shape"]))
    sz_p_cv2 = int(np.prod(m_p0c2["weights_shape"]))
    n_pair_chunks = 4 if _stream_outer else 1
    if _stream_outer:
        chunk_sz_p_cv1 = sz_p_cv1 // n_pair_chunks
        chunk_sz_p_cv2 = sz_p_cv2 // n_pair_chunks
        k_pair_cv1 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv1_streamed_conv2dk3_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_heavy_inner_pair_cv1_streamed_{block_name}.o",
            [_i8((in_w, 1, cp))] * 3
            + [_i8((chunk_sz_p_cv1,)), _i32((cp,)), _i8((256,)), _i8((in_w, 1, cp))]
            + [np.int32] * 9,  # in_w..rs + n_chunks + chunk_idx
        )
        k_pair_cv2_skip = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_streamed_{block_name}.o",
            [_i8((in_w, 1, cp))] * 3
            + [
                _i8((chunk_sz_p_cv2,)),
                _i32((cp,)),
                _i8((256,)),
                _i8((in_w, 1, cp)),  # skip (pair input row)
                _i8((in_w, 1, cp)),
            ]  # output row
            + [np.int32] * 12,  # ...+ n_chunks + chunk_idx + 3 skip args
        )
    else:
        k_pair_cv1 = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv1_conv2dk3_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_heavy_inner_pair_cv1_{block_name}.o",
            [_i8((in_w, 1, cp))] * 3
            + [_i8((sz_p_cv1,)), _i32((cp,)), _i8((256,)), _i8((in_w, 1, cp))]
            + [np.int32] * 8,
        )
        k_pair_cv2_skip = Kernel(
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_heavy_inner_pair_cv2_skip_{block_name}.o",
            [_i8((in_w, 1, cp))] * 3
            + [
                _i8((sz_p_cv2,)),
                _i32((cp,)),
                _i8((256,)),
                _i8((in_w, 1, cp)),  # skip (pair input row)
                _i8((in_w, 1, cp)),
            ]  # output row
            + [np.int32] * 10,
        )

    def _build_inner_pair(
        name, prev_fifo, out_fifo_local, meta_a, meta_b, tile_key, lut_prefix
    ):
        bias_a = _bias_for(meta_a, cp)
        bias_b = _bias_for(meta_b, cp)
        rs_a = meta_a["right_shift"]
        rs_b = meta_b["right_shift"]
        lut_a = _silu_lut(lut_prefix + "/cv1")
        lut_b = _silu_lut(lut_prefix + "/cv2")
        ka, kb = k_pair_cv1, k_pair_cv2_skip
        tile = placement.PLACEMENT[block_name][tile_key]
        f_inner = ObjectFifo(_i8((in_w, 1, cp)), depth=3)

        if _stream_outer:
            # Stream per-pair cv1 + cv2 weights from a same-column memtile.
            # Each pair gets its own pair of streams; n_pair_chunks chunks
            # per output row delivered via StaticWeightStream.
            data_a = _load_bin(meta_a["weights_file"], np.int8, sz_p_cv1)
            data_b = _load_bin(meta_b["weights_file"], np.int8, sz_p_cv2)
            ws_a = StaticWeightStream(
                obj_type=_i8((sz_p_cv1,)),
                initial_value=data_a,
                name=f"{block_name}_{tile_key}_cv1_wts",
                recv_type=_i8((chunk_sz_p_cv1,)),
                repeat_count=in_h,
                memtile_placement=Tile(tile.col, 1),
                compute_placement=tile,
                # Each StaticWeightStream uses 2 consecutive locks per
                # endpoint (ping-pong). Space by 2 across the 4 streams in
                # this memtile (2 pairs × 2 convs).
                mem_lock_id=0 if tile_key == "inner_pair_0" else 4,
                comp_lock_id=0 if tile_key == "inner_pair_0" else 0,
            )
            ws_b = StaticWeightStream(
                obj_type=_i8((sz_p_cv2,)),
                initial_value=data_b,
                name=f"{block_name}_{tile_key}_cv2_wts",
                recv_type=_i8((chunk_sz_p_cv2,)),
                repeat_count=in_h,
                memtile_placement=Tile(tile.col, 1),
                compute_placement=tile,
                mem_lock_id=2 if tile_key == "inner_pair_0" else 6,
                comp_lock_id=2 if tile_key == "inner_pair_0" else 2,
            )
        else:
            wts_a, _ = _wts_for(meta_a)
            wts_b, _ = _wts_for(meta_b)
            ws_a, ws_b = wts_a, wts_b

        def pair_fn(
            in_c_view, ws_a, bs_a, la, ws_b, bs_b, lb, out_p, p_inner, c_inner, k1, k2
        ):
            # Identical schedule to c3k2_small's m_0_inner_fn, just with cp/cp/cp widths.
            if _stream_outer:

                def _la(top, mid, bot, border):
                    row_int = p_inner.acquire(1)
                    for wi in range_(n_pair_chunks):
                        chunk_a = ws_a.acquire(1)
                        k1(
                            top,
                            mid,
                            bot,
                            chunk_a,
                            bs_a,
                            la,
                            row_int,
                            in_w,
                            cp,
                            cp,
                            3,
                            3,
                            border,
                            rs_a,
                            n_pair_chunks,
                            wi,
                        )
                        ws_a.release(1)
                    p_inner.release(1)

                def _lb(top, mid, bot, border, skip_row):
                    row_out = out_p.acquire(1)
                    for wi in range_(n_pair_chunks):
                        chunk_b = ws_b.acquire(1)
                        k2(
                            top,
                            mid,
                            bot,
                            chunk_b,
                            bs_b,
                            lb,
                            skip_row,
                            row_out,
                            in_w,
                            cp,
                            cp,
                            3,
                            3,
                            border,
                            rs_b,
                            n_pair_chunks,
                            wi,
                            SKIP_Y_MULT,
                            SKIP_CV2_MULT,
                            SKIP_RSH,
                        )
                        ws_b.release(1)
                    out_p.release(1)

            else:

                def _la(top, mid, bot, border):
                    row_int = p_inner.acquire(1)
                    k1(
                        top,
                        mid,
                        bot,
                        ws_a,
                        bs_a,
                        la,
                        row_int,
                        in_w,
                        cp,
                        cp,
                        3,
                        3,
                        border,
                        rs_a,
                        0,
                    )
                    p_inner.release(1)

                def _lb(top, mid, bot, border, skip_row):
                    row_out = out_p.acquire(1)
                    k2(
                        top,
                        mid,
                        bot,
                        ws_b,
                        bs_b,
                        lb,
                        skip_row,
                        row_out,
                        in_w,
                        cp,
                        cp,
                        3,
                        3,
                        border,
                        rs_b,
                        SKIP_Y_MULT,
                        SKIP_CV2_MULT,
                        SKIP_RSH,
                    )
                    out_p.release(1)

            # preamble
            bots = in_c_view.acquire(3)
            _la(bots[0], bots[0], bots[1], border=0)
            _la(bots[0], bots[1], bots[2], border=1)
            ints = c_inner.acquire(2)
            _lb(ints[0], ints[0], ints[1], border=0, skip_row=bots[0])
            in_c_view.release(1)
            # middle
            for _ in range_(in_h - 3):
                bots = in_c_view.acquire(3)
                _la(bots[0], bots[1], bots[2], border=1)
                ints = c_inner.acquire(3)
                _lb(ints[0], ints[1], ints[2], border=1, skip_row=bots[0])
                in_c_view.release(1)
                c_inner.release(1)
            # postamble step 1
            bots = in_c_view.acquire(2)
            _la(bots[0], bots[1], bots[1], border=2)
            ints = c_inner.acquire(3)
            _lb(ints[0], ints[1], ints[2], border=1, skip_row=bots[0])
            in_c_view.release(1)
            c_inner.release(1)
            # postamble step 2
            # acquire(1) returns the row directly — see c3k2_small's m_0_inner
            # for the same gotcha.
            bot_last = in_c_view.acquire(1)
            ints = c_inner.acquire(2)
            _lb(ints[0], ints[1], ints[1], border=2, skip_row=bot_last)
            in_c_view.release(1)
            c_inner.release(2)

        worker_kwargs = dict(
            fn_args=[
                prev_fifo.cons(),
                ws_a,
                bias_a,
                lut_a,
                ws_b,
                bias_b,
                lut_b,
                out_fifo_local.prod(),
                f_inner.prod(),
                f_inner.cons(),
                ka,
                kb,
            ],
            tile=tile,
        )
        if _stream_outer:
            # Match the m10 conv_pool pattern — keep the streaming inner
            # loop intact so each chunk acquire/release stays in the
            # generated core code.
            worker_kwargs["dynamic_objfifo_lowering"] = True
        return Worker(pair_fn, **worker_kwargs)

    # cv3 (2-input concat) + cv2 (3-input concat) chained on the cv3_cv2 tile.
    # cv3 fits on tile (small, ~16 KB max); cv2 is m8's biggest weight (~98 KB).
    wts_m0c3, sz_m0c3 = _wts_for(m_m0c3)
    bias_m0c3 = _bias_for(m_m0c3, c)
    rs_m0c3 = m_m0c3["right_shift"]
    rs_cv2 = m_cv2["right_shift"]

    sz_cv2 = out_c * (3 * c)  # 3c input concat width
    data_cv2 = _load_bin(m_cv2["weights_file"], np.int8, sz_cv2)
    bias_cv2 = _bias_for(m_cv2, out_c)
    if _stream_outer:
        chunk_sz_cv2 = sz_cv2 // n_cv2_chunks
        # cv2 lives on its own tile in the streamed (m8) path — stream
        # weights directly to it from a same-column memtile.
        cv2_compute_tile = placement.PLACEMENT[block_name]["cv2"]
        wts_cv2 = StaticWeightStream(
            obj_type=_i8((sz_cv2,)),
            initial_value=data_cv2,
            name=f"{block_name}_cv2_wts",
            recv_type=_i8((chunk_sz_cv2,)),
            repeat_count=in_h,
            memtile_placement=Tile(cv2_compute_tile.col, 1),
            compute_placement=cv2_compute_tile,
            mem_lock_id=2,  # cv3 lives on the same tile; offset locks
            comp_lock_id=2,
        )
    else:
        wts_cv2 = Buffer(_i8((sz_cv2,)), initial_value=data_cv2)

    k_cv3 = Kernel(
        f"yolo_c3k2_heavy_cv3_concat2_silu_bias_i8_i8_{block_name}",
        f"yolo_c3k2_heavy_cv3_concat2_{block_name}.o",
        [
            _i8((in_w, 1, cp)),  # inner_1_out slice
            _i8((in_w, 1, cp)),  # split_b slice
            _i8((sz_m0c3,)),
            _i32((c,)),
            _i8((256,)),  # silu_lut
            _i8((in_w, 1, c)),  # cv3 output row
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    if _stream_outer:
        k_cv2 = Kernel(
            f"yolo_c3k2_small_cv2_concat3_streamed_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_small_cv2_concat3_streamed_{block_name}.o",
            [
                _i8((in_w, 1, c)),  # y0 slice
                _i8((in_w, 1, c)),  # y1 slice
                _i8((in_w, 1, c)),  # cv3 output
                _i8((chunk_sz_cv2,)),
                _i32((out_c,)),
                _i8((256,)),  # silu_lut
                _i8((in_w, 1, out_c)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
            # in_w, c, out_c, n_chunks, chunk_idx, right_shift
        )
    else:
        k_cv2 = Kernel(
            f"yolo_c3k2_small_cv2_concat3_silu_bias_i8_i8_{block_name}",
            f"yolo_c3k2_small_cv2_concat3_{block_name}.o",
            [
                _i8((in_w, 1, c)),
                _i8((in_w, 1, c)),
                _i8((in_w, 1, c)),
                _i8((sz_cv2,)),
                _i32((out_c,)),
                _i8((256,)),  # silu_lut
                _i8((in_w, 1, out_c)),
                np.int32,
                np.int32,
                np.int32,
                np.int32,
            ],
        )

    # ----- Workers -----
    if _stream_outer:

        def cv1_fn(act_in_f, wts_pb, bias, lut, top_p, bot_a_p, bot_b_p, k):
            for _ in range_(in_h):
                row_in = act_in_f.acquire(1)
                r_top = top_p.acquire(1)
                r_bot_a = bot_a_p.acquire(1)
                r_bot_b = bot_b_p.acquire(1)
                for wi in range_(n_cv1_chunks):
                    chunk = wts_pb.acquire(1)
                    k(
                        row_in,
                        chunk,
                        bias,
                        lut,
                        r_top,
                        r_bot_a,
                        r_bot_b,
                        in_w,
                        in_c,
                        twoc,
                        n_cv1_chunks,
                        wi,
                        m_cv1["right_shift"],
                    )
                    wts_pb.release(1)
                act_in_f.release(1)
                top_p.release(1)
                bot_a_p.release(1)
                bot_b_p.release(1)

    else:

        def cv1_fn(act_in_f, wts, bias, lut, top_p, bot_p, k):
            for _ in range_(in_h):
                row_in = act_in_f.acquire(1)
                r_top = top_p.acquire(1)
                r_bot = bot_p.acquire(1)
                k(
                    row_in,
                    wts,
                    bias,
                    lut,
                    r_top,
                    r_bot,
                    in_w,
                    in_c,
                    twoc,
                    m_cv1["right_shift"],
                )
                act_in_f.release(1)
                top_p.release(1)
                bot_p.release(1)

    # Streamed path adds a third bot output fifo (no fanout).
    if _stream_outer:
        cv1_fn_args = [
            act_in.cons(),
            wts_cv1,
            bias_cv1,
            silu_cv1,
            top_fifo.prod(),
            bot_fifo.prod(),
            bot_to_cv2_fifo.prod(),
            k_cv1_split,
        ]
    else:
        cv1_fn_args = [
            act_in.cons(),
            wts_cv1,
            bias_cv1,
            silu_cv1,
            top_fifo.prod(),
            bot_fifo.prod(),
            k_cv1_split,
        ]
    w_cv1 = Worker(
        cv1_fn,
        fn_args=cv1_fn_args,
        tile=placement.PLACEMENT[block_name]["cv1"],
    )

    def m_0_split_fn(bot_c, ws_a, bs_a, la, ws_b, bs_b, lb, a_p, b_p, k):
        # Parallel branches: one input row -> two output rows (one each into split_a, split_b).
        # Each branch has its OWN right_shift (m.0/cv1 and m.0/cv2 may have
        # different output scales).
        for _ in range_(in_h):
            row_in = bot_c.acquire(1)
            ra = a_p.acquire(1)
            rb = b_p.acquire(1)
            k(
                row_in,
                ws_a,
                bs_a,
                la,
                ws_b,
                bs_b,
                lb,
                ra,
                rb,
                in_w,
                c,
                cp,
                cp,
                rs_m0c1,
                rs_m0c2,
            )
            bot_c.release(1)
            a_p.release(1)
            b_p.release(1)

    w_m_0_split = Worker(
        m_0_split_fn,
        fn_args=[
            bot_fifo.cons(depth=6),  # one of bot_fifo's two consumers
            wts_m0c1,
            bias_m0c1,
            silu_m0_cv1,
            wts_m0c2,
            bias_m0c2,
            silu_m0_cv2,
            split_a.prod(),
            split_b.prod(),
            k_m0_split,
        ],
        tile=placement.PLACEMENT[block_name]["m_0_split"],
    )

    w_inner_pair_0 = _build_inner_pair(
        "inner_pair_0",
        split_a,
        inner_0_out,
        m_p0c1,
        m_p0c2,
        "inner_pair_0",
        "m.0/m/m.0",
    )
    w_inner_pair_1 = _build_inner_pair(
        "inner_pair_1",
        inner_0_out,
        inner_1_out,
        m_p1c1,
        m_p1c2,
        "inner_pair_1",
        "m.0/m/m.1",
    )

    # cv3 and cv2 wiring differs by topology:
    #  - non-streamed (m6): fused on a single cv3_cv2 tile, internal fifo
    #    f_cv3_cv2 same-tile producer/consumer.
    #  - streamed (m8): split onto separate cv3 + cv2 tiles. cv2 needs an
    #    extra DMA channel for its streamed weights, which exceeds the
    #    2-S2MM budget of a fused tile when top + bot are already DMA in.
    if _stream_outer:

        def cv3_fn(inner1_c, split_b_c, ws3, bs3, l3, cv3_out_p, kc3):
            for _ in range_(in_h):
                ri1 = inner1_c.acquire(1)
                rsb = split_b_c.acquire(1)
                rmid = cv3_out_p.acquire(1)
                kc3(ri1, rsb, ws3, bs3, l3, rmid, in_w, 2 * cp, c, rs_m0c3)
                inner1_c.release(1)
                split_b_c.release(1)
                cv3_out_p.release(1)

        def cv2_fn(top_c, bot_c, cv3_out_c, ws2_pb, bs2, l2, out_p, kc2):
            for _ in range_(in_h):
                ry0 = top_c.acquire(1)
                ry1 = bot_c.acquire(1)
                rm = cv3_out_c.acquire(1)
                rout = out_p.acquire(1)
                for wi in range_(n_cv2_chunks):
                    chunk = ws2_pb.acquire(1)
                    kc2(
                        ry0,
                        ry1,
                        rm,
                        chunk,
                        bs2,
                        l2,
                        rout,
                        in_w,
                        c,
                        out_c,
                        n_cv2_chunks,
                        wi,
                        rs_cv2,
                    )
                    ws2_pb.release(1)
                cv3_out_c.release(1)
                top_c.release(1)
                bot_c.release(1)
                out_p.release(1)

        cv3_to_cv2 = ObjectFifo(_i8((in_w, 1, c)), depth=2, via_DMA=True)

        w_cv3 = Worker(
            cv3_fn,
            fn_args=[
                inner_1_out.cons(),
                split_b.cons(),
                wts_m0c3,
                bias_m0c3,
                silu_m0_cv3,
                cv3_to_cv2.prod(),
                k_cv3,
            ],
            tile=placement.PLACEMENT[block_name]["cv3"],
        )
        w_cv2 = Worker(
            cv2_fn,
            # In the streamed path, bot for cv2 comes from a separate
            # cv1-output fifo (bot_to_cv2_fifo) — eliminates the fanout
            # that forced both consumers onto DMA channels.
            fn_args=[
                top_fifo.cons(),
                bot_to_cv2_fifo.cons(),
                cv3_to_cv2.cons(),
                wts_cv2,
                bias_cv2,
                silu_cv2,
                out_fifo.prod(),
                k_cv2,
            ],
            tile=placement.PLACEMENT[block_name]["cv2"],
        )
        return out_fifo, [
            w_cv1,
            w_m_0_split,
            w_inner_pair_0,
            w_inner_pair_1,
            w_cv3,
            w_cv2,
        ]

    def cv3_cv2_fn(
        inner1_c,
        split_b_c,
        top_c,
        bot_c,
        ws3,
        bs3,
        l3,
        ws2,
        bs2,
        l2,
        out_p,
        p_int,
        c_int,
        kc3,
        kc2,
    ):
        for _ in range_(in_h):
            ri1 = inner1_c.acquire(1)
            rsb = split_b_c.acquire(1)
            r_mid = p_int.acquire(1)
            kc3(ri1, rsb, ws3, bs3, l3, r_mid, in_w, 2 * cp, c, rs_m0c3)
            p_int.release(1)
            inner1_c.release(1)
            split_b_c.release(1)

            ry0 = top_c.acquire(1)
            ry1 = bot_c.acquire(1)
            rm = c_int.acquire(1)
            rout = out_p.acquire(1)
            kc2(ry0, ry1, rm, ws2, bs2, l2, rout, in_w, 3 * c, out_c, rs_cv2)
            c_int.release(1)
            top_c.release(1)
            bot_c.release(1)
            out_p.release(1)

    f_cv3_cv2 = ObjectFifo(_i8((in_w, 1, c)), depth=2)

    w_cv3_cv2 = Worker(
        cv3_cv2_fn,
        fn_args=[
            inner_1_out.cons(),
            split_b.cons(),
            top_fifo.cons(),
            bot_fifo.cons(depth=6),  # the OTHER bot_fifo consumer
            wts_m0c3,
            bias_m0c3,
            silu_m0_cv3,
            wts_cv2,
            bias_cv2,
            silu_cv2,
            out_fifo.prod(),
            f_cv3_cv2.prod(),
            f_cv3_cv2.cons(),
            k_cv3,
            k_cv2,
        ],
        tile=placement.PLACEMENT[block_name]["cv3_cv2"],
    )

    return out_fifo, [w_cv1, w_m_0_split, w_inner_pair_0, w_inner_pair_1, w_cv3_cv2]


# ---------------------------------------------------------------------------
# m9 (PSA) — the only block with act×act MatMul + Softmax.
#
# Topology (ultralytics PSA, ours is single-PSABlock):
#     a, b = chunk(cv1(x), 2, dim=ch)
#     b = b + attn(b)                # PSABlock residual #1
#     b = b + ffn(b)                 # PSABlock residual #2
#     out = cv2(cat([a, b], dim=ch)) # cv2 input is 2c (NOT 3c — unlike C3k2)
#
# Channel widths (c = c1 // 2 = 128 for our c1=256):
#     cv1     : 256 -> 256 (=2c)
#     a, b    : each 128 (=c)
#     attn    : 128 -> 128 (qkv: 128 -> 256, then attn math, then proj 128 -> 128)
#     ffn     : 128 -> 256 -> 128
#     cv2     : 256 (=2c) -> 256
#
# Attention internals (yolo Attention, num_heads=2, key_dim=32, head_dim=64).
# Shapes are PER-SAMPLE — the design runs once per sample; host loops for batch.
#     N      = H*W = 256
#     qkv    : 1x1 conv c -> 2*key_dim + head_dim per head = 256/head * 2 heads = 256
#     reshape: qkv -> (nh=2, kd+kd+hd=128, N=256)
#     q, k, v = split into (2,32,N), (2,32,N), (2,64,N)
#     scores  = q.T @ k = (2,N,32) @ (2,32,N) = (2,N,N)            [MatMul #1]
#     attn    = softmax(scores, dim=-1)                              [INT8 LUT]
#     x_attn  = v @ attn.T = (2,64,N) @ (2,N,N) = (2,64,N)         [MatMul #2]
#     x_attn  = x_attn.reshape(128, H, W)
#     pe_add  = pe(v.reshape(128, H, W))   # pe is dw3x3 stride-1, c -> c
#     x       = x_attn + pe_add
#     out     = proj(x)   # 1x1 c -> c
#
# 5-tile placement.PLACEMENT["m9"]:
#     cv1        : cv1 + chunk
#     qkv        : attn/qkv (1x1 c -> 2c-packed-for-heads)
#     attn_core  : Q@Kᵀ + softmax + S@V + pe + add (ONE kernel; sample-barrier)
#     proj_cv2   : Phase A: attn/proj + skip-add(b) -> attn_block_out -> to ffn
#                  Phase B: cv2(cat(a, ffn_block_out)) -> block out
#     ffn        : ffn.0 + ffn.1-with-skip-add (skip = attn_block_out)
#
# Batch handling: the design runs per-sample (mobilenet convention). The
# host invokes the design B=4 times back-to-back for throughput. yolo_spec
# MatMul records keep ONNX shape (B=4, nh, S, d) because that's what Quark
# emitted, but per-sample-per-invocation inside attn_core sees (nh, S, d).
# See README § "Known limitations" for the PSA batch-loop follow-up.
#
# Notes on fifo depth + kernel splits (still being tuned):
#   - `b` fifo: 2 consumers (qkv tile + proj_cv2 skip) with very different
#     consumption rates (qkv streams row-by-row; proj_cv2 skip needs b after
#     the entire attn_core barrier). May benefit from memtile staging.
#   - `a` fifo: 1 consumer (cv2 phase) but waits across the entire attn+ffn
#     round-trip. Same staging concern.
#   - The attention path is split into per-row kernels (qkv_pack, qk_row,
#     softmax_row, v_pack, sv_row, sv_row_acc, pe_add_row, proj_skip_row)
#     rather than one monolithic attn_core kernel — keeps each .cc small
#     and lets us place them across multiple tiles.
#   - ffn 1x1 weights: 32K + 32K = 64K on the ffn tile. Right at the L1
#     budget; may need StaticWeightStream chunking (cf. m10's conv).
#
# Kernel objects used (kernels/yolo_m9_*.cc):
#   cv1_split           1x1 + chunk(2)
#   qkv / qkv_pack      1x1 c -> 2c, output reshape for per-head packing
#   qk_row / qk_pack    Q @ K^T per-row
#   attn_scale          INT8 requant + scale fold
#   softmax_row         per-row INT8 softmax
#   v_pack / sv_row     S @ V per-row
#   pe_add_row          position encoding add
#   proj_skip_row       1x1 + skip-add(b)
#   ffn_0_silu_row      1x1 c -> 2c, +SiLU
#   ffn_1_skip_row      1x1 2c -> c, +skip-add(attn_block_out)
#   cv2_concat2_streamed 1x1 + 2-input channel concat(a, ffn_block_out)
# ---------------------------------------------------------------------------
def _build_psa(block_name, act_in, manifest):
    assert block_name == "m9"
    blk = yolo_spec.block("m9")
    # Layers in declaration order — pull them out by name for clarity since
    # PSA has more layers than the dispatch indices feel comfortable for.
    by_name = {l.name: l for l in blk.layers}
    L_cv1 = by_name["cv1"]
    L_qkv = by_name["attn/qkv"]
    L_qk = by_name["attn/qk"]  # MatMul
    L_softmax = by_name["attn/softmax"]
    L_pe = by_name["attn/pe"]
    L_sv = by_name["attn/sv"]  # MatMul
    L_proj = by_name["attn/proj"]
    L_ffn0 = by_name["ffn/ffn.0"]
    L_ffn1 = by_name["ffn/ffn.1"]
    L_cv2 = by_name["cv2"]

    in_w, in_h, in_c = L_cv1.in_shape  # (16, 16, 256)
    twoc = L_cv1.out_shape[2]  # 256
    c = twoc // 2  # 128
    out_c = L_cv2.out_shape[2]  # 256
    N = in_w * in_h  # 256 tokens
    n_heads = 2
    key_dim = 32
    head_dim = 64
    # Note on batching: yolo_spec MatMul records carry (B=4, n_heads, S, d)
    # because that's the manifest/ONNX shape from Quark's batch=4 calibration.
    # The IRON design runs per-sample (one slice of the batch=4 tensor per
    # design invocation, matching mobilenet's per-frame convention). For
    # throughput=4 the host invokes the design 4 times back-to-back.
    # Per-sample matmul shapes inside attn_core: (n_heads, S, d).

    # Helpers
    def _wts_for(meta):
        sz = int(np.prod(meta["weights_shape"]))
        return (
            Buffer(
                _i8((sz,)), initial_value=_load_bin(meta["weights_file"], np.int8, sz)
            ),
            sz,
        )

    def _bias_for(meta, oc):
        return Buffer(
            _i32((oc,)), initial_value=_load_bin(meta["bias_file"], np.int32, oc)
        )

    m_cv1 = _op_meta(manifest, L_cv1.manifest_name)
    m_qkv = _op_meta(manifest, L_qkv.manifest_name)
    m_pe = _op_meta(manifest, L_pe.manifest_name)
    m_proj = _op_meta(manifest, L_proj.manifest_name)
    m_ffn0 = _op_meta(manifest, L_ffn0.manifest_name)
    m_ffn1 = _op_meta(manifest, L_ffn1.manifest_name)
    m_cv2 = _op_meta(manifest, L_cv2.manifest_name)
    # Note: softmax + matmuls aren't in the static-weights manifest (they have
    # no learnable parameters). Their right_shifts come from a separate pass
    # over the ONNX graph; here we use the q@k matmul's right_shift as a
    # placeholder for the broader attention requant chain.

    # ----- ObjectFifos -----
    # cv1 output split: a forwards across the whole PSA, b feeds qkv + skip.
    # PSA's pipeline is the deepest in the network (cv1 → qkv → attn_core
    # barrier → proj phase A → ffn → proj phase B). Generous fifo depths
    # absorb the inter-phase lag (HW-correct values are ~16 for top/bot,
    # ~4 for inner).
    SIM_PIPE_DEPTH = 64
    top_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=SIM_PIPE_DEPTH)
    bot_fifo = ObjectFifo(_i8((in_w, 1, c)), depth=SIM_PIPE_DEPTH)
    # qkv frame (per-sample). Element shape is (n_heads, kd+kd+hd, N) — attn_core
    # needs the full N=H*W tokens of one sample before computing the matmuls.
    qkv_fifo = ObjectFifo(_i8((n_heads, 2 * key_dim + head_dim, N)), depth=4)
    # attn_core emits (B, c=128, H, W) reshaped — pre-projection.
    attn_pre_proj = ObjectFifo(_i8((in_w, 1, c)), depth=4)
    # proj output + residual; goes to ffn input
    attn_block_out = ObjectFifo(_i8((in_w, 1, c)), depth=SIM_PIPE_DEPTH)
    # ffn output + residual; comes back to proj_cv2
    ffn_block_out = ObjectFifo(_i8((in_w, 1, c)), depth=SIM_PIPE_DEPTH)
    out_fifo = ObjectFifo(_i8((in_w, 1, out_c)), depth=4)

    # ----- Weights / Biases -----
    wts_cv1, sz_cv1 = _wts_for(m_cv1)
    bias_cv1 = _bias_for(m_cv1, twoc)
    wts_qkv, sz_qkv = _wts_for(m_qkv)
    bias_qkv = _bias_for(m_qkv, 2 * c)  # 256
    wts_pe, sz_pe = _wts_for(m_pe)
    bias_pe = _bias_for(m_pe, c)
    wts_proj, sz_proj = _wts_for(m_proj)
    bias_proj = _bias_for(m_proj, c)
    wts_ffn0, sz_ffn0 = _wts_for(m_ffn0)
    bias_ffn0 = _bias_for(m_ffn0, 2 * c)
    wts_ffn1, sz_ffn1 = _wts_for(m_ffn1)
    bias_ffn1 = _bias_for(m_ffn1, c)
    wts_cv2, sz_cv2 = _wts_for(m_cv2)
    bias_cv2 = _bias_for(m_cv2, out_c)

    # ----- Kernels -----
    k_cv1_split = Kernel(
        "yolo_m9_cv1_split_silu_bias_i8_i8",
        "yolo_m9_cv1_split.o",
        [
            _i8((in_w, 1, in_c)),
            _i8((sz_cv1,)),
            _i32((twoc,)),
            _i8((in_w, 1, c)),
            _i8((in_w, 1, c)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    # qkv emits a per-sample packed frame; the kernel runs on every input row
    # of a sample, accumulating its 1x1 output into the packed (B, 2, 128, 256)
    # frame buffer until the sample is complete. Sample-end signaling TBD —
    # one approach: kernel takes (row_idx, num_rows) args and asserts on the
    # last row. Cleaner alternative: a fused qkv-pack kernel that fires once
    # per sample, but row-streaming matches the upstream cv1's emit pattern.
    k_qkv = Kernel(
        "yolo_m9_qkv_i8_i8",
        "yolo_m9_qkv.o",
        [
            _i8((in_w, 1, c)),  # input row (b half)
            _i8((sz_qkv,)),  # qkv weights
            _i32((2 * c,)),  # qkv bias
            _i8(
                (n_heads, 2 * key_dim + head_dim, N)
            ),  # packed output frame (read-modify-write across rows of one sample)
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],  # in_w, in_c, out_c, row_idx, n_rows, right_shift
    )
    # attn_core: the monolithic attention kernel. Sample-barrier.
    k_attn_core = Kernel(
        "yolo_m9_attn_core_i8_i8",
        "yolo_m9_attn_core.o",
        [
            _i8(
                (n_heads, 2 * key_dim + head_dim, N)
            ),  # qkv packed frame in (one sample)
            _i8((sz_pe,)),  # pe (dw3x3) weights
            _i32((c,)),  # pe bias
            _i8((c, in_h, in_w)),  # attention out (reshaped, pre-projection)
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        # n_heads, key_dim, head_dim, N, H, W, c, right_shift(s) packed
    )
    # attn/proj + residual against b
    k_attn_proj_skip = Kernel(
        "yolo_m9_attn_proj_skip_i8_i8",
        "yolo_m9_attn_proj_skip.o",
        [
            _i8((c, in_h, in_w)),  # attn_pre_proj FULL frame (kernel slices row yi)
            _i8((sz_proj,)),  # proj weights
            _i32((c,)),  # proj bias
            _i8((in_w, 1, c)),  # b skip row
            _i8((in_w, 1, c)),  # output row
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
        # in_w, c, c, right_shift, skip_scale, yi
    )
    # ffn.0 with SiLU
    k_ffn0 = Kernel(
        "yolo_m9_ffn_0_silu_bias_i8_i8",
        "yolo_m9_ffn_0_silu.o",
        [
            _i8((in_w, 1, c)),
            _i8((sz_ffn0,)),
            _i32((2 * c,)),
            _i8((in_w, 1, 2 * c)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    # ffn.1 with skip-add against attn_block_out
    k_ffn1_skip = Kernel(
        "yolo_m9_ffn_1_skip_bias_i8_i8",
        "yolo_m9_ffn_1_skip.o",
        [
            _i8((in_w, 1, 2 * c)),
            _i8((sz_ffn1,)),
            _i32((c,)),
            _i8((in_w, 1, c)),  # skip = attn_block_out row
            _i8((in_w, 1, c)),  # output row
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )
    # cv2 with 2-input channel concat
    k_cv2 = Kernel(
        "yolo_m9_cv2_concat2_silu_bias_i8_i8",
        "yolo_m9_cv2_concat2.o",
        [
            _i8((in_w, 1, c)),  # a slice
            _i8((in_w, 1, c)),  # ffn_block_out slice
            _i8((sz_cv2,)),
            _i32((out_c,)),
            _i8((in_w, 1, out_c)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    # ----- Workers -----
    rs_cv1 = m_cv1["right_shift"]
    rs_qkv = m_qkv["right_shift"]
    rs_proj = m_proj["right_shift"]
    rs_ffn0 = m_ffn0["right_shift"]
    rs_ffn1 = m_ffn1["right_shift"]
    rs_cv2 = m_cv2["right_shift"]

    def cv1_fn(act_in_f, wts, bias, top_p, bot_p, k):
        # Row-by-row 1x1 + chunk (same shape as c3k2_*).
        # Outer batch loop is implicit — the host runtime fills act_in for
        # B*in_h rows, and this worker processes them all sequentially.
        for _ in range_(in_h):
            row_in = act_in_f.acquire(1)
            r_top = top_p.acquire(1)
            r_bot = bot_p.acquire(1)
            k(row_in, wts, bias, r_top, r_bot, in_w, in_c, twoc, rs_cv1)
            act_in_f.release(1)
            top_p.release(1)
            bot_p.release(1)

    w_cv1 = Worker(
        cv1_fn,
        fn_args=[
            act_in.cons(),
            wts_cv1,
            bias_cv1,
            top_fifo.prod(),
            bot_fifo.prod(),
            k_cv1_split,
        ],
        tile=placement.PLACEMENT["m9"]["cv1"],
    )

    def qkv_fn(bot_c, wts, bias, qkv_p, k):
        # Per design invocation = one sample. Acquire 1 packed-output frame,
        # run 1x1 kernel `in_h` times accumulating row-by-row into the packed
        # (nh, kd+kd+hd, N) layout, then release the qkv frame to attn_core.
        qkv_frame = qkv_p.acquire(1)
        for yi in range_(in_h):
            row_in = bot_c.acquire(1)
            k(row_in, wts, bias, qkv_frame, in_w, c, 2 * c, yi, in_h, rs_qkv)
            bot_c.release(1)
        qkv_p.release(1)

    w_qkv = Worker(
        qkv_fn,
        fn_args=[
            bot_fifo.cons(depth=8),  # one of bot_fifo's two consumers
            wts_qkv,
            bias_qkv,
            qkv_fifo.prod(),
            k_qkv,
        ],
        tile=placement.PLACEMENT["m9"]["qkv"],
    )

    def attn_core_fn(qkv_c, wts_pe_b, bias_pe_b, attn_p, k):
        # Per design invocation = one sample. Sample-barrier: wait for full
        # qkv frame, run the monolithic attention kernel, emit the
        # pre-projection attention output as a contiguous (c, H, W) frame.
        # Downstream proj_cv2 reads it row-by-row.
        qkv_frame = qkv_c.acquire(1)
        attn_frame = attn_p.acquire(1)
        k(
            qkv_frame,
            wts_pe_b,
            bias_pe_b,
            attn_frame,
            n_heads,
            key_dim,
            head_dim,
            N,
            in_h,
            in_w,
            c,
            m_pe["right_shift"],
        )
        qkv_c.release(1)
        attn_p.release(1)

    # attn_pre_proj's element type is one sample's (c, H, W) attn frame.
    attn_frame_fifo = ObjectFifo(_i8((c, in_h, in_w)), depth=2)

    w_attn_core = Worker(
        attn_core_fn,
        fn_args=[
            qkv_fifo.cons(),
            wts_pe,
            bias_pe,
            attn_frame_fifo.prod(),
            k_attn_core,
        ],
        tile=placement.PLACEMENT["m9"]["attn_core"],
    )

    def proj_cv2_fn(
        attn_frame_c,
        b_skip_c,
        top_c,
        ffn_back_c,
        ws_proj,
        bs_proj,
        ws_cv2,
        bs_cv2,
        pab_p,
        out_p,
        k_proj,
        kc2,
    ):
        # Two phases per sample, sequential (not interleaved across this worker):
        # Phase A — produce attn_block_out from (attn_frame slice, b skip):
        #   for yi in in_h:
        #     proj kernel(attn_frame[row], b_skip[row]) -> attn_block_out[row]
        #     (forward to ffn)
        # Phase B — wait for ffn_block_out, run cv2:
        #   for yi in in_h:
        #     cv2 kernel(a[row], ffn_block_out[row]) -> out[row]
        #
        # Slicing of attn_frame is done via memref_view inside the kernel call
        # (sketch-level: just pass the whole frame + row index).
        # Phase A
        attn_frame = attn_frame_c.acquire(1)
        for yi in range_(in_h):
            r_b = b_skip_c.acquire(1)
            r_pab = pab_p.acquire(1)
            k_proj(
                attn_frame,
                ws_proj,
                bs_proj,
                r_b,
                r_pab,
                in_w,
                c,
                c,
                rs_proj,
                rs_proj,
                yi,
            )
            b_skip_c.release(1)
            pab_p.release(1)
        attn_frame_c.release(1)

        # Phase B
        for yi in range_(in_h):
            r_top = top_c.acquire(1)
            r_ffn = ffn_back_c.acquire(1)
            r_out = out_p.acquire(1)
            kc2(r_top, r_ffn, ws_cv2, bs_cv2, r_out, in_w, c, out_c, rs_cv2)
            top_c.release(1)
            ffn_back_c.release(1)
            out_p.release(1)

    w_proj_cv2 = Worker(
        proj_cv2_fn,
        fn_args=[
            attn_frame_fifo.cons(),
            bot_fifo.cons(depth=8),  # the OTHER bot_fifo consumer (b skip)
            top_fifo.cons(),  # a
            ffn_block_out.cons(),
            wts_proj,
            bias_proj,
            wts_cv2,
            bias_cv2,
            attn_block_out.prod(),
            out_fifo.prod(),
            k_attn_proj_skip,
            k_cv2,
        ],
        tile=placement.PLACEMENT["m9"]["proj_cv2"],
    )

    def ffn_fn(in_c, ws0, bs0, ws1, bs1, out_p, p12, c12, k0, k1):
        # Two chained 1x1s + skip-add. No sliding window (both 1x1).
        for _ in range_(in_h):
            row_in = in_c.acquire(1)
            row_mid = p12.acquire(1)
            k0(row_in, ws0, bs0, row_mid, in_w, c, 2 * c, rs_ffn0)
            p12.release(1)

            row_mid_r = c12.acquire(1)
            row_out = out_p.acquire(1)
            k1(row_mid_r, ws1, bs1, row_in, row_out, in_w, 2 * c, c, rs_ffn1, rs_ffn1)
            c12.release(1)
            in_c.release(1)
            out_p.release(1)

    f_ffn = ObjectFifo(_i8((in_w, 1, 2 * c)), depth=2)

    w_ffn = Worker(
        ffn_fn,
        fn_args=[
            attn_block_out.cons(),
            wts_ffn0,
            bias_ffn0,
            wts_ffn1,
            bias_ffn1,
            ffn_block_out.prod(),
            f_ffn.prod(),
            f_ffn.cons(),
            k_ffn0,
            k_ffn1_skip,
        ],
        tile=placement.PLACEMENT["m9"]["ffn"],
    )

    return out_fifo, [w_cv1, w_qkv, w_attn_core, w_proj_cv2, w_ffn]


def _build_m8_chain(act_in, manifest):
    """Chain builder for m8 — 2-tile megakernel (scripts/m8_megakernel_2tile.py).

    Fuses cv1+m_0_split on tile A (5,3) and pair1+cv3+cv2 on tile B (5,4),
    delivering the chain's best-known m8 throughput (~404 ms/sample batched
    N=15 vs ~560 ms/sample with the older 8-tile design).
    """
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "m8_megakernel_2tile",
        pathlib.Path(__file__).parent / "scripts" / "m8_megakernel_2tile.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(act_in_external=act_in, return_program=False)


def _build_m9_chain(act_in, manifest):
    """Chain builder for m9 (PSA) — delegates to scripts/m9_stage.py.

    Stage defaults to 10 (full PSA block: cv1 → qkv → attn_core → proj →
    ffn → cv2). Override via M9_CHAIN_STAGE env var to bisect.
    """
    import os
    import importlib.util
    import pathlib

    stage = int(os.environ.get("M9_CHAIN_STAGE", "10"))
    spec = importlib.util.spec_from_file_location(
        "m9_stage",
        pathlib.Path(__file__).parent / "scripts" / "m9_stage.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(stage=stage, act_in_external=act_in, return_program=False)


_BUILDERS = {
    "m0": _build_m0,
    "m1": lambda act_in, m: _build_conv_stride_block("m1", act_in, m),
    # m3/m5/m7: weights exceed 64KB AIE2P tile budget. _build_conv_stride_block_streamed
    # below contains the IRON wiring (StaticWeightStream + chunked-kernel) BUT
    # lowlevel_dma.StaticWeightStream emits a single source BD per repeat —
    # it doesn't actually decompose the source buffer into chunks, so the
    # MemTile sends a full 36864/147456/294912 B buffer while the recv tile
    # expects 9216 B chunks → DMA size mismatch → runtime timeout. Fixing
    # this needs chunked source BDs (aie.dma_bd with offset+length) in
    # lowlevel_dma.py; that's a change to the IRON-side helper (upstream too).
    # Until then m3/m5/m7 stay on the non-streamed path and fail to compile.
    # m3/m5/m7: weights exceed AIE2P tile budget — stream chunked from MemTile
    # via chunked_dma.ChunkedWeightStream (fork of StaticWeightStream that
    # actually emits N source BDs, one per chunk).
    "m3": lambda act_in, m: _build_conv_stride_block_streamed("m3", act_in, m),
    "m5": lambda act_in, m: _build_conv_stride_block_streamed("m5", act_in, m),
    "m7": lambda act_in, m: _build_conv_stride_block_streamed("m7", act_in, m),
    "m2": lambda act_in, m: _build_c3k2_small("m2", act_in, m),
    "m4": lambda act_in, m: _build_c3k2_small("m4", act_in, m),
    "m6": lambda act_in, m: _build_c3k2_heavy("m6", act_in, m),
    # m8 uses the 2-tile fused megakernel (scripts/m8_megakernel_2tile.py).
    "m8": lambda act_in, m: _build_m8_chain(act_in, m),
    # m9 uses the staged split-attn_core design from scripts/m9_stage.py.
    # The legacy _build_psa monolithic sketch above is kept for sim reference
    # only — the HW path goes through the staged builder.
    "m9": lambda act_in, m: _build_m9_chain(act_in, m),
    "m10": lambda act_in, m: _build_head("m10", act_in, m),
}


def per_block_iron(block_name: str) -> str:
    if block_name not in _BUILDERS:
        raise NotImplementedError(
            f"per-block IRON not yet drafted for {block_name!r} (have: {sorted(_BUILDERS)})"
        )
    manifest = _load_manifest()

    # m9 stages can produce outputs LARGER than the final block output
    # (e.g. stage 4 emits 128KB scores vs cv2's 64KB), so we can't size
    # the runtime sequence from yolo_spec's last-layer shape. Delegate
    # the full Runtime build to the staged builder which knows its own
    # per-stage output size.
    if block_name == "m9":
        import importlib.util, pathlib, os

        # M9_STAGE is the canonical env var for selecting which staged build
        # of m9 to produce. Defaults to 10 (full PSA block) so a bare
        # `make BLOCK=m9` matches what `run_ort BLOCK=m9` compares against
        # (cv2 post-SiLU). Set M9_STAGE=1 to build cv1-only for tighter
        # iteration on cv1 changes. M9_CHAIN_STAGE is accepted as a legacy
        # alias.
        stage = int(
            os.environ.get("M9_STAGE", os.environ.get("M9_CHAIN_STAGE", "10"))
        )
        spec = importlib.util.spec_from_file_location(
            "m9_stage",
            pathlib.Path(__file__).parent / "scripts" / "m9_stage.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build(stage=stage, return_program=True)

    # m8: 2-tile megakernel (see scripts/m8_megakernel_2tile.py).
    if block_name == "m8":
        import importlib.util, pathlib

        spec = importlib.util.spec_from_file_location(
            "m8_megakernel_2tile",
            pathlib.Path(__file__).parent / "scripts" / "m8_megakernel_2tile.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build(return_program=True)

    blk = yolo_spec.block(block_name)
    in_w, in_h, in_c_raw = blk.layers[0].in_shape
    last_out_shape = blk.layers[-1].out_shape

    # m0 specifically: input is RGB padded to 8 ch.
    in_c = 8 if block_name == "m0" else in_c_raw

    # Per-block N_SAMPLES knob (for multi-sample standalone bisects of
    # the chain hang). Default 1.
    import os as _os

    _n_samples = int(_os.environ.get("BLOCK_N_SAMPLES", "1"))

    # Output byte count depends on last layer shape. Spatial conv blocks use
    # (W, H, C); m10's final Gemm layer uses (out_c,) — flat vector.
    in_bytes = in_w * in_h * in_c
    out_bytes = int(np.prod(last_out_shape))
    in_i32_per = in_bytes // 4
    out_i32_per = (out_bytes + 3) // 4
    in_ty = _i32((_n_samples * in_i32_per,))
    out_ty = _i32((_n_samples * out_i32_per,))

    # m8's tile A is L1-tight (the 2-tile megakernel hosts cv1 + split + pair0
    # weights, all OFs delegated to neighbor (5,2)). depth=2 fits; depth=5
    # busts. All other blocks use depth=5 to absorb shim<->compute jitter.
    _act_in_depth = 2 if block_name == "m8" else 5
    act_in = ObjectFifo(_i8((in_w, 1, in_c)), depth=_act_in_depth)
    out_fifo, workers = _BUILDERS[block_name](act_in, manifest)

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (inp, out):
        if TRACE_SIZE_PER_WORKER > 0:
            # Per-tile packet trace → DRAM, appended after `out` (ddr_id=-1
            # default, override via TRACE_DDR_ID env). TRACE_EVENTS env (csv
            # of CoreEventAIE2P names, e.g. ACTIVE,DISABLED,INSTR_EVENT_0,...)
            # overrides the default 8 events. See aie2_yolo_iron_partial for
            # the same idiom.
            _ddr_id = int(_os.environ.get("TRACE_DDR_ID", "-1"))
            _events_kwargs = {}
            if TRACE_EVENTS is not None:
                from aie.utils.trace.events import CoreEventAIE2P

                evs = list(TRACE_EVENTS)
                while len(evs) < 8:
                    evs.append(CoreEventAIE2P.NONE)
                _events_kwargs["coretile_events"] = evs[:8]
            rt.enable_trace(
                trace_size=TRACE_SIZE_PER_WORKER * len(workers),
                workers=list(workers),
                ddr_id=_ddr_id,
                **_events_kwargs,
            )
        rt.start(*workers)
        tg = rt.task_group()
        if _n_samples == 1:
            rt.fill(
                act_in.prod(),
                inp,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
            )
            rt.drain(
                out_fifo.cons(),
                out,
                wait=True,
                tile=placement.PLACEMENT["shim"]["output"],
                task_group=tg,
            )
        else:
            # Single configure_task per direction with HW BD-chain replay
            # (dim 3 size → shim BD repeat_count). Avoids the shim per-
            # channel task queue depth limit (~6) that hangs N>=7 with
            # per-sample tasks. See feedback_shim_task_queue_depth.md.
            from aie.helpers.taplib import TensorAccessPattern

            in_total = _n_samples * in_i32_per
            out_total = _n_samples * out_i32_per
            assert (
                _n_samples <= 255
            ), f"BLOCK_N_SAMPLES={_n_samples} exceeds shim repeat_count cap 255"
            in_tap = TensorAccessPattern(
                (in_total,),
                offset=0,
                sizes=[_n_samples, 1, 1, in_i32_per],
                strides=[in_i32_per, 0, 0, 1],
            )
            out_tap = TensorAccessPattern(
                (out_total,),
                offset=0,
                sizes=[_n_samples, 1, 1, out_i32_per],
                strides=[out_i32_per, 0, 0, 1],
            )
            rt.fill(
                act_in.prod(),
                inp,
                tap=in_tap,
                tile=placement.PLACEMENT["shim"]["input"],
                task_group=tg,
            )
            rt.drain(
                out_fifo.cons(),
                out,
                tap=out_tap,
                wait=True,
                tile=placement.PLACEMENT["shim"]["output"],
                task_group=tg,
            )
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Emit MLIR for one yolo26n-cls block.")
    ap.add_argument("block", help="block name from yolo_spec.NETWORK (e.g. m0)")
    args = ap.parse_args()
    print(per_block_iron(args.block))
