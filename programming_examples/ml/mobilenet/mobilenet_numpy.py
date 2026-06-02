#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Pure-numpy reference for MobileNet V3 (algorithm onramp).

Drives the network from network_spec.NETWORK; mirrors the AIE bottleneck-kernel
arithmetic (int32 acc, round-half-even shift-right, saturating clamp,
zero-padding for borders). Reads the same int8 weight files (`*_chain.txt`)
the AIE design loads.

Usage:
    python3 mobilenet_numpy.py
    python3 mobilenet_numpy.py --dump-intermediates /tmp/np_inter

Programmatic:
    from mobilenet_numpy import run
    out, intermediates = run(input_i8, weights_dir, scales, return_intermediates=True)
    # intermediates["bn3"] is the (H,W,C) tensor after bn3.

Kernel sources (translated into numpy below):
    aie_kernels/aie2/bottleneck/bn_conv2dk1_relu.cc  (1x1 + ReLU, 1x1 + avgpool)
    aie_kernels/aie2/bottleneck/bn_conv2dk1_skip.cc  (1x1 + skip add)
    aie_kernels/aie2/bottleneck/bn_conv2dk3_dw.cc    (DW 3x3, ui8 -> ui8)
    aie_kernels/aie2/bottleneck/bn_conv2dk3.cc       (init: 3x3 stride-2)

STATUS — reference outputs to compare against:

    1. data/golden_output.txt        — brevitas-quantized reference (the
                                        "ground truth" target). The full AIE
                                        design matches this within atol=9
                                        (a known-acceptable gap tracked in
                                        mlir-aie issue #3009).
    2. mobilenet_numpy.run() output  — this file. 929/1280 bit-exact vs
                                        golden (max=2, mean=0.30).

Per-bn / per-chain bit-exactness (verified against brevitas fixtures in
bottleneck_A/B/C with their own scale_factors.json):

    bn1, bn2, bn3, bn6, bn7, bn8       → 100% BIT-EXACT (bottleneck_A)
    bn10 → bn11 → bn12 chain           → 100% BIT-EXACT (bottleneck_B)
    bn13 → bn14 cascade chain          → 100% BIT-EXACT (bottleneck_C)

So all conv1x1, conv1x1_skip, dw3x3, conv3x3, and cascade kernels are correct.
The remaining max=2 / mean=0.30 gap on the FULL network most likely comes from
init / bn0 (the unique 2-layer skip block) / post_l1 / post_l2 — blocks for
which no per-bn brevitas fixture exists in this repo.
"""

import os
import json

import numpy as np

from network_spec import NETWORK, block as nsblock


# ---------------------------------------------------------------------------
# Quantization arithmetic
# ---------------------------------------------------------------------------
# All bn_* kernels use **round-half-even** SRS:
#   sum_srs = ((sum + (1 << (scale-1)) - 1 + ((sum >> scale) & 1)) >> scale)
# This rounds .5 to nearest even (banker's rounding) instead of always-up.
# Verified across bn_conv2dk1_relu.cc, bn_conv2dk1_skip.cc, bn_conv2dk3_dw.cc,
# bn_conv2dk3.cc (init).
#
# The conv2dk1_skip kernels also branch on `skip_scale > 0`: if false they fall
# back to round-half-up. We mirror that exactly.
def _srs_even(acc, scale):
    """Round-half-to-even shift right by `scale` bits — the bottleneck SRS."""
    if scale == 0:
        return acc.astype(np.int64)
    a = acc.astype(np.int64)
    return ((a + (1 << (scale - 1)) - 1 + ((a >> scale) & 1)) >> scale).astype(np.int64)


def _srs_half_up(acc, scale):
    """Round-half-up shift right (used by skip add when skip_scale == 0)."""
    if scale == 0:
        return acc.astype(np.int64)
    a = acc.astype(np.int64)
    return ((a + (1 << (scale - 1))) >> scale).astype(np.int64)


def _sat_i8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def _sat_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Weight layout decoders (AIE OIYXI8O8 → standard OIYX)
# ---------------------------------------------------------------------------
def _decode_oiyxi8o8(flat_i8, out_c, in_c, ky=1, kx=1):
    """Decode pointwise / 3x3 weights from flat OIYXI8O8 → (out_c, in_c, ky, kx).

    Layout: flat[oc/8, ic/8, y, x, ic8, oc8] (verified against bn_conv2dk1_relu.cc
    and bn_conv2dk3.cc indexing).
    """
    assert out_c % 8 == 0, f"out_c={out_c} must be a multiple of 8"
    assert in_c % 8 == 0, f"in_c={in_c} must be a multiple of 8"
    expected = out_c * in_c * ky * kx
    assert flat_i8.size == expected, f"weight size {flat_i8.size} != {expected}"
    blocked = flat_i8.reshape(out_c // 8, in_c // 8, ky, kx, 8, 8)
    # Permute (Oo, Io, Y, X, Ii, Oi) → (Oo, Oi, Io, Ii, Y, X).
    standard = blocked.transpose(0, 5, 1, 4, 2, 3).reshape(out_c, in_c, ky, kx)
    return standard.astype(np.int32)


def _decode_dw3x3(flat_i8, channels):
    """Decode AIE depthwise 3x3 weights → (channels, 3, 3).

    Layout (per bn_conv2dk3_dw.cc): flat[c/8, y, x, c8] = wts[c, y, x].
    """
    assert channels % 8 == 0
    expected = 9 * channels
    assert flat_i8.size == expected
    blocked = flat_i8.reshape(channels // 8, 3, 3, 8)
    standard = blocked.transpose(0, 3, 1, 2).reshape(channels, 3, 3)
    return standard.astype(np.int32)


# ---------------------------------------------------------------------------
# Activation I/O
# ---------------------------------------------------------------------------
def _load_input_image(data_dir, in_h, in_w, in_c):
    """Read before_ifm_mem_fmt_1x1.txt as (H, W, C) int8.

    File holds quantized int8 in (C, H, W) order per test_mobilenet.py.
    """
    path = os.path.join(data_dir, "before_ifm_mem_fmt_1x1.txt")
    raw = np.loadtxt(path, delimiter=",", dtype=np.int8)
    assert raw.size == in_h * in_w * in_c, f"input size {raw.size} != {in_h*in_w*in_c}"
    return raw.reshape(in_c, in_h, in_w).transpose(1, 2, 0).copy()


# ---------------------------------------------------------------------------
# Kernel: 1x1 pointwise (translated from bn_conv2dk1_relu.cc/conv2dk1_i8_scalar)
# ---------------------------------------------------------------------------
def conv1x1(x, w_oiyx, scale, out_dtype):
    """1x1 conv. x: (H, W, IC); w: (OC, IC, 1, 1); returns (H, W, OC)."""
    H, W, IC = x.shape
    OC = w_oiyx.shape[0]
    flat = x.astype(np.int32).reshape(-1, IC)
    w2d = w_oiyx.reshape(OC, IC).T  # (IC, OC)
    acc = flat @ w2d  # (H*W, OC) int32
    sum_srs = _srs_even(acc, scale)
    out = _sat_u8(sum_srs) if out_dtype == np.uint8 else _sat_i8(sum_srs)
    return out.reshape(H, W, OC)


# ---------------------------------------------------------------------------
# Kernel: 1x1 + scaled skip add (translated from bn_conv2dk1_skip.cc)
# ---------------------------------------------------------------------------
# Two flavors:
#   _ui8_i8_i8: input uint8 (DW out), weights int8, skip int8, output int8.
#               Used by bn2/4/5/7-9 last layer (regular skip blocks).
#   _ui8_ui8_i8: input uint8, weights int8, skip uint8, output int8.
#               Used by bn0 only (skip = init-conv uint8 output).
# Arithmetic (both):
#   acc = sum(input * w) [i32]
#   sum_srs = round_half_even(acc, scale), clamp to int8
#   skip_sum = sum_srs (int8) + skip (i8 or ui8) [as i32]
#   if skip_scale > 0: final = round_half_even(skip_sum, skip_scale)
#   else:              final = round_half_up(skip_sum, skip_scale)
#   out = clamp(final, int8)
def conv1x1_skip(x_u8, w_oiyx, scale, skip, scale_add):
    H, W, IC = x_u8.shape
    OC = w_oiyx.shape[0]
    flat = x_u8.astype(np.int32).reshape(-1, IC)
    w2d = w_oiyx.reshape(OC, IC).T
    acc = flat @ w2d
    sum_srs = _srs_even(acc, scale)
    sum_srs_i8 = np.clip(sum_srs, -128, 127)  # clamp BEFORE skip add (kernel int8_t)
    skip_flat = skip.astype(np.int32).reshape(-1, OC)
    skip_sum = sum_srs_i8 + skip_flat
    if scale_add > 0:
        final = _srs_even(skip_sum, scale_add)
    else:
        final = _srs_half_up(skip_sum, scale_add)
    return _sat_i8(final).reshape(H, W, OC)


# ---------------------------------------------------------------------------
# Kernel: depthwise 3x3 (translated from bn_conv2dk3_dw.cc/conv2dk3_ui8_scalar)
# ---------------------------------------------------------------------------
# Border behavior: zero-pad in BOTH dimensions (the kernel skips ki=0 for left
# col and ki=2 for right col, equivalent to zero padding).
# Vertical: caller passes the right 3 lines per output row; for stride-2
# output row 0 caller sets line0=line1=row0 with check=top — semantically
# equivalent to zero padding the top.
def dw3x3(x, w_chw, scale, stride, out_dtype):
    """Depthwise 3x3. x: (H, W, C); w: (C, 3, 3); returns (H', W', C).

    out_dtype is np.uint8 (relu fused: clamp [0,255]); kernel always relu-fuses.
    """
    H, W, C = x.shape
    out_h = H // stride
    out_w = W // stride
    # Pad with 1 row top + 1 row bottom (zero); 1 col left + 1 col right (zero).
    padded = np.zeros((H + 2, W + 2, C), dtype=np.int32)
    padded[1:-1, 1:-1, :] = x.astype(np.int32)
    acc = np.zeros((out_h, out_w, C), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            # For output (y, x) with stride s, input position (s*y - 1 + ky + 1,
            # s*x - 1 + kx + 1) in padded coords = (s*y + ky, s*x + kx).
            inp = padded[
                ky : ky + stride * out_h : stride, kx : kx + stride * out_w : stride, :
            ]
            assert inp.shape == (out_h, out_w, C)
            w_kykx = w_chw[:, ky, kx].astype(np.int64)  # (C,)
            acc += inp.astype(np.int64) * w_kykx[None, None, :]
    sum_srs = _srs_even(acc, scale)
    return _sat_u8(sum_srs)  # all DW kernels apply relu


# ---------------------------------------------------------------------------
# Kernel: init 3x3 stride 2 (bn_conv2dk3.cc/conv2dk3_i8_stride2_scalar)
# ---------------------------------------------------------------------------
# Standard 3x3 stride-2 conv with zero-padding. Output is always uint8 (relu).
def conv3x3_stride2_init(x, w_oiyx, scale):
    """init: 3x3 stride-2 conv. x: (H, W, IC) int8; w: (OC, IC, 3, 3); → (H/2, W/2, OC) uint8."""
    H, W, IC = x.shape
    OC = w_oiyx.shape[0]
    out_h = H // 2
    out_w = W // 2
    padded = np.zeros((H + 2, W + 2, IC), dtype=np.int32)
    padded[1:-1, 1:-1, :] = x.astype(np.int32)
    acc = np.zeros((out_h, out_w, OC), dtype=np.int64)
    for ky in range(3):
        for kx in range(3):
            # Stride 2 in both dims; padded coord per (y, x): (2y + ky, 2x + kx)
            inp = padded[ky : ky + 2 * out_h : 2, kx : kx + 2 * out_w : 2, :]
            assert inp.shape == (out_h, out_w, IC)
            w_kykx = w_oiyx[:, :, ky, kx]  # (OC, IC)
            inp_flat = inp.reshape(-1, IC).astype(np.int64)
            w_t = w_kykx.T.astype(np.int64)  # (IC, OC)
            acc += (inp_flat @ w_t).reshape(out_h, out_w, OC)
    sum_srs = _srs_even(acc, scale)
    return _sat_u8(sum_srs)


# ---------------------------------------------------------------------------
# Kernel: post_l1 (1x1 expand fused with global avg-pool, padded output)
# Translated from fused_conv2dk1_xy_pool_i8_large_padded_scalar in
# bn_conv2dk1_relu.cc. The kernel:
#   1. For each output channel: conv1x1 per (H, W) pixel + srs + clamp uint8
#   2. Accumulate post-clamp uint8 values across all 49 pixels
#   3. Divide accumulator by 49 with banker's rounding (round .5 to even)
#   4. Pad output channels (960 → 1280) with zeros
# ---------------------------------------------------------------------------
def post_l1_avgpool_conv1x1(x_i8, w_flat, scale, out_padded_c=1280):
    """post_l1: (7,7,80) int8 -> (1,1,1280) uint16."""
    H, W, IC = x_i8.shape
    OC_unpadded = 960  # the raw 1x1 expansion before zero-padding
    n_pixels = H * W  # = 49
    # Decode weights and run conv1x1 + srs + clamp for each pixel.
    w = _decode_oiyxi8o8(w_flat, OC_unpadded, IC)
    flat = x_i8.astype(np.int32).reshape(-1, IC)
    w2d = w.reshape(OC_unpadded, IC).T
    acc = flat @ w2d  # (n_pixels, OC_unpadded) int32
    sum_srs = _srs_even(acc, scale)  # int64
    sum_srs_clamped = np.clip(sum_srs, 0, 255).astype(
        np.uint32
    )  # per-pixel uint8 stored as uint32
    # Sum across all pixels per channel.
    accumulator = sum_srs_clamped.sum(axis=0)  # (OC_unpadded,) uint32
    # Banker's-round divide by 49.
    avg = _banker_round_div(accumulator, n_pixels)  # (OC_unpadded,) int64
    out = np.zeros(out_padded_c, dtype=np.uint16)
    out[:OC_unpadded] = avg.astype(np.uint16)
    return out.reshape(1, 1, out_padded_c)


def _banker_round_div(numer, denom):
    """Match the kernel's quirky avg rounding (NOT standard banker's):

        float avg = numer / denom;
        if ((int)(avg * 10) % 10 == 5):
            # first decimal digit is 5 → round int_part to nearest even
            int_avg = (int)avg; return int_avg if even else int_avg + 1
        else:
            return (int)(avg + 0.5)   # round half-up

    For denom=49 the "first decimal == 5" branch triggers when avg ∈ [0.5, 0.6) ∪
    [1.5, 1.6) ∪ ... — i.e. NOT actual half-integers (49 is odd so 2r==denom never
    holds). This is a kernel idiosyncrasy we have to replicate for bit-exactness.
    """
    # Integer-only mirror of the C float arithmetic.
    # first_decimal_digit = (int)(10*numer/denom) % 10   ≡ (10*numer // denom) % 10
    # int_avg              = numer // denom
    # round_half_up_result = (int)(avg + 0.5) for avg >= 0
    #                      = (2*numer + denom) // (2*denom)
    n = numer.astype(np.int64)
    d = np.int64(denom)
    int_avg = n // d
    first_dec = (10 * n // d) % 10
    is_5 = first_dec == 5
    even_avg = (int_avg & 1) == 0
    banker_result = np.where(even_avg, int_avg, int_avg + 1)
    half_up_result = (2 * n + d) // (2 * d)
    return np.where(is_5, banker_result, half_up_result).astype(np.int64)


# ---------------------------------------------------------------------------
# Kernel: post_l2 FC (1x1 conv on uint16 inputs)
# Translated from conv2dk1_ui16_scalar_pad.
# Output stored as uint16 but clamped to [0, 255] (UMAX) — i.e. uint8 value
# zero-extended into uint16 memory.
# ---------------------------------------------------------------------------
def post_l2_fc(x_u16, w_flat, scale, in_c_padded):
    """FC layer. x: (1,1,IC) uint16 (with last channels possibly zero-padded);
    w: OIYXI8O8 with in_channels = in_c_padded; returns (1,1,OC) uint16.
    """
    H, W, IC = x_u16.shape
    assert IC == in_c_padded
    OC = w_flat.size // in_c_padded
    w = _decode_oiyxi8o8(w_flat, OC, in_c_padded)
    flat = x_u16.astype(np.int32).reshape(-1, IC)
    w2d = w.reshape(OC, IC).T
    acc = flat @ w2d  # (1, OC) int32
    sum_srs = _srs_even(acc, scale)
    out_clamped = np.clip(sum_srs, 0, 255).astype(np.uint16)  # uint8 in uint16 storage
    return out_clamped.reshape(1, 1, OC)


# ---------------------------------------------------------------------------
# Block runner: handles per-block weight layout + kernel dispatch + skip
# ---------------------------------------------------------------------------
def _load_chain(data_dir, filename):
    return np.loadtxt(os.path.join(data_dir, filename), delimiter=",", dtype=np.int8)


def _run_init(x, sf, data_dir):
    blk = nsblock("init")
    layer = blk.layers[0]
    in_c = layer.in_shape[2]
    out_c = layer.out_shape[2]
    chain = _load_chain(data_dir, "init_chain.txt")
    w = _decode_oiyxi8o8(chain, out_c, in_c, ky=3, kx=3)
    return conv3x3_stride2_init(x, w, sf["INIT"]["conv3x3"])


def _run_regular_block(blk, x, sf, data_dir):
    """Run a bottleneck block from {bn0..bn12} via combined `_chain.txt` weights.

    bn0:        layers = (dw3x3, conv1x1) + skip   (skip type = uint8)
    bn1/3/6:    layers = (conv1x1+relu, dw3x3, conv1x1)        (no skip)
    bn2/4/5/7:  layers = (conv1x1+relu, dw3x3, conv1x1) + skip (skip type = int8)
    bn8/9:      same as bn7 (fused-pair on AIE; logically identical)
    bn10/11/12: same kernel set; bn10/12 no skip, bn11 has skip.

    All non-pair blocks have a single `bnN_chain.txt`. bn4/5 share `bn4_5_chain.txt`,
    bn8/9 share `bn8_9_chain.txt`. bn10..12 split weights across 2-3 files.
    """
    name = blk.name
    sf_blk = sf[name.upper()]
    x_in = x

    # Slice + decode weights per layer.
    ws = _load_block_weights(blk, data_dir)
    for layer, w_decoded in zip(blk.layers, ws):
        scale = sf_blk[layer.sf_key]
        out_dtype = np.uint8 if layer.activation == "relu" else np.int8
        if layer.kind == "conv1x1":
            x = conv1x1(x, w_decoded, scale, out_dtype)
        elif layer.kind == "dw3x3":
            x = dw3x3(x, w_decoded, scale, layer.stride, np.uint8)
        else:
            raise ValueError(f"{name}: unhandled layer kind {layer.kind!r}")
    if blk.skip:
        # Last layer's int8 output + block input; final kernel is conv2dk1_skip.
        # The skip ALWAYS includes the final 1x1 conv (i.e., the LAST layer is
        # the one that does the skip add fused). We re-do that 1x1 here as
        # conv1x1_skip rather than bare conv1x1.
        # Need to re-run the last conv1x1 with skip integration.
        # Re-derive last-layer info.
        raise NotImplementedError(
            "Skip-fused conv1x1 path needs re-architecting — see _run_regular_block_skip"
        )
    return x


def _load_block_weights(blk, data_dir):
    """Load + decode all weights for a block. Returns list aligned with blk.layers."""
    name = blk.name
    if name in ("bn10", "bn11", "bn12"):
        # Pipeline blocks: separate file per layer.
        chains = [_load_chain(data_dir, f"{name}_{i+1}_chain.txt") for i in range(3)]
        result = []
        for layer, chain in zip(blk.layers, chains):
            in_c = layer.in_shape[2]
            out_c = layer.out_shape[2]
            if layer.kind == "conv1x1":
                result.append(_decode_oiyxi8o8(chain, out_c, in_c))
            elif layer.kind == "dw3x3":
                result.append(_decode_dw3x3(chain, in_c))
        return result

    if name in ("bn13", "bn14"):
        # Cascade blocks: L1 = TWO concatenated halves in one file (PUT then GET);
        #                 L2 = single DW file;
        #                 L3 = PUT + GET in SEPARATE files (also concatenated halves).
        # gen_golden's chunk_weights_depth_cascade lays each cascade chunk out as
        # an independent OIYXI8O8 block of shape (full OC, IC/InputSplit), then
        # concatenates them. So a "single OIYXI8O8 (OC, full_IC)" decode mismashes
        # them — we must split-decode-stitch along the IC axis.
        l1_chain = _load_chain(data_dir, f"{name}_1_chain.txt")
        l2_chain = _load_chain(data_dir, f"{name}_2_chain.txt")
        l3_put = _load_chain(data_dir, f"{name}_3_put_chain.txt")
        l3_get = _load_chain(data_dir, f"{name}_3_get_chain.txt")
        l1_layer, l2_layer, l3_layer = blk.layers
        # L1: file = [put_half | get_half], each half is OIYXI8O8(OC=full, IC=full/2).
        l1_full_oc = l1_layer.out_shape[2]
        l1_full_ic = l1_layer.in_shape[2]
        l1_half = l1_full_ic // 2
        half_sz = l1_full_oc * l1_half  # bytes per half
        assert (
            l1_chain.size == 2 * half_sz
        ), f"{name} L1: expected {2*half_sz} bytes, got {l1_chain.size}"
        w_l1_put = _decode_oiyxi8o8(l1_chain[:half_sz], l1_full_oc, l1_half)
        w_l1_get = _decode_oiyxi8o8(l1_chain[half_sz:], l1_full_oc, l1_half)
        w_l1 = np.concatenate([w_l1_put, w_l1_get], axis=1)
        # L2: single DW file (no cascade split for L2 — the kernel produces split
        # outputs but consumes the full DW weights as one buffer).
        w_l2 = _decode_dw3x3(l2_chain, l2_layer.in_shape[2])
        # L3: SEPARATE files for put and get. Each is OIYXI8O8(OC=full, IC=full/2).
        w_l3_put = _decode_oiyxi8o8(
            l3_put, l3_layer.out_shape[2], l3_layer.in_shape[2] // 2
        )
        w_l3_get = _decode_oiyxi8o8(
            l3_get, l3_layer.out_shape[2], l3_layer.in_shape[2] // 2
        )
        w_l3 = np.concatenate([w_l3_put, w_l3_get], axis=1)
        return [w_l1, w_l2, w_l3]

    # Regular blocks: single combined chain file.
    chain = _load_chain(data_dir, f"{name}_chain.txt")
    sizes = []
    for layer in blk.layers:
        in_c = layer.in_shape[2]
        out_c = layer.out_shape[2]
        if layer.kind == "conv1x1":
            sizes.append(in_c * out_c)
        elif layer.kind == "dw3x3":
            sizes.append(9 * in_c)
        else:
            raise ValueError(f"{name}: unhandled layer kind {layer.kind!r}")
    assert (
        sum(sizes) == chain.size
    ), f"{name} chain size {chain.size} != layers {sum(sizes)}"
    cur = 0
    result = []
    for layer, sz in zip(blk.layers, sizes):
        chunk = chain[cur : cur + sz]
        cur += sz
        in_c = layer.in_shape[2]
        out_c = layer.out_shape[2]
        if layer.kind == "conv1x1":
            result.append(_decode_oiyxi8o8(chunk, out_c, in_c))
        elif layer.kind == "dw3x3":
            result.append(_decode_dw3x3(chunk, in_c))
    return result


def _run_block(blk, x, sf, data_dir):
    """Run one logical bottleneck (init / bn0..bn12).

    For skip blocks, the final 1x1 is computed via conv1x1_skip — fusing the
    last conv with the residual add. Otherwise bare conv1x1 / dw3x3.
    """
    name = blk.name
    if name == "init":
        return _run_init(x, sf, data_dir)
    sf_blk = sf[name.upper()]
    x_in = x
    ws = _load_block_weights(blk, data_dir)

    last_idx = len(blk.layers) - 1
    for i, (layer, w) in enumerate(zip(blk.layers, ws)):
        is_last = i == last_idx
        scale = sf_blk[layer.sf_key]
        if layer.kind == "dw3x3":
            x = dw3x3(x, w, scale, layer.stride, np.uint8)
        elif layer.kind == "conv1x1":
            if is_last and blk.skip:
                # Final 1x1 + skip fusion. Skip dtype is uint8 for bn0
                # (input came from init=uint8), int8 otherwise.
                scale_add = sf_blk[blk.skip_sf_key]
                x = conv1x1_skip(x, w, scale, x_in, scale_add)
            else:
                out_dtype = np.uint8 if layer.activation == "relu" else np.int8
                x = conv1x1(x, w, scale, out_dtype)
        else:
            raise ValueError(f"{name}: unhandled layer kind {layer.kind!r}")
    return x


# ---------------------------------------------------------------------------
# Driver — walk NETWORK end to end
# ---------------------------------------------------------------------------
def run(input_i8, weights_dir, scales, return_intermediates=False, stop_after=None):
    """Run mobilenet end-to-end.

    Implemented through bn12. bn13/bn14 (cascade) and post_l1/post_l2 land in
    follow-up.
    """
    x = input_i8
    intermediates = {}
    for blk in NETWORK:
        if blk.name == "post_l1":
            chain = _load_chain(weights_dir, "post_conv_chain.txt")
            x = post_l1_avgpool_conv1x1(x, chain, scales["POST"]["conv1x1_1"])
        elif blk.name == "post_l2":
            # Two FC layers; weights split across 4 tile files each (concatenate
            # along output-channel dim for the bit-exact equivalent).
            for fc_idx, sf_key in enumerate(("FC1", "FC2")):
                chunks = [
                    _load_chain(weights_dir, f"FC{fc_idx+1}_{i}_chain.txt")
                    for i in range(4)
                ]
                full_w = np.concatenate(chunks)
                x = post_l2_fc(x, full_w, scales["POST"][sf_key], in_c_padded=1280)
            intermediates[blk.name] = x.copy()
            print(
                f"  {blk.name:8s} -> shape={x.shape}, dtype={x.dtype}, "
                f"min={x.min()}, max={x.max()}, sum_abs={np.abs(x.astype(np.int64)).sum()}"
            )
            continue
        else:
            x = _run_block(blk, x, scales, weights_dir)
        intermediates[blk.name] = x.copy()
        print(
            f"  {blk.name:8s} -> shape={x.shape}, dtype={x.dtype}, "
            f"min={x.min()}, max={x.max()}, sum_abs={np.abs(x.astype(np.int64)).sum()}"
        )
        if stop_after == blk.name:
            break
    return (x, intermediates) if return_intermediates else x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Pure-numpy mobilenet reference.")
    ap.add_argument(
        "--dump-intermediates",
        metavar="DIR",
        help="Save each block's output as <DIR>/<block_name>.bin (for AIE comparison).",
    )
    args = ap.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "data") + "/"
    scales = json.load(open(data_dir + "scale_factors_final.json"))
    init_blk = nsblock("init")
    in_w, in_h, in_c = init_blk.layers[0].in_shape
    inp = _load_input_image(data_dir, in_h, in_w, in_c)
    print(f"Input: shape={inp.shape}, range=[{inp.min()}, {inp.max()}]\n")
    out, inter = run(inp, data_dir, scales, return_intermediates=True)
    print(f"\nFinal: shape={out.shape}, dtype={out.dtype}")

    # Compare against both references: brevitas golden + actual AIE output.
    # See module docstring re: known issue #3009 (AIE-vs-golden delta=9).
    ours = out.flatten().astype(np.int32)
    ours[ours > 127] -= 256  # interpret uint8-in-uint16 as int8 for fair cmp

    def _cmp(label, path):
        if not os.path.exists(path):
            print(f"\n{label}: skipped ({path} not found)")
            return
        ref = np.loadtxt(path, delimiter=",", dtype=np.int8).astype(np.int32)
        if ref.size != ours.size:
            print(f"\n{label}: shape mismatch ({ref.size} vs {ours.size})")
            return
        diff = ours - ref
        n_match = int((diff == 0).sum())
        max_d = int(np.abs(diff).max())
        mean_d = float(np.abs(diff).mean())
        verdict = (
            "✓ BIT-EXACT" if max_d == 0 else f"diff: max={max_d}, mean={mean_d:.2f}"
        )
        print(f"\n{label}: {n_match}/{ours.size} matches  {verdict}")

    _cmp("vs brevitas golden (data/golden_output.txt)", data_dir + "golden_output.txt")

    if args.dump_intermediates:
        os.makedirs(args.dump_intermediates, exist_ok=True)
        for name, tensor in inter.items():
            tensor.tofile(os.path.join(args.dump_intermediates, f"{name}.bin"))
        print(f"\nDumped {len(inter)} intermediates to {args.dump_intermediates}/")


if __name__ == "__main__":
    _main()
