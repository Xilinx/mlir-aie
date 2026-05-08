#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Bit-exact pure-numpy reference for MobileNet V3.

Drives the network from network_spec.NETWORK; mirrors the AIE kernel arithmetic
exactly (int32 accumulator, round-half-up shift, saturating clamp). Reads the
same int8 weight files (`*_chain.txt`) the AIE design loads.

The reference is the algorithm "onramp": read it once to understand mobilenet
end-to-end, then map blocks back to the IRON design.

Usage:
    python3 mobilenet_numpy.py
    # expects data/before_ifm_mem_fmt_1x1.txt + data/golden_output.txt;
    # asserts bit-exactness against golden.

Programmatic:
    from mobilenet_numpy import run
    out, intermediates = run(input_i8, weights_dir, scales, return_intermediates=True)
    # intermediates["bn3"] is the (W,H,C) tensor after bn3.

STATUS: SCAFFOLD. conv1x1 is implemented; remaining kernels (dw3x3, skip-add,
avgpool, FC, cascade variants) raise NotImplementedError. Bit-exact validation
against golden_output.txt will land in the follow-up commit.
"""

import os
import sys
import json

import numpy as np

from network_spec import NETWORK, block as nsblock


# ---------------------------------------------------------------------------
# Quantization arithmetic (matches aie_kernels/aie2/conv2dk*.cc scalar paths)
# ---------------------------------------------------------------------------
def _srs(acc_i32, scale):
    """Round-half-up shift right by `scale` bits — the AIE 'srs' op.

    sum_srs = (sum + (1 << (scale-1))) >> scale
    Mirrors the C scalar kernels exactly. `acc_i32` is int32; output is int32
    (caller saturates to int8/uint8/uint16 as needed).
    """
    if scale <= 0:
        return acc_i32 >> 0 if scale == 0 else acc_i32 << (-scale)
    bias = 1 << (scale - 1)
    return (acc_i32.astype(np.int64) + bias).astype(np.int64) >> scale


def _saturate_i8(x):
    return np.clip(x, -128, 127).astype(np.int8)


def _saturate_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _saturate_u16(x):
    return np.clip(x, 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# Weight layout decoders
# ---------------------------------------------------------------------------
# AIE pointwise (1x1) weights are stored OIYXI8O8 = flat[oc/8, ic/8, ic8, oc8]
# (with Y=X=1 for 1x1; for 3x3 DW the layout is similar with Y,X axes between).
def _decode_oiyxi8o8(flat_i8, out_c, in_c, ky=1, kx=1):
    """Convert AIE pointwise / 3x3 weights from flat OIYXI8O8 → standard OIYX.

    Standard: shape (out_c, in_c, ky, kx)
    Layout:   flat[oc/8, ic/8, y, x, ic8, oc8]
    """
    assert out_c % 8 == 0, f"out_c={out_c} must be a multiple of 8"
    assert in_c % 8 == 0, f"in_c={in_c} must be a multiple of 8"
    expected = out_c * in_c * ky * kx
    assert (
        flat_i8.size == expected
    ), f"weight size {flat_i8.size} != expected {expected}"
    blocked = flat_i8.reshape(out_c // 8, in_c // 8, ky, kx, 8, 8)
    # Permute axes from (Oo, Io, Y, X, Ii, Oi) → (Oo, Oi, Io, Ii, Y, X)
    standard = blocked.transpose(0, 5, 1, 4, 2, 3).reshape(out_c, in_c, ky, kx)
    return standard.astype(np.int32)  # promote for arithmetic


def _decode_dw3x3(flat_i8, channels):
    """Decode AIE depthwise 3x3 weights → standard (channels, 3, 3).

    Depthwise has only one filter per channel (no I dim). AIE stores it
    blocked by channel/8: flat[c/8, y, x, c8] = wts[c, y, x].
    Total bytes: 9 * channels.
    """
    assert channels % 8 == 0
    expected = 9 * channels
    assert flat_i8.size == expected
    blocked = flat_i8.reshape(channels // 8, 3, 3, 8)
    standard = blocked.transpose(0, 3, 1, 2).reshape(channels, 3, 3)
    return standard.astype(np.int32)


# ---------------------------------------------------------------------------
# Activation layout
# ---------------------------------------------------------------------------
# AIE per-row activations are stored CYXC8 = flat[ic/8, x, ic8] per row.
# For our reference we use standard (H, W, C) tensors throughout.
def _load_input_image(data_dir, in_h, in_w, in_c):
    """Read before_ifm_mem_fmt_1x1.txt as (H, W, C) int8.

    File is uint8 in (C, H, W) order per test_mobilenet.py; this is the input
    activation pre-quantization (centered around -50 for the dark image).
    """
    path = os.path.join(data_dir, "before_ifm_mem_fmt_1x1.txt")
    raw = np.loadtxt(
        path, delimiter=",", dtype=np.int8
    )  # int8 even though stored as uint8 nums
    assert raw.size == in_h * in_w * in_c, f"input size {raw.size} != {in_h*in_w*in_c}"
    chw = raw.reshape(in_c, in_h, in_w)
    return chw.transpose(1, 2, 0).copy()  # (H, W, C)


# ---------------------------------------------------------------------------
# Kernel: 1x1 pointwise convolution
# Mirrors aie_kernels/aie2/conv2dk1_i8.cc scalar:
#   acc_i32 = sum_{ic} input[h,w,ic] * weight[oc,ic]
#   acc_srs = (acc_i32 + (1 << (scale-1))) >> scale
#   out     = saturate(acc_srs, dtype)
# ---------------------------------------------------------------------------
def conv1x1(x_i8, w_oiyx_i32, scale, out_dtype):
    """1x1 convolution. x: (H, W, IC); w: (OC, IC, 1, 1); returns (H, W, OC).

    out_dtype: np.int8 (no-relu) or np.uint8 (relu fused: clamps to [0, 255]).
    """
    H, W, IC = x_i8.shape
    OC = w_oiyx_i32.shape[0]
    assert w_oiyx_i32.shape == (OC, IC, 1, 1)
    # GEMM: (H*W, IC) @ (IC, OC) → (H*W, OC) in int32.
    flat = x_i8.astype(np.int32).reshape(-1, IC)
    w2d = w_oiyx_i32.reshape(OC, IC).T  # (IC, OC)
    acc = flat @ w2d  # (H*W, OC) int32
    acc_srs = _srs(acc, scale)
    if out_dtype == np.int8:
        out = _saturate_i8(acc_srs)
    elif out_dtype == np.uint8:
        out = _saturate_u8(acc_srs)
    else:
        raise ValueError(f"unsupported out_dtype {out_dtype}")
    return out.reshape(H, W, OC)


# ---------------------------------------------------------------------------
# Kernel stubs — implemented in the follow-up commit
# ---------------------------------------------------------------------------
def conv3x3(x, w_oiyx, scale, stride, out_dtype):
    """3x3 conv (used by `init` only — full conv, not depthwise)."""
    raise NotImplementedError("conv3x3 — landing in follow-up")


def dw3x3(x, w_chw, scale, stride, out_dtype):
    """Depthwise 3x3. x: (H, W, C); w: (C, 3, 3); returns (H', W', C)."""
    raise NotImplementedError("dw3x3 — landing in follow-up")


def skip_add(activation, skip, scale_add, out_dtype):
    """Element-wise add: out = saturate(srs(activation + skip, scale_add))."""
    raise NotImplementedError("skip_add — landing in follow-up")


def avgpool_conv1x1(x, w_oiyx, scale):
    """post_l1: 1x1 expand to 960 fused with global avg-pool to (1,1,960),
    then padded to 1280 channels."""
    raise NotImplementedError("avgpool_conv1x1 — landing in follow-up")


def fc(x, w_oiyx, scale, out_dtype):
    """Fully-connected (1x1 conv on (1,1,C) input)."""
    raise NotImplementedError("fc — landing in follow-up")


# ---------------------------------------------------------------------------
# Driver — walk NETWORK and apply each layer
# ---------------------------------------------------------------------------
def _load_chain(data_dir, filename):
    """Load a *_chain.txt file as flat int8 array."""
    path = os.path.join(data_dir, filename)
    return np.loadtxt(path, delimiter=",", dtype=np.int8)


def _split_chain_for_block(blk, chain):
    """Slice a *_chain.txt into per-layer weight arrays.

    Returns a list aligned with blk.layers — one int8 array per layer.
    Layout assumption: layers are concatenated in order; sizes derived from
    layer in/out channels.
    """
    sizes = []
    for layer in blk.layers:
        in_c = layer.in_shape[2]
        out_c = layer.out_shape[2]
        if layer.kind == "conv1x1" or layer.kind == "fc":
            sizes.append(in_c * out_c)
        elif layer.kind == "conv3x3":
            sizes.append(in_c * out_c * 9)
        elif layer.kind == "dw3x3":
            sizes.append(9 * in_c)  # depthwise: one 3x3 per channel
        elif layer.kind == "avgpool":
            # post_l1 weights are the 1x1 expansion (80 -> 960), fused with pool
            sizes.append(in_c * 960)
        else:
            raise ValueError(f"unknown layer kind: {layer.kind}")
    assert sum(sizes) == chain.size, (
        f"{blk.name} chain size mismatch: file has {chain.size}, "
        f"layers want {sum(sizes)}"
    )
    out = []
    cur = 0
    for sz in sizes:
        out.append(chain[cur : cur + sz])
        cur += sz
    return out


def _apply_layer(layer, x, w_flat, scale):
    """Dispatch to the right kernel based on layer.kind."""
    in_c = layer.in_shape[2]
    out_c = layer.out_shape[2]
    if layer.activation == "relu":
        out_dtype = np.uint8
    elif layer.activation is None:
        out_dtype = np.int8
    else:
        raise ValueError(f"unknown activation {layer.activation!r}")

    if layer.kind == "conv1x1":
        w = _decode_oiyxi8o8(w_flat, out_c, in_c)
        return conv1x1(x, w, scale, out_dtype)
    if layer.kind == "conv3x3":
        w = _decode_oiyxi8o8(w_flat, out_c, in_c, ky=3, kx=3)
        return conv3x3(x, w, scale, layer.stride, out_dtype)
    if layer.kind == "dw3x3":
        w = _decode_dw3x3(w_flat, in_c)
        return dw3x3(x, w, scale, layer.stride, out_dtype)
    if layer.kind == "avgpool":
        return avgpool_conv1x1(x, w_flat, scale)
    if layer.kind == "fc":
        return fc(x, w_flat, scale, out_dtype)
    raise ValueError(f"unhandled layer: {layer.kind}")


def run(input_i8, weights_dir, scales, return_intermediates=False, stop_after=None):
    """Run mobilenet end-to-end in numpy.

    input_i8:           (224, 224, 8) int8
    weights_dir:        directory containing *_chain.txt files
    scales:             dict from scale_factors_final.json
    return_intermediates: if True, also returns dict[block_name -> output]
    stop_after:         block name; stop execution after this block (debugging)
    """
    x = input_i8
    intermediates = {}
    for blk in NETWORK:
        x_in = x
        # Cascade and post blocks not yet wired up to single-chain.txt files.
        if blk.name in ("bn13", "bn14", "post_l1", "post_l2"):
            raise NotImplementedError(
                f"{blk.name}: weight loading + kernels land in follow-up"
            )
        chain = _load_chain(weights_dir, f"{blk.name}_chain.txt")
        per_layer_weights = _split_chain_for_block(blk, chain)
        for layer, w_flat in zip(blk.layers, per_layer_weights):
            sf = scales[_sf_block_key(blk.name)][layer.sf_key]
            x = _apply_layer(layer, x, w_flat, sf)
        if blk.skip:
            sf_add = scales[_sf_block_key(blk.name)][blk.skip_sf_key]
            x = skip_add(x, x_in, sf_add, x.dtype)
        intermediates[blk.name] = x.copy()
        if stop_after == blk.name:
            break
    return (x, intermediates) if return_intermediates else x


def _sf_block_key(name):
    """Block name → scale_factors_final.json key ('bn3' → 'BN3', 'init' → 'INIT')."""
    if name.startswith("post"):
        return "POST"
    return name.upper()


# ---------------------------------------------------------------------------
# CLI: validate against golden_output.txt
# ---------------------------------------------------------------------------
def _main():
    data_dir = os.path.join(os.path.dirname(__file__), "data") + "/"
    scales = json.load(open(data_dir + "scale_factors_final.json"))
    init_blk = nsblock("init")
    in_w, in_h, in_c = init_blk.layers[0].in_shape
    inp = _load_input_image(data_dir, in_h, in_w, in_c)
    print(f"Input loaded: shape={inp.shape}, dtype={inp.dtype}")
    print(f"  range: [{inp.min()}, {inp.max()}]")

    # SCAFFOLD: run only conv1x1 layers we have implemented.
    # Prove the driver wiring by running the FIRST conv1x1 layer (bn1's L1).
    print("\nSmoke test: bn1's first 1x1 layer (16 → 64 channels) ...")
    bn1 = nsblock("bn1")
    chain = _load_chain(data_dir, "bn1_chain.txt")
    per_layer = _split_chain_for_block(bn1, chain)
    # Fake input: zeros at bn1 input shape (since init isn't implemented yet).
    fake_in = np.zeros(bn1.layers[0].in_shape, dtype=np.int8)
    out = _apply_layer(bn1.layers[0], fake_in, per_layer[0], scales["BN1"]["conv1x1_1"])
    print(f"  bn1 conv1x1_1 output: shape={out.shape}, dtype={out.dtype}")
    assert out.shape == bn1.layers[0].out_shape, "shape mismatch"
    assert (out == 0).all(), "zero input should give zero output (no bias)"
    print("  ✓ scaffold sanity OK (correct shape; identity-on-zero)")

    print(
        "\nSCAFFOLD ONLY: dw3x3 / skip_add / conv3x3 / avgpool / FC / cascade "
        "are stubs.\nFull bit-exact validation against data/golden_output.txt "
        "lands in the follow-up commit."
    )


if __name__ == "__main__":
    _main()
