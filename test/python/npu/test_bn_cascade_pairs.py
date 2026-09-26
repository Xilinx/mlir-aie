# test_bn_cascade_pairs.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""Device tests for MobileNet's cascade-split 1x1 conv pairs.

bn13 and bn14 split each 1x1 conv across two cores. The PUT core sums its
half of the input channels and streams 7 pixels x 8 output channels of
partial sums onto the cascade. The GET core adds its own half and
requantizes. Neither half has an output the generic one-Worker builder
can judge, so this file builds each pair the way ``cascade.py`` calls it
and judges the rows against a numpy model of the whole conv.

* ``width``: ``bn_conv2dk1_partial_put_i8`` and
  ``bn_conv2dk1_partial_get_relu_i8``. Both read the same int8 row, PUT the
  first half of its channels and GET the second; the GET core writes
  ``sat_u8(round_even(sum, scale))``.
* ``input``: ``bn_conv2dk1_input_split_partial_put_ui8`` and
  ``bn_conv2dk1_input_split_partial_skip_get``. Each reads its own uint8
  half row; the GET core adds a skip row as ``bn_conv2dk1_skip`` does.

Each call covers one block of 8 output channels for the 7 pixels the
kernels are written for. ``weight_index`` picks which of the
``output_split`` weight chunks a call's block sits in.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.extras.dialects.memref import view as memref_view
from aie.iron import (
    Buffer,
    CascadeFlow,
    CompileTime,
    InOut,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    kernels,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernels.conv import _requant_even

pytestmark = pytest.mark.supported_devices("npu2")

_W = 7
_INPUT_SPLIT = 2


def _pair_kernels(kind, block, ic, oc, weight_count):
    if kind == "width":
        put = kernels.bn_conv2dk1_partial_put_i8(
            input_width=_W,
            input_channels=ic,
            weight_count=weight_count,
            block_index=block,
        )
        get = kernels.bn_conv2dk1_partial_get_relu_i8(
            input_width=_W,
            input_channels=ic,
            output_channels=oc,
            weight_count=weight_count,
            block_index=block,
        )
    else:
        put = kernels.bn_conv2dk1_input_split_partial_put_ui8(
            input_width=_W,
            input_channels=ic // _INPUT_SPLIT,
            weight_count=weight_count,
            block_index=block,
        )
        get = kernels.bn_conv2dk1_input_split_partial_skip_get(
            input_width=_W,
            input_channels=ic // _INPUT_SPLIT,
            output_channels=oc,
            weight_count=weight_count,
            block_index=block,
        )
    return put, get


def _weights(seed, count):
    """Return the PUT and GET cores' weights.

    Each is ``output_split`` chunks of ``[OC/8/output_split][IC/2/8][8][8]``.
    """
    rng = np.random.default_rng([seed, 1])
    return [rng.integers(-128, 128, count, dtype=np.int8) for _ in range(2)]


@iron.jit
def _pair_design(
    *tensors: InOut,
    kind: CompileTime[str],
    block: CompileTime[int],
    ic: CompileTime[int],
    oc: CompileTime[int],
    output_split: CompileTime[int],
    rows: CompileTime[int],
    scale: CompileTime[int],
    skip_scale: CompileTime[int],
    seed: CompileTime[int],
):
    oc8 = oc // (8 * output_split)
    weight_count = ic // _INPUT_SPLIT * oc // output_split
    put, get = _pair_kernels(kind, block, ic, oc, weight_count)
    put_in_ty = put.arg_types()[0]
    get_in_ty, _, out_ty = get.arg_types()[:3]
    # Static weights leave the GET core's two input channels to the
    # activations and the skip row.
    wp, wg = (
        Buffer(np.ndarray[(w.size,), np.dtype[np.int8]], initial_value=w, name=n)
        for w, n in zip(_weights(seed, weight_count * output_split), ("wp", "wg"))
    )

    of_put_in = ObjectFifo(put_in_ty, name="put_in")
    of_get_in = of_put_in if kind == "width" else ObjectFifo(get_in_ty, name="get_in")
    of_out = ObjectFifo(out_ty, name="out")
    of_skip = ObjectFifo(out_ty, name="skip") if kind == "input" else None

    def chunk(wts, wi):
        return memref_view(wts.op, [weight_count], shift=wi * weight_count)

    def put_core(of_in, wts, k):
        for _ in range_(rows):
            x = of_in.acquire(1)
            for wi in range(output_split):
                for o in range_(oc8):
                    k(x, chunk(wts, wi), _W, ic, oc, _INPUT_SPLIT, wi, 0, o)
            of_in.release(1)

    def get_relu_core(of_in, wts, of_out, k):
        for _ in range_(rows):
            x, y = of_in.acquire(1), of_out.acquire(1)
            for wi in range(output_split):
                for o in range_(oc8):
                    w = chunk(wts, wi)
                    k(x, w, y, _W, ic, oc, scale, _INPUT_SPLIT, output_split, wi, 0, o)
            of_in.release(1)
            of_out.release(1)

    def get_skip_core(of_in, wts, of_out, of_skip, k):
        for _ in range_(rows):
            x, y, s = of_in.acquire(1), of_out.acquire(1), of_skip.acquire(1)
            for wi in range(output_split):
                for o in range_(oc8):
                    w = chunk(wts, wi)
                    k(
                        *(x, w, y, s, _W, ic, oc, scale, skip_scale),
                        *(_INPUT_SPLIT, output_split, wi, 0, o),
                    )
            of_in.release(1)
            of_out.release(1)
            of_skip.release(1)

    # The cascade runs north to south.
    put_worker = Worker(put_core, [of_put_in.cons(), wp, put], tile=Tile(0, 3))
    if kind == "width":
        get_fn, get_args = get_relu_core, [of_get_in.cons(), wg, of_out.prod(), get]
    else:
        get_fn = get_skip_core
        get_args = [of_get_in.cons(), wg, of_out.prod(), of_skip.cons(), get]
    get_worker = Worker(get_fn, get_args, tile=Tile(0, 2))
    CascadeFlow(put_worker, get_worker)

    handles = [of_put_in.prod()]
    if kind == "input":
        handles += [of_get_in.prod(), of_skip.prod()]
    handles.append(of_out.cons())

    def seq(*args):
        hosts, handles = args[: len(args) // 2], args[len(args) // 2 :]
        for handle, host in zip(handles[:-1], hosts[:-1]):
            handle.fill(host)
        handles[-1].drain(hosts[-1], wait=True)

    fifo_tys = [put_in_ty] + ([get_in_ty, out_ty] if kind == "input" else []) + [out_ty]
    host_tys = [np.ndarray[(rows * t.__args__[0][0],), t.__args__[1]] for t in fifo_tys]
    rt = Runtime(seq, host_tys + handles)
    return Program(
        iron.get_current_device(), rt, workers=[put_worker, get_worker]
    ).resolve_program()


def _conv_half(x, wts, ic_half, oc):
    """int64 ``[rows][OC/8][7][8]`` sums over one core's ``[rows][ic_half/8][W][8]``."""
    w = wts.reshape(oc // 8, ic_half // 8, 8, 8).astype(np.int64)
    return np.einsum("rcpi,ocij->ropj", x[:, :, :_W].astype(np.int64), w)


def _run_pair(kind, block, ic, oc, output_split, rows, scale, skip_scale, seed):
    rng = np.random.default_rng(seed)
    half = ic // _INPUT_SPLIT
    wp, wg = _weights(seed, half * oc)
    if kind == "width":
        x = rng.integers(-128, 128, rows * ic * _W, dtype=np.int8)
        xs = x.reshape(rows, ic // 8, _W, 8)
        xp, xg = xs[:, : half // 8], xs[:, half // 8 :]
        ins = [x]
    else:
        xp_flat = rng.integers(0, 256, rows * half * _W, dtype=np.uint8)
        xg_flat = rng.integers(0, 256, rows * half * _W, dtype=np.uint8)
        xp = xp_flat.reshape(rows, half // 8, _W, 8)
        xg = xg_flat.reshape(rows, half // 8, _W, 8)
        skip = rng.integers(-128, 128, rows * oc * _W, dtype=np.int8)
        ins = [xp_flat, xg_flat, skip]
    acc = _conv_half(xp, wp, half, oc) + _conv_half(xg, wg, half, oc)
    if kind == "width":
        expected = conv = _requant_even(acc, scale)
    else:
        conv = _requant_even(acc, scale, -128, 127, np.int64)
        total = conv + skip.reshape(rows, oc // 8, _W, 8)
        if skip_scale:
            expected = _requant_even(total, skip_scale, -128, 127, np.int8)
        else:
            expected = np.clip(total, -128, 127).astype(np.int8)
    out = np.full(rows * oc * _W, 0x5A, dtype=expected.dtype)
    tensors = [iron.tensor(t, dtype=t.dtype, device="npu") for t in [*ins, out]]
    _pair_design(
        *tensors,
        kind=kind,
        block=block,
        ic=ic,
        oc=oc,
        output_split=output_split,
        rows=rows,
        scale=scale,
        skip_scale=skip_scale,
        seed=seed,
    )
    return tensors[-1].numpy().copy(), expected.reshape(-1), conv


# (kind, block, ic, oc, output_split, rows, scale, skip_scale): the bn13/bn14
# shapes as cascade.py runs them (bn13 adds its skip unscaled), then small
# ones with other chunk counts.
_CASES = [
    pytest.param("width", 13, 80, 960, 2, 7, 8, 1, id="width-bn13"),
    pytest.param("input", 13, 960, 80, 2, 7, 11, 0, id="input-bn13"),
    pytest.param("width", 14, 80, 960, 2, 7, 8, 1, id="width-bn14"),
    pytest.param("input", 14, 960, 80, 2, 7, 11, 1, id="input-bn14"),
    pytest.param("width", 13, 16, 32, 2, 2, 7, 1, id="width-ic16"),
    pytest.param("width", 13, 48, 64, 1, 1, 7, 1, id="width-ic48-split1"),
    pytest.param("input", 13, 32, 16, 2, 2, 8, 2, id="input-ic32"),
    pytest.param("input", 13, 112, 48, 3, 1, 9, 1, id="input-ic112-split3"),
]


@pytest.mark.parametrize("kind,block,ic,oc,output_split,rows,scale,skip_scale", _CASES)
@pytest.mark.parametrize("seed", [0, 1])
def test_bn_cascade_pair(
    kind, block, ic, oc, output_split, rows, scale, skip_scale, seed
):
    got, expected, conv = _run_pair(
        kind, block, ic, oc, output_split, rows, scale, skip_scale, seed
    )
    bad = got != expected
    assert not bad.any(), (
        f"{int(bad.sum())} of {got.size} outputs differ, max |d| "
        f"{int(np.abs(got.astype(np.int64) - expected.astype(np.int64)).max())}"
    )
    # The conv's requantization reaches both clamps, so a pass covers them.
    lo, hi = (0, 255) if kind == "width" else (-128, 127)
    assert (conv == lo).any() and (conv == hi).any()
