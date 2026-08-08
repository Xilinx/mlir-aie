# mm_activation_epilogue/mm_activation_epilogue.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One resident program, three RTP-selected GEMM-epilogue modes -- IRON API
+ ``@iron.jit``.

NPU2-only: the underlying ``mm_activation_epilogue_row`` kernel lives under
``aie_kernels/aie2p/`` and has no aie2 counterpart.

``float32`` in, ``float32`` out, per-element:

  * mode 0 (identity): ``out = acc``
  * mode 1 (SiLU, hybrid precision): ``out = acc * sigmoid(acc)``
  * mode 2 (GELU, tanh approximation): ``out = 0.5*acc*(1+tanh(...))``

The design compiles ONCE into a single xclbin. A per-core
``Buffer(..., use_write_rtp=True)`` carries the mode; a
``WorkerRuntimeBarrier`` synchronizes each of the three dispatches (one per
mode) with the workers so the new mode value is visible before a worker
starts the next phase -- structurally this mirrors ``ml/scale_shift``'s
two-phase (multiply, then add) runtime-parameter dispatch, extended to
three phases and three separate output tensors so each mode's result can
be checked against its own reference independently. No reconfiguration
(no new xclbin, no new hardware context) happens between phases; only the
RTP word and the DMA fill/drain addressing differ.
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.utils.hostruntime.argparse import device_from_args
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.helpers.util import np_ndarray_type_get_shape
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_SRC = (
    Path(__file__).resolve().parents[3] / "aie_kernels/aie2p/mm_activation_epilogue.cc"
)

_MODE_IDENTITY = 0
_MODE_SILU = 1
_MODE_GELU = 2


def _epilogue_extern(tile_ty):
    return ExternalFunction(
        "mm_activation_epilogue_row",
        source_file=str(_KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, np.int32, np.int32],
        include_dirs=[config.cxx_header_path()],
    )


@iron.jit
def mm_activation_epilogue(
    a_in: In,
    identity_out: Out,
    silu_out: Out,
    gelu_out: Out,
    *,
    size: CompileTime[int] = 65536,
    n_cores: CompileTime[int] = 2,
):
    tile_size = 1024
    if size % (tile_size * n_cores) != 0:
        raise ValueError(f"size ({size}) must be a multiple of {tile_size * n_cores}")
    tiles_per_core = size // tile_size // n_cores

    tensor_ty = np.ndarray[(size,), np.dtype[np.float32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    memtile_ty = np.ndarray[(tile_size * n_cores,), np.dtype[np.float32]]

    def _split(of, name):
        offsets = [
            (np.prod(np_ndarray_type_get_shape(memtile_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
        return of.cons().split(
            offsets,
            obj_types=[tile_ty] * n_cores,
            names=[f"{name}{i}" for i in range(n_cores)],
        )

    def _join(of, name):
        offsets = [
            (np.prod(np_ndarray_type_get_shape(memtile_ty)) // n_cores) * i
            for i in range(n_cores)
        ]
        return of.prod().join(
            offsets,
            obj_types=[tile_ty] * n_cores,
            names=[f"{name}{i}" for i in range(n_cores)],
        )

    # One input, one output ObjectFifo per core -- the AIE2P compute tile's
    # DMA channel budget (2 in / 2 out) does not stretch to a separate
    # output per mode, and the real epilogue this mirrors only ever has one
    # input and one output tile anyway (the C accumulator in, the
    # activated tile out). All three modes reuse this SAME pair; only the
    # RTP `mode` word and which host tensor the runtime sequence drains
    # into change between epochs.
    inA = ObjectFifo(memtile_ty, name="inA")
    inA_fifos = _split(inA, "memA")
    outC = ObjectFifo(memtile_ty, name="outC")
    outC_fifos = _join(outC, "memC")

    epilogue_fn = _epilogue_extern(tile_ty)

    modes = [
        Buffer(
            np.ndarray[(1,), np.dtype[np.int32]],
            name=f"mode{i}",
            initial_value=np.array([_MODE_IDENTITY], dtype=np.int32),
            use_write_rtp=True,
        )
        for i in range(n_cores)
    ]
    barriers = [WorkerRuntimeBarrier() for _ in range(n_cores)]

    def core_fn(of_a, of_c, epilogue, my_mode, barrier):
        # One "epoch" per host dispatch: block until the host has written
        # this epoch's mode and released the barrier, run one full pass
        # over this core's tiles under that mode, then release back to the
        # host and (per the AIE core's implicit outer loop) return to
        # waiting for the next epoch -- no new xclbin, no new hw_context.
        barrier.wait_for_value(1)
        mode = my_mode[0]
        for _ in range_(tiles_per_core):
            elem_in = of_a.acquire(1)
            elem_out = of_c.acquire(1)
            epilogue(elem_in, elem_out, tile_size, mode)
            of_a.release(1)
            of_c.release(1)
        barrier.release_with_value(1)

    workers = [
        Worker(
            core_fn,
            fn_args=[
                inA_fifos[i].cons(),
                outC_fifos[i].prod(),
                epilogue_fn,
                modes[i],
                barriers[i],
            ],
        )
        for i in range(n_cores)
    ]

    def _set_modes_to(value):
        for m in modes:
            m[0] = value

    def sequence(a, id_out, silu_result, gelu_result, inA_h, outC_h):
        # One epoch per mode: write this epoch's RTP mode, release every
        # core's barrier for one pass, fill the SAME input `a` again, and
        # drain the SAME output ObjectFifo into that mode's own host
        # tensor. No new xclbin, no new hw_context between epochs -- only
        # the RTP word and the drain destination differ.
        for mode_value, out_tensor in (
            (_MODE_IDENTITY, id_out),
            (_MODE_SILU, silu_result),
            (_MODE_GELU, gelu_result),
        ):
            _set_modes_to(mode_value)
            for barrier in barriers:
                barrier.set(1)
            tg = TaskGroup()
            inA_h.fill(a, group=tg)
            outC_h.drain(out_tensor, wait=True, group=tg)
            tg.finish()

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            tensor_ty,
            tensor_ty,
            tensor_ty,
            inA.prod(),
            outC.cons(),
        ],
    )

    device = iron.get_current_device()
    return Program(device, rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE mm_activation_epilogue")
    add_compile_args(p, with_elf=True)
    p.add_argument("-l", "--length", type=int, default=65536, help="elements")
    p.add_argument("-co", "--cores", type=int, default=2, help="number of cores")
    return p


def _compile_kwargs(opts):
    return dict(size=opts.length, n_cores=opts.cores)


def _silu_ref_f32(x):
    return x / (1.0 + np.exp(-x))


def _gelu_tanh_ref_f32(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    n = opts.length

    a_np = rng.uniform(-8.0, 8.0, size=(n,)).astype(np.float32)
    a_t = iron.tensor(a_np, dtype=np.float32, device="npu")
    id_t = iron.zeros_like(a_t)
    silu_t = iron.zeros_like(a_t)
    gelu_t = iron.zeros_like(a_t)

    mm_activation_epilogue(a_t, id_t, silu_t, gelu_t, **_compile_kwargs(opts))

    assert_pass(id_t.numpy(), a_np, atol=0.0, fail_msg="identity mode mismatch")
    assert_pass(
        silu_t.numpy(),
        _silu_ref_f32(a_np),
        atol=0.05,
        fail_msg="SiLU mode mismatch",
    )
    assert_pass(
        gelu_t.numpy(),
        _gelu_tanh_ref_f32(a_np),
        atol=0.05,
        fail_msg="GELU mode mismatch",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        mm_activation_epilogue,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
