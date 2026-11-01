# test_compilabledesign_chain.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of designs chained into xclbins with ``--xclbin-input``.

Three designs, each copying its own slice of the input, chain two to an
xclbin: the second xclbin holds both kernels of its chain, the third design's
xclbin starts a new one, and each kernel runs from the xclbin it lands in.
"""

import aie.iron as iron
import numpy as np
import pyxrt
from aie.dialects._aie_enum_gen import AIETileType
from aie.iron import (
    CompileTime,
    ExternalFunction,
    Flow,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.utils import DefaultNPURuntime
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.npukernel import NPUKernel

N = 1024
SLICE = 256
TILE = 64


def copy_slice(*, index: CompileTime[int]):
    in_ty = np.ndarray[(N,), np.dtype[np.int32]]
    out_ty = np.ndarray[(SLICE,), np.dtype[np.int32]]
    feed = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    drain = Tile(col=1, row=0, tile_type=AIETileType.ShimNOCTile)
    link = Flow(feed, drain, src_channel=1, dst_channel=0)

    def seq(A, C):
        link.fill(A, tap=A[index * SLICE : (index + 1) * SLICE])
        link.drain(C, wait=True)

    rt = Runtime(seq, [in_ty, out_ty])
    rt.add_flow(link)
    return Program(iron.get_current_device(), rt).resolve_program()


def add_bias(*, func: CompileTime[object]):
    ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile = np.ndarray[(TILE,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile, name="in")
    of_out = ObjectFifo(tile, name="out")

    def core(of_in, of_out, fn):
        for _ in range_(N // TILE):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            fn(a, b)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core, fn_args=[of_in.cons(), of_out.prod(), func])

    def seq(A, C, feed, drain):
        feed.fill(A)
        drain.drain(C, wait=True)

    rt = Runtime(seq, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


class TwoPerXclbin(CompilableDesign):
    XCLBIN_KERNELS = 2


def test_chained_kernels_run_from_the_xclbins_they_land_in():
    designs = [
        CompilableDesign(copy_slice, compile_kwargs={"index": i}) for i in range(3)
    ]
    chained = TwoPerXclbin.chain(designs, "slices")
    images = [image for _, image, _ in chained]
    assert images[0] == images[1] != images[2]
    first, second, third = (design.aiecc_flags for design, _, _ in chained)
    assert not any(f.startswith("--xclbin-input") for f in first + third)
    assert any(f.startswith("--xclbin-input") for f in second)
    kernels = {k.get_name() for k in pyxrt.xclbin(str(images[0])).get_kernels()}
    assert kernels == {chained[0][2], chained[1][2]}

    host = np.random.default_rng(5).integers(-(2**31), 2**31 - 1, N, dtype=np.int32)
    for index, (design, image, kernel) in enumerate(chained):
        a = iron.tensor(host, dtype=np.int32, device="npu")
        c = iron.zeros(SLICE, dtype=np.int32, device="npu")
        entry = design.get_cache_entry()
        assert entry is not None
        DefaultNPURuntime.load_and_run(
            NPUKernel(image, entry.insts, kernel_name=kernel), [a, c]
        )
        c.to("cpu")
        np.testing.assert_array_equal(
            c.numpy(), host[index * SLICE : (index + 1) * SLICE]
        )


def test_a_chain_of_one_keeps_the_default_kernel():
    [(design, image, kernel)] = CompilableDesign.chain(
        [CompilableDesign(copy_slice, compile_kwargs={"index": 1})], "one"
    )
    assert kernel == "MLIR_AIE" and design.aiecc_flags == ()


def test_an_image_rebuilt_in_place_rebuilds_the_image_chained_onto_it(tmp_path):
    header = tmp_path / "bias.h"
    tile = np.ndarray[(TILE,), np.dtype[np.int32]]
    host = np.random.default_rng(6).integers(-1000, 1000, N, dtype=np.int32)
    for bias in (1, 2):
        header.write_text(f"#define BIAS {bias}\n")
        func = ExternalFunction(
            "add_bias",
            source_string=(
                '#include "bias.h"\n'
                'extern "C" void add_bias(int *a, int *b) {\n'
                f"  for (int i = 0; i < {TILE}; i++) b[i] = a[i] + BIAS;\n"
                "}\n"
            ),
            arg_types=[tile, tile],
            include_dirs=[str(tmp_path)],
        )
        designs = [
            CompilableDesign(add_bias, compile_kwargs={"func": func}),
            CompilableDesign(copy_slice, compile_kwargs={"index": 0}),
        ]
        (design, image, kernel), _ = TwoPerXclbin.chain(designs, "bias")
        a = iron.tensor(host, dtype=np.int32, device="npu")
        c = iron.zeros(N, dtype=np.int32, device="npu")
        entry = design.get_cache_entry()
        assert entry is not None
        DefaultNPURuntime.load_and_run(
            NPUKernel(image, entry.insts, kernel_name=kernel), [a, c]
        )
        c.to("cpu")
        np.testing.assert_array_equal(c.numpy(), host + bias)
