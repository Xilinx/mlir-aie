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
from aie.iron import CompileTime, Flow, Program, Runtime
from aie.iron.device import Tile
from aie.utils import DefaultNPURuntime
from aie.utils.compile.jit.compilabledesign import CompilableDesign
from aie.utils.npukernel import NPUKernel

N = 1024
SLICE = 256


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
