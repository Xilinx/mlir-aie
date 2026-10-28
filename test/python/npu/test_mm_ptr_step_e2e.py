# test_mm_ptr_step_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""Two int8 16x16x16 products from one buffer, the pointers stepped between.

The #3778 pattern: the kernel calls an inlined 4x2 mmul matmul twice, stepping
A, B and C by one product after each call. ``issue`` is the matmul as filed;
before Xilinx/llvm-aie#1313's fix, Peano based some of its second call's A
loads on the first call's pointer. ``shipped`` is ``mm_aie2.h``'s.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.utils.config import aie_kernels_dir

PRODUCTS = 2
DIM = 16
R, S, T = 4, 8, 8

STEP_AND_CALL = """
extern "C" void SYMBOL(const int8 *a, const int8 *b, int32 *c) {
  for (int i = 0; i < 512; i += 16)
    aie::store_v(c + i, aie::zeros<int32, 16>());
  for (int i = 0; i < 2; i++) {
    MATMUL(a, b, c);
    a += 256;
    b += 256;
    c += 256;
  }
}
"""

SHIPPED = '#include "linalg/mm_aie2.h"\n' + STEP_AND_CALL.replace(
    "SYMBOL", "mm_ptr_step_shipped"
).replace(
    "MATMUL",
    "matmul_vectorized_4x2_mmul<int8, int32, 4, 2, 2, 4, 8, 8, false, true>",
)

ISSUE = """
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

template <unsigned rowA, unsigned colA, unsigned colB>
static inline void mm_4x2(const int8 *__restrict pA, const int8 *__restrict pB,
                          int32 *__restrict pC) {
  using MMUL = aie::mmul<4, 8, 8, int8, int8, accauto>;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (unsigned z = 0; z < rowA; z += 4) {
    int32 *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
    int32 *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;
    int32 *__restrict pC3 = pC + ((z + 2) * colB) * MMUL::size_C;
    int32 *__restrict pC4 = pC + ((z + 3) * colB) * MMUL::size_C;
    for (unsigned j = 0; j < colB; j += 2) {
      const int8 *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
      const int8 *__restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
      const int8 *__restrict pA3 = pA + ((z + 2) * colA) * MMUL::size_A;
      const int8 *__restrict pA4 = pA + ((z + 3) * colA) * MMUL::size_A;
      const int8 *__restrict pB1 = pB + (j * colA) * MMUL::size_B;
      const int8 *__restrict pB2 = pB + ((j + 1) * colA) * MMUL::size_B;
      MMUL C00(aie::load_v<MMUL::size_C>(pC1));
      MMUL C01(aie::load_v<MMUL::size_C>(pC1 + MMUL::size_C));
      MMUL C10(aie::load_v<MMUL::size_C>(pC2));
      MMUL C11(aie::load_v<MMUL::size_C>(pC2 + MMUL::size_C));
      MMUL C20(aie::load_v<MMUL::size_C>(pC3));
      MMUL C21(aie::load_v<MMUL::size_C>(pC3 + MMUL::size_C));
      MMUL C30(aie::load_v<MMUL::size_C>(pC4));
      MMUL C31(aie::load_v<MMUL::size_C>(pC4 + MMUL::size_C));
      for (unsigned i = 0; i < colA; i++) {
        auto A0 = aie::load_v<MMUL::size_A>(pA1);
        pA1 += MMUL::size_A;
        auto A1 = aie::load_v<MMUL::size_A>(pA2);
        pA2 += MMUL::size_A;
        auto A2 = aie::load_v<MMUL::size_A>(pA3);
        pA3 += MMUL::size_A;
        auto A3 = aie::load_v<MMUL::size_A>(pA4);
        pA4 += MMUL::size_A;
        auto B0 = aie::transpose(aie::load_v<MMUL::size_B>(pB1), 8, 8);
        pB1 += MMUL::size_B;
        auto B1 = aie::transpose(aie::load_v<MMUL::size_B>(pB2), 8, 8);
        pB2 += MMUL::size_B;
        C00.mac(A0, B0);
        C01.mac(A0, B1);
        C10.mac(A1, B0);
        C11.mac(A1, B1);
        C20.mac(A2, B0);
        C21.mac(A2, B1);
        C30.mac(A3, B0);
        C31.mac(A3, B1);
      }
      aie::store_v(pC1, C00.template to_vector<int32>());
      pC1 += MMUL::size_C;
      aie::store_v(pC1, C01.template to_vector<int32>());
      pC1 += MMUL::size_C;
      aie::store_v(pC2, C10.template to_vector<int32>());
      pC2 += MMUL::size_C;
      aie::store_v(pC2, C11.template to_vector<int32>());
      pC2 += MMUL::size_C;
      aie::store_v(pC3, C20.template to_vector<int32>());
      pC3 += MMUL::size_C;
      aie::store_v(pC3, C21.template to_vector<int32>());
      pC3 += MMUL::size_C;
      aie::store_v(pC4, C30.template to_vector<int32>());
      pC4 += MMUL::size_C;
      aie::store_v(pC4, C31.template to_vector<int32>());
      pC4 += MMUL::size_C;
    }
  }
}
""" + STEP_AND_CALL.replace("SYMBOL", "mm_ptr_step_issue").replace(
    "MATMUL", "mm_4x2<4, 2, 2>"
)


@iron.jit
def _design(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    symbol: CompileTime[str],
    source: CompileTime[str],
):
    n = PRODUCTS * DIM * DIM
    ab_ty = np.ndarray[(n,), np.dtype[np.int8]]
    c_ty = np.ndarray[(n,), np.dtype[np.int32]]
    kernel = ExternalFunction(
        symbol,
        source_string=source,
        arg_types=[ab_ty, ab_ty, c_ty],
        include_dirs=[aie_kernels_dir()],
    )
    of_a = ObjectFifo(ab_ty, name="a")
    of_b = ObjectFifo(ab_ty, name="b")
    of_c = ObjectFifo(c_ty, name="c")

    def core(of_a, of_b, of_c, kernel):
        a, b, c = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
        kernel(a, b, c)
        of_a.release(1)
        of_b.release(1)
        of_c.release(1)

    worker = Worker(core, [of_a.cons(), of_b.cons(), of_c.prod(), kernel])

    def seq(a, b, c, h_a, h_b, h_c):
        h_a.fill(a)
        h_b.fill(b)
        h_c.drain(c, wait=True)

    rt = Runtime(seq, [ab_ty, ab_ty, c_ty, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "symbol, source",
    [("mm_ptr_step_shipped", SHIPPED), ("mm_ptr_step_issue", ISSUE)],
    ids=["shipped", "issue"],
)
def test_mm_ptr_step(symbol, source, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, (PRODUCTS, DIM, DIM), dtype=np.int8)
    b = rng.integers(-128, 128, (PRODUCTS, DIM, DIM), dtype=np.int8)
    ref = a.astype(np.int32) @ b.astype(np.int32)

    # A and C in row-major (r, s) and (r, t) tiles; B as (t, s) tiles of B^T,
    # a column of K tiles at a time.
    a_tiles = a.reshape(PRODUCTS, DIM // R, R, DIM // S, S).transpose(0, 1, 3, 2, 4)
    b_tiles = b.reshape(PRODUCTS, DIM // S, S, DIM // T, T).transpose(0, 3, 1, 4, 2)
    n = PRODUCTS * DIM * DIM
    a_dev = iron.tensor(np.ascontiguousarray(a_tiles).reshape(n), device="npu")
    b_dev = iron.tensor(np.ascontiguousarray(b_tiles).reshape(n), device="npu")
    c_dev = iron.zeros((n,), dtype=np.int32, device="npu")

    _design(a_dev, b_dev, c_dev, symbol=symbol, source=source)

    c = (
        c_dev.numpy()
        .reshape(PRODUCTS, DIM // R, DIM // T, R, T)
        .transpose(0, 1, 3, 2, 4)
        .reshape(PRODUCTS, DIM, DIM)
    )
    np.testing.assert_array_equal(c, ref)
