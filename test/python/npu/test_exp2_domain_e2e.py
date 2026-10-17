# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""exp2_nonpos_bf16 against exp2_bf16 at every float32 the softmax gives it.

The online softmax only takes 2^x of x <= 0: -0, every negative finite
float32 and -inf, the bit patterns 0x80000000 to 0xff800000. One core walks
all 2139095041 of them, 64 a step, in ``_DISPATCHES`` calls, comparing
exp2_nonpos_bf16's bf16 outputs, at 32 lanes and on AIE2P at 64, with
exp2_bf16's bit for bit under -DEXP2_BF16_ACCURATE.
"""

import numpy as np

import aie.iron as iron
from aie.iron import (
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
)
from aie.utils.config import aie_kernels_dir

_DISPATCHES = 16
_STEPS = (1 << 31) // 64 // _DISPATCHES

_SOURCE = r"""
#include "common/exp2_bf16.h"

alignas(aie::vector_decl_align) static const int32_t lane[32] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

// How many x of the 64 * steps patterns from base[0] that are <= 0 or -inf
// take a 2^x from exp2_nonpos_bf16, at 32 lanes or (AIE2P) 64, other than
// exp2_bf16's; into count[0].
extern "C" void exp2_domain_mismatches(int32_t *base, int32_t *count,
                                       int32_t steps) {
  aie::rounding_mode saved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const auto lanes = aie::load_v<32>(lane).cast_to<uint32_t>();
  const auto top = aie::broadcast<uint32_t, 32>(0xff800000u);
  const auto ones = aie::broadcast<int32_t, 32>(1);
  aie::vector<int32_t, 32> bad = aie::zeros<int32_t, 32>();
  uint32_t first = (uint32_t)base[0];
  for (int32_t i = 0; i < steps; i++) {
    const auto lo = aie::add(lanes, aie::broadcast<uint32_t, 32>(first));
    const auto hi = aie::add(lanes, aie::broadcast<uint32_t, 32>(first + 32));
#if AIE_ARCH_AIE2P
    const auto now = exp2_nonpos_bf16<64>(aie::concat(lo.cast_to<float>(),
                                                      hi.cast_to<float>()))
                         .cast_to<uint16_t>();
#endif
    for (unsigned h = 0; h < 2; h++) {
      const auto bits = h ? hi : lo;
      const auto half = bits.cast_to<float>();
      const auto was = exp2_bf16_32(half).cast_to<uint16_t>();
      auto differ = aie::neq(was, exp2_nonpos_bf16<32>(half).cast_to<uint16_t>());
#if AIE_ARCH_AIE2P
      differ = differ | aie::neq(was, now.extract<32>(h));
#endif
      bad = aie::add(bad, aie::select(aie::zeros<int32_t, 32>(), ones,
                                      differ & aie::le(bits, top)));
    }
    first += 64;
  }
  count[0] = aie::reduce_add(bad);
  aie::set_rounding(saved);
}
"""


@jit
def exp2_domain(base_in: In, count_out: Out):
    """One call of ``exp2_domain_mismatches`` over ``_STEPS`` steps."""
    word_ty = np.ndarray[(16,), np.dtype[np.int32]]
    check = ExternalFunction(
        "exp2_domain_mismatches",
        source_string=_SOURCE,
        arg_types=[word_ty, word_ty, np.int32],
        include_dirs=[aie_kernels_dir()],
        compile_flags=["-DEXP2_BF16_ACCURATE=1"],
    )
    of_base = ObjectFifo(word_ty, name="base", depth=1)
    of_count = ObjectFifo(word_ty, name="count", depth=1)

    def core(of_base, of_count, check):
        check(of_base.acquire(1), of_count.acquire(1), _STEPS)
        of_base.release(1)
        of_count.release(1)

    worker = Worker(core, fn_args=[of_base.cons(), of_count.prod(), check])

    def sequence(base_h, count_h, bf, cf):
        bf.fill(base_h)
        cf.drain(count_h, wait=True)

    rt = Runtime(sequence, [word_ty, word_ty, of_base.prod(), of_count.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_exp2_nonpos_matches_exp2_on_the_whole_softmax_domain():
    mismatches = []
    for d in range(_DISPATCHES):
        base = np.zeros(16, np.int32)
        base[0] = np.uint32(0x80000000 + d * _STEPS * 64).view(np.int32)
        count = iron.zeros((16,), dtype=np.int32)
        exp2_domain(iron.tensor(base, dtype=np.int32), count)
        mismatches.append(int(count.numpy()[0]))
    assert mismatches == [0] * _DISPATCHES, mismatches
