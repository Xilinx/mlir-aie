/*
Copyright (C) 2014-2022 Xilinx, Inc.
Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
    SPDX-License-Identifier: MIT
*/

// Lock helpers and the vector zero, copy, narrow and scale templates that the
// flm_gemma4 kernels share. Independent of the model geometry.
#ifndef AIE_KERNELS_FLM_GEMMA4_UTILS_H
#define AIE_KERNELS_FLM_GEMMA4_UTILS_H

#include "../aie_kernel_utils.h" // AIE_LOOP_*

#include <aie_api/aie.hpp>

using bf16 = bfloat16;

// The lock-id offset selects the tile: +48 this tile, +0 down, +16 left.
inline void _lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id + 48, num_locks);
}

inline void _lock_release(int lock_id, int num_locks = 1) {
  release(lock_id + 48, num_locks);
}

// Pointer forms. Peano's aie2p_locks.h provides acquire/release overloads that
// take the address of the guarded buffer and tie the lock to that buffer. The
// 2-argument forms carry no memory operand, so the optimizer may move guarded
// accesses across the lock. Use the pointer forms wherever a lock guards a
// buffer.
inline void _lock_acquire_p(const void *a, int lock_id, int num_locks = 1) {
  acquire_greater_equal(a, lock_id + 48, num_locks);
}
inline void _lock_release_p(void *a, int lock_id, int num_locks = 1) {
  release(a, lock_id + 48, num_locks);
}

// The parity of a ping-pong buffer pair. It must persist across calls, so keep
// the object static or at file scope, and never reset it.
struct PingPong {
  bool is_ping = false;

  // Flip the parity and return the buffer it selects.
  template <typename T>
  T *next(T *ping, T *pong) {
    is_ping = !is_ping;
    return is_ping ? ping : pong;
  }

  // next(), then acquire the selected buffer's lock.
  template <typename T>
  T *acquire(T *ping, T *pong, int lock_id) {
    T *buf = next(ping, pong);
    _lock_acquire_p(buf, lock_id);
    return buf;
  }
};

inline void _down_lock_acquire(int lock_id, int num_locks = 1) {
  acquire_greater_equal(lock_id, num_locks);
}

// Pointer forms for the locks of the neighbour tiles.
inline void _down_lock_acquire_p(const void *a, int lock_id,
                                 int num_locks = 1) {
  acquire_greater_equal(a, lock_id, num_locks);
}
inline void _down_lock_release_p(void *a, int lock_id, int num_locks = 1) {
  release(a, lock_id, num_locks);
}
inline void _left_lock_acquire_p(const void *a, int lock_id,
                                 int num_locks = 1) {
  acquire_greater_equal(a, lock_id + 16, num_locks);
}
inline void _left_lock_release_p(void *a, int lock_id, int num_locks = 1) {
  release(a, lock_id + 16, num_locks);
}

// Stores 256 bits at a time, so c needs only 32-byte alignment. A wider store
// to a buffer aligned below its width drops part of the store on AIE2P.
template <typename T, int M>
void zero_256(T *__restrict c) {
  constexpr int r = 256 / (sizeof(T) * 8);
  static_assert((M) % r == 0);
  const aie::vector<T, r> zeros = aie::zeros<T, r>();
  const T *__restrict c_end = c + M;
  for (; c < c_end; c += r) {
    aie::store_v(c, zeros);
  }
}

// Copy M elements with vector loads and stores. Do not use memcpy in a loop:
// memcpy is an opaque call, and the pipeliner does not schedule a loop that
// contains a call.
//
// Both pointers must be 32-byte aligned. The copy uses 256-bit loads and
// stores, and AIE2P silently drops part of an under-aligned access. The
// static_assert checks only the element count. IRON buffers are 32-byte
// aligned by default, so an offset into a buffer must be a multiple of 32
// bytes.
template <typename T, int M>
void copy_vectorized(T *__restrict dst, const T *__restrict src) {
  constexpr int r = 256 / (sizeof(T) * 8); // one 256-bit store unit
  static_assert((M) % r == 0);
  const T *__restrict src_end = src + M;
  for (; src < src_end; src += r, dst += r) {
    aie::store_v(dst, aie::load_v<r>(src));
  }
}

template <int M>
void scale_vectorized(bf16 *y, const bf16 scale) {
  for (int i = 0; i < M / 16; i++) {
    aie::vector<bf16, 16> y_vec = aie::load_v<16>(y);
    aie::vector<bf16, 16> out_vec =
        aie::mul(y_vec, aie::broadcast<bf16, 16>(scale));
    aie::store_v(y, out_vec);
    y += 16;
  }
}

// Narrows N floats to bf16 through an accfloat accumulator. The attention
// kernels narrow in place, with the same buffer as source and destination: the
// bf16 pointer advances at half the byte rate of the float one and so trails
// it. The pointers must not be `restrict`, because they may alias.
template <int N>
void narrow_to_bf16(bf16 *out, const float *in) {
  constexpr int vec_factor = 16;
  static_assert(N % vec_factor == 0);
  for (int i = 0; i < N / vec_factor; i++) {
    aie::accum<accfloat, vec_factor> acc;
    acc.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, acc.template to_vector<bf16>());
    in += vec_factor;
    out += vec_factor;
  }
}

// out[l] = the sum of x[32 l .. 32 l + 31], for the N / 32 groups of x: the
// column sums that fold a q4nx block's minima into one MAC per group.
template <int N>
void group_sums_32(bf16 *out, const bf16 *x) {
  AIE_LOOP_UNROLL_FULL
  for (int l = 0; l < N / 32; l++) {
    out[l] = bf16(aie::reduce_add(aie::load_v<32>(x + 32 * l)));
  }
}

#endif // AIE_KERNELS_FLM_GEMMA4_UTILS_H
