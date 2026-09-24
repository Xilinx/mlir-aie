//===- mm.cc ----------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// bfp16ebs8 stores 8 mantissa bytes plus one shared exponent byte per 8
// elements, so one block is 9 bytes and the mmul sub-tile the shuffle makes
// contiguous is 8 rows of one block.
static constexpr size_t kBlockBytes = 9;
static constexpr size_t kSubtileRows = 8;

namespace {

// Both directions below move a run of blocks whose destination is contiguous
// and whose source is strided, and both visit the destination in strictly
// increasing address order. That is what lets a block be stored as part of an
// unaligned vector: the bytes the store writes beyond the block are rewritten
// by the store that follows it. Only the last block of the buffer has no
// successor to repair it, so the callers finish on copyRunsAtEnd.
//
// 9 bytes is neither a power of two nor vector aligned, so a byte at a time the
// target serializes every lda.s8 against its st.s8 at the full load-to-use
// latency: 94 cycles for one block. Merging three blocks into a single
// 32-byte store also amortizes the read-modify-write an unaligned store costs,
// which is what the per-block dependence was really paying for: 20 cycles for
// three blocks.

using bytes32 = aie::vector<uint8_t, 32>;

constexpr aie::mask<32> blockMask(unsigned index) {
  return aie::mask<32>::from_uint32(((1u << kBlockBytes) - 1)
                                    << (index * kBlockBytes));
}

inline bytes32 blockAt(const uint8_t *src, unsigned index) {
  return aie::shuffle_up(aie::load_unaligned_v<32>(src), index * kBlockBytes);
}

// 27 of the 32 bytes stored are wanted.
inline void storeThree(const uint8_t *src, size_t srcStride, uint8_t *dst) {
  bytes32 v = aie::load_unaligned_v<32>(src);
  v = aie::select(v, blockAt(src + srcStride, 1), blockMask(1));
  v = aie::select(v, blockAt(src + 2 * srcStride, 2), blockMask(2));
  aie::store_unaligned_v(dst, v);
}

// 18 of 32.
inline void storeTwo(const uint8_t *src, size_t srcStride, uint8_t *dst) {
  bytes32 v = aie::load_unaligned_v<32>(src);
  v = aie::select(v, blockAt(src + srcStride, 1), blockMask(1));
  aie::store_unaligned_v(dst, v);
}

// 9 of 16.
inline void storeOne(const uint8_t *src, uint8_t *dst) {
  aie::store_unaligned_v(dst, aie::load_unaligned_v<16>(src));
}

inline void storeOneExact(const uint8_t *src, uint8_t *dst) {
  for (size_t j = 0; j < kBlockBytes; ++j)
    dst[j] = src[j];
}

// n / 3 for the block counts in play. Spelled as a division the target wants
// the high half of a 32-bit product, which is a __muldi3 call in the middle of
// the loop's trip count; these widths fit in 16 bits, so the reciprocal
// multiply fits in a native 32-bit product.
inline size_t divideByThree(size_t n) { return (n * 0xAAABu) >> 17; }

// `groups` merged triples followed by `tail` (0, 1 or 2) leftover blocks:
// `3 * groups + tail` blocks `srcStride` apart into that many contiguous blocks
// at `dst`.
inline void copyRun(const uint8_t *__restrict src, size_t srcStride,
                    uint8_t *__restrict dst, size_t groups, size_t tail) {
  for (size_t g = 0; g < groups; ++g) {
    storeThree(src, srcStride, dst);
    src += 3 * srcStride;
    dst += 3 * kBlockBytes;
  }
  if (tail == 2)
    storeTwo(src, srcStride, dst);
  else if (tail == 1)
    storeOne(src, dst);
}

// The same run, but ending at the end of the destination buffer: `groups`
// triples then `singles` single blocks leave exactly one block, which is copied
// at its own width so that nothing is written past `dst`.
inline void copyRunAtEnd(const uint8_t *__restrict src, size_t srcStride,
                         uint8_t *__restrict dst, size_t groups,
                         size_t singles) {
  for (size_t g = 0; g < groups; ++g) {
    storeThree(src, srcStride, dst);
    src += 3 * srcStride;
    dst += 3 * kBlockBytes;
  }
  for (size_t s = 0; s < singles; ++s) {
    storeOne(src, dst);
    src += srcStride;
    dst += kBlockBytes;
  }
  storeOneExact(src, dst);
}

} // namespace

// There is a CPU version of this function in the helper.h file.
// Internal linkage lets MATMUL_ONLY and SHUFFLE_ONLY objects coexist.
//
// The blocked side is written straight through, a sub-tile at a time, while the
// plain side is gathered one column of blocks at a time.
[[maybe_unused]] static void shuffleBfp16ebs8(size_t blocksPerRow,
                                              size_t tileHeight,
                                              const uint8_t *__restrict in,
                                              uint8_t *__restrict out) {
  const size_t rowBytes = blocksPerRow * kBlockBytes;
  const size_t lastX = rowBytes - kBlockBytes;
  constexpr size_t subtileBytes = kSubtileRows * kBlockBytes;

  uint8_t *dst = out;
  for (size_t sy = 0; sy < tileHeight; sy += kSubtileRows) {
    const uint8_t *rowBase = in + sy * rowBytes;
    // Stop one sub-tile short of the end; the peel below finishes it.
    const size_t xEnd = sy + kSubtileRows < tileHeight ? rowBytes : lastX;
    for (size_t sx = 0; sx < xEnd; sx += kBlockBytes) {
      // A sub-tile is 8 blocks: two triples and a pair.
      copyRun(rowBase + sx, rowBytes, dst, 2, 2);
      dst += subtileBytes;
    }
  }
  copyRunAtEnd(in + (tileHeight - kSubtileRows) * rowBytes + lastX, rowBytes,
               dst, 2, 1);
}

// The inverse. Here the plain side is the one written, so its rows move outside
// the sub-tile loop -- a row is contiguous and the sub-tiles feeding it are
// what becomes strided -- and the rows are still visited in order.
[[maybe_unused]] static void unshuffleBfp16ebs8(size_t blocksPerRow,
                                                size_t tileHeight,
                                                const uint8_t *__restrict in,
                                                uint8_t *__restrict out) {
  constexpr size_t subtileBytes = kSubtileRows * kBlockBytes;
  const size_t rowBytes = blocksPerRow * kBlockBytes;

  // A row is as long as the tile is wide, so the split into triples is the same
  // for every row and is worth finding once.
  const size_t groups = divideByThree(blocksPerRow);
  const size_t tail = blocksPerRow - 3 * groups;
  // The last row has to leave a block over for the exact copy.
  const size_t endGroups = tail ? groups : groups - 1;
  const size_t endSingles = blocksPerRow - 3 * endGroups - 1;

  for (size_t sy = 0; sy < tileHeight; sy += kSubtileRows) {
    const uint8_t *blockBase = in + sy * rowBytes;
    const size_t rows =
        sy + kSubtileRows < tileHeight ? kSubtileRows : kSubtileRows - 1;
    for (size_t i = 0; i < rows; ++i)
      copyRun(blockBase + i * kBlockBytes, subtileBytes,
              out + (sy + i) * rowBytes, groups, tail);
  }
  copyRunAtEnd(in + (tileHeight - kSubtileRows) * rowBytes +
                   (kSubtileRows - 1) * kBlockBytes,
               subtileBytes, out + (tileHeight - 1) * rowBytes, endGroups,
               endSingles);
}

// This kernel mirrors the one found in
// https://xilinx.github.io/aie_api/group__group__mmul.html Go through them in
// parallel to understand how the bfp datatype modifies accesses to memory Note
// that this kernel assumes that the B matrix is already transposed, which is
// not the case for the example in the link. The equivalent transformations for
// a non transposed B matrix are commented out below. Also note that assuming
// the 8x8 tile are already transposed (the ones done during the shuffle), the
// higher level tiling transposition should be free using data layout
// transformations.
template <unsigned rowA, unsigned colA, unsigned colB, unsigned r, unsigned s,
          unsigned t>
void matmul_vectorized_2x2_bfp16(const bfp16ebs8 *__restrict pA,
                                 const bfp16ebs8 *__restrict pB,
                                 bfp16ebs8 *__restrict pC) {
  const unsigned sizeA = r * s;
  const unsigned sizeB = s * t;
  const unsigned sizeC = r * t;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(4)
  for (unsigned z = 0; z < rowA; z += 2) {
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC1In(pC);
    pC1In.seek(z * colB);
    aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pC2In(pC);
    pC2In.seek((z + 1) * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC1Out(pC);
    pC1Out.seek(z * colB);
    aie::block_vector_output_buffer_stream<bfp16ebs8, 64> pC2Out(pC);
    pC2Out.seek((z + 1) * colB);

    for (unsigned j = 0; j < colB; j += 2)
#ifdef OPT_PERF_ENABLED
      AIE_LOOP_FLATTEN
#endif
      {
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA1bfp16(pA);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pA2bfp16(pA);
        pA1bfp16.seek(z * colA);
        pA2bfp16.seek((z + 1) * colA);

        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB1bfp16(pB);
        aie::block_vector_input_buffer_stream<bfp16ebs8, 64> pB2bfp16(pB);
        // For non transposed matrix
        // pB1bfp16.seek(j);
        // pB2bfp16.seek(j + 1);
        pB1bfp16.seek(j * colA);
        pB2bfp16.seek((j + 1) * colA);

        aie::block_vector<bfp16ebs8, sizeA> A0;
        aie::block_vector<bfp16ebs8, sizeA> A1;
        aie::block_vector<bfp16ebs8, sizeB> B0;
        aie::block_vector<bfp16ebs8, sizeB> B1;

        // Note that unlike the example mentioned above, we need
        // to use a mac to take into account results from previous kernel
        // calls but this is completely unrelated to the block datatype.
        aie::accum<accfloat, sizeC> accC00(pC1In.pop());
        aie::accum<accfloat, sizeC> accC01(pC1In.pop());
        aie::accum<accfloat, sizeC> accC10(pC2In.pop());
        aie::accum<accfloat, sizeC> accC11(pC2In.pop());

        for (unsigned i = 0; i < colA; ++i)
#ifdef OPT_PERF_ENABLED
          AIE_LOOP_FLATTEN
#endif
          {
            A0 = pA1bfp16.pop();
            A1 = pA2bfp16.pop();

            // For non transposed matrix
            // B0 = pB1bfp16.pop_seek(colB - 1);
            // B1 = pB2bfp16.pop_seek(colB - 1);
            B0 = pB1bfp16.pop();
            B1 = pB2bfp16.pop();

            accC00 = mac_8x8_8x8T(A0, B0, accC00);
            accC01 = mac_8x8_8x8T(A0, B1, accC01);
            accC10 = mac_8x8_8x8T(A1, B0, accC10);
            accC11 = mac_8x8_8x8T(A1, B1, accC11);
          }

        pC1Out.push(accC00.template to_vector<bfp16ebs8>());
        pC1Out.push(accC01.template to_vector<bfp16ebs8>());
        pC2Out.push(accC10.template to_vector<bfp16ebs8>());
        pC2Out.push(accC11.template to_vector<bfp16ebs8>());
      }
  }
}

extern "C" {

#ifndef DIM_M
#define DIM_M 64
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

#ifndef DIM_N
#define DIM_N 64
#endif

// MATMUL_ONLY / SHUFFLE_ONLY let callers (e.g. @iron.jit
// ExternalFunction) compile a .o containing exactly one of the two
// entry points, avoiding duplicate-symbol errors when the same .cc is
// compiled multiple times for distinct ExternalFunctions in one design.
// Without any macro, both symbols are emitted.
#if !defined(MATMUL_ONLY) && !defined(SHUFFLE_ONLY)
#define MATMUL_ONLY
#define SHUFFLE_ONLY
#endif

#ifdef MATMUL_ONLY
void matmul_vectorized_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB,
                             bfp16ebs8 *__restrict pC) {
  event0();

  constexpr int r = 8;
  constexpr int s = 8;
  constexpr int t = 8;

  constexpr int m = DIM_M;
  constexpr int k = DIM_K;
  constexpr int n = DIM_N;

  static_assert(m % (2 * r) == 0);
  static_assert(k % s == 0);
  static_assert(n % (2 * t) == 0);

  matmul_vectorized_2x2_bfp16<m / r, k / s, n / t, r, s, t>(pA, pB, pC);
  event1();
}
#endif

#ifdef SHUFFLE_ONLY
void scalar_shuffle(uint8_t *pA, uint8_t *pC, size_t tileWidth,
                    size_t tileHeight, bool unshuffle = false) {
  event0();
  // A row is 9/8 bytes per element. Spelling that as *1.125 round-trips the
  // size_t through double, which on aie2p is three soft-float calls
  // (__floatunsidf, __muldf3, __fixunsdfsi) sitting in the address math;
  // counting blocks instead keeps it to a shift.
  const size_t blocksPerRow = tileWidth / kSubtileRows;

  // The direction is a runtime flag, but left in the innermost loop it costs
  // two sel.nez and indexed addressing on every byte. Specialized, each side
  // walks its own pointers.
  if (!unshuffle)
    shuffleBfp16ebs8(blocksPerRow, tileHeight, pA, pC);
  else
    unshuffleBfp16ebs8(blocksPerRow, tileHeight, pA, pC);
  event1();
}
#endif
}
