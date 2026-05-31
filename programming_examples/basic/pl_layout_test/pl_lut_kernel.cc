// PL layout discovery test kernel.
// Builds an int16 LUT with sentinel values (lut[k] = k - 128) and uses
// aie::parallel_lookup<int8, lut<4, int16, int16>, truncate> with bias=128.
// If layout is correct, output[i] should equal (int8_t)input[i] for all i.
// Tests several candidate layouts via -DLAYOUT_VARIANT=N.

#include <aie_api/aie.hpp>
#include <stdint.h>

#if LAYOUT_VARIANT == 13
// File-scope MUTABLE LUTs, filled at runtime in kernel entry. Simulates the
// yolo silu_lut runtime-loading case while keeping linker-placed storage.
namespace {
alignas(64) int16_t RUNTIME_LUT_AB[512];
alignas(64) int16_t RUNTIME_LUT_CD[512];
} // namespace
#endif

#if LAYOUT_VARIANT == 8 || LAYOUT_VARIANT == 9 || LAYOUT_VARIANT == 11
// File-scope const LUTs (.rodata, linker-placed) for alignment.
namespace {
struct FlatLUT {
  int16_t data[512];
};
constexpr FlatLUT make_flat_lut_v8() {
  FlatLUT l{};
  for (int k = 0; k < 256; ++k) {
    l.data[2 * k] = (int16_t)(k - 128);
    l.data[2 * k + 1] = 0;
  }
  return l;
}
constexpr FlatLUT make_flat_lut_v9() {
  FlatLUT l{};
  for (int k = 0; k < 256; ++k)
    l.data[k] = (int16_t)(k - 128);
  for (int k = 256; k < 512; ++k)
    l.data[k] = 0;
  return l;
}
constexpr FlatLUT make_flat_lut_v11() {
  // 8-grouped bank-duplicated: chunks of 8 entries appear twice in a row.
  // pl[i*2 + b] = pl[i*2 + 8 + b] = sentinel[i+b], for i in 0..255 step 8.
  FlatLUT l{};
  for (int i = 0; i < 256; i += 8) {
    for (int b = 0; b < 8; ++b) {
      int16_t v = (int16_t)((i + b) - 128);
      l.data[i * 2 + b] = v;
      l.data[i * 2 + 8 + b] = v;
    }
  }
  return l;
}
#if LAYOUT_VARIANT == 8
alignas(64) constexpr FlatLUT FLAT_LUT_AB = make_flat_lut_v8();
alignas(64) constexpr FlatLUT FLAT_LUT_CD = make_flat_lut_v8();
#elif LAYOUT_VARIANT == 9
alignas(64) constexpr FlatLUT FLAT_LUT_AB = make_flat_lut_v9();
alignas(64) constexpr FlatLUT FLAT_LUT_CD = make_flat_lut_v9();
#else
alignas(64) constexpr FlatLUT FLAT_LUT_AB = make_flat_lut_v11();
alignas(64) constexpr FlatLUT FLAT_LUT_CD = make_flat_lut_v11();
#endif
} // namespace
#endif

#if LAYOUT_VARIANT == 15
// LAYOUT 15 — int8 LUT, 2-byte stride hypothesis.
// Observed in v14: idx-stride is 2 bytes (not the nominal 4). Pack picks the
// low byte of an int16 word. So layout each entry as (sentinel, 0) over
// 2 bytes; AB and CD identical.
namespace {
struct FlatLUT8_v15 {
  int8_t data[512]; // 256 entries × 2 bytes
};
constexpr FlatLUT8_v15 make_flat_lut_v15() {
  FlatLUT8_v15 l{};
  for (int k = 0; k < 256; ++k) {
    l.data[k * 2 + 0] = (int8_t)(k - 128);
    l.data[k * 2 + 1] = 0;
  }
  return l;
}
alignas(64) constexpr FlatLUT8_v15 FLAT_LUT8_AB = make_flat_lut_v15();
alignas(64) constexpr FlatLUT8_v15 FLAT_LUT8_CD = make_flat_lut_v15();
} // namespace
#endif

#if LAYOUT_VARIANT == 14
// LAYOUT 14 — int8 LUT path, not previously tested.
// Use aie::lut<4, int8, int8>: 4 bytes per entry, AB & CD halves. The
// parallel_lookup fetch path packs the int16 coeff back to int8 at the end
// (ValueWords==1 branch in detail/aie2/parallel_lookup.hpp line 211), so
// we get 32-wide int8→int8 lookups per fetch.
//
// First-pass layout: each 4-byte entry filled (sentinel, 0, 0, 0); AB and
// CD identical. Sentinel: lut[k] = k - 128 (identity over int8 input).
// If output != input we iterate on layout based on the failure pattern.
namespace {
struct FlatLUT8 {
  int8_t data[1024]; // 256 entries × 4 bytes
};
constexpr FlatLUT8 make_flat_lut_v14() {
  FlatLUT8 l{};
  for (int k = 0; k < 256; ++k) {
    int8_t v = (int8_t)(k - 128);
    l.data[k * 4 + 0] = v;
    l.data[k * 4 + 1] = 0;
    l.data[k * 4 + 2] = 0;
    l.data[k * 4 + 3] = 0;
  }
  return l;
}
alignas(64) constexpr FlatLUT8 FLAT_LUT8_AB = make_flat_lut_v14();
alignas(64) constexpr FlatLUT8 FLAT_LUT8_CD = make_flat_lut_v14();
} // namespace
#endif

extern "C" {

#if LAYOUT_VARIANT == 14 || LAYOUT_VARIANT == 15
void pl_lookup(int8_t *in, int8_t *out, int32_t n_bytes) {
  using lut_t = aie::lut<4, int8, int8>;
  int8_t __aie_dm_resource_a *p_ab =
      (int8_t __aie_dm_resource_a *)FLAT_LUT8_AB.data;
  int8_t __aie_dm_resource_b *p_cd =
      (int8_t __aie_dm_resource_b *)FLAT_LUT8_CD.data;
  lut_t my_lut(256, (const void *)p_ab, (const void *)p_cd);
#ifndef STEP_BITS
#define STEP_BITS 2 // linear_approx int8 sets min_step_bits = 2
#endif
  aie::parallel_lookup<uint8, lut_t, aie::lut_oor_policy::truncate> lookup(
      my_lut, STEP_BITS);

  for (int i = 0; i < n_bytes; i += 32) {
    aie::vector<int8, 32> v_in = aie::load_v<32>((int8_t *)(in + i));
    aie::vector<uint8, 32> v_idx =
        aie::add(v_in.cast_to<uint8>(), aie::broadcast<uint8, 32>(128));
    aie::vector<int8, 32> v_res = lookup.fetch(v_idx);
    aie::store_v((int8_t *)(out + i), v_res);
  }
}
#else
void pl_lookup(int8_t *in, int8_t *out, int32_t n_bytes) {
  // Build LUT in stack (256 sentinel int16 values, with chosen layout).
  // 256 entries × 4 bytes/entry (offset+slope pair) × 2 copies (bank-dup) =
  // 2048
  struct LUT {
    alignas(aie::vector_decl_align) int16_t ab[2048];
    alignas(aie::vector_decl_align) int16_t cd[2048];
  } lut;

  // Sentinel data: silu_lut[k] = (k - 128) as int8, sign-extended to int16.
  int8_t sentinel[256];
  for (int k = 0; k < 256; ++k)
    sentinel[k] = (int8_t)(k - 128);

#if LAYOUT_VARIANT == 1
  // Layout 1: entry-interleaved. pl.ab[2k] = pl.ab[2k+1] = sentinel[k].
  for (int k = 0; k < 256; ++k) {
    int16_t v = (int16_t)sentinel[k];
    lut.ab[2 * k] = v;
    lut.ab[2 * k + 1] = v;
    lut.cd[2 * k] = v;
    lut.cd[2 * k + 1] = v;
  }
#elif LAYOUT_VARIANT == 2
  // Layout 2: 8-grouped bank-duplicated (matches rgba2hue's lut_inv_8b.h).
  for (int i = 0; i < 256; i += 8) {
    for (int b = 0; b < 8; ++b) {
      int16_t v = (int16_t)sentinel[i + b];
      lut.ab[i * 2 + b] = v;
      lut.ab[i * 2 + 8 + b] = v;
      lut.cd[i * 2 + b] = v;
      lut.cd[i * 2 + 8 + b] = v;
    }
  }
#elif LAYOUT_VARIANT == 3
  // Layout 3: flat one-per-slot, no duplication. ab[k] = sentinel[k].
  for (int k = 0; k < 256; ++k) {
    int16_t v = (int16_t)sentinel[k];
    lut.ab[k] = v;
    lut.cd[k] = v;
  }
  // Zero out the trailing half so reads past the end are deterministic.
  for (int k = 256; k < 512; ++k) {
    lut.ab[k] = 0;
    lut.cd[k] = 0;
  }
#elif LAYOUT_VARIANT == 4
  // Layout 4: whole-LUT duplicated as [LUT | LUT].
  for (int k = 0; k < 256; ++k) {
    int16_t v = (int16_t)sentinel[k];
    lut.ab[k] = v;
    lut.ab[256 + k] = v;
    lut.cd[k] = v;
    lut.cd[256 + k] = v;
  }
#elif LAYOUT_VARIANT == 5
  // Layout 5: bank-interleave at 4-element granularity (try smaller
  // bank-width).
  for (int i = 0; i < 256; i += 4) {
    for (int b = 0; b < 4; ++b) {
      int16_t v = (int16_t)sentinel[i + b];
      lut.ab[i * 2 + b] = v;
      lut.ab[i * 2 + 4 + b] = v;
      lut.cd[i * 2 + b] = v;
      lut.cd[i * 2 + 4 + b] = v;
    }
  }
#elif LAYOUT_VARIANT == 6
  // Layout 6: layout 2 + zero-init trailing region.
  for (int k = 0; k < 2048; ++k) {
    lut.ab[k] = 0;
    lut.cd[k] = 0;
  }
  for (int i = 0; i < 256; i += 8) {
    for (int b = 0; b < 8; ++b) {
      int16_t v = (int16_t)sentinel[i + b];
      lut.ab[i * 2 + b] = v;
      lut.ab[i * 2 + 8 + b] = v;
      lut.cd[i * 2 + b] = v;
      lut.cd[i * 2 + 8 + b] = v;
    }
  }
#elif LAYOUT_VARIANT == 8 || LAYOUT_VARIANT == 9 || LAYOUT_VARIANT == 11
  (void)sentinel;
#elif LAYOUT_VARIANT == 13
  // Runtime-fill the file-scope buffers (flat layout, equivalent to v9).
  for (int k = 0; k < 256; ++k) {
    RUNTIME_LUT_AB[k] = (int16_t)(k - 128);
    RUNTIME_LUT_CD[k] = (int16_t)(k - 128);
  }
  for (int k = 256; k < 512; ++k) {
    RUNTIME_LUT_AB[k] = 0;
    RUNTIME_LUT_CD[k] = 0;
  }
#elif LAYOUT_VARIANT == 7
  // Layout 7: (offset, slope=0) pair per entry, bank-duplicated every 4
  // entries. Bank width 128b = 16 bytes = 4 entries × 4 bytes/entry. Duplicate
  // each 4-entry chunk (= 8 int16) so 256 entries occupy 2048 int16 = 4096
  // bytes.
  for (int k = 0; k < 2048; ++k) {
    lut.ab[k] = 0;
    lut.cd[k] = 0;
  }
  for (int chunk = 0; chunk < 64; ++chunk) { // 64 chunks of 4 entries
    int base = chunk * 16;                   // 16 int16 per duplicated chunk
    for (int e = 0; e < 4; ++e) {
      int k = chunk * 4 + e;
      int16_t off = (int16_t)sentinel[k];
      int16_t slp = 0;
      // chunk copy 0
      lut.ab[base + e * 2 + 0] = off;
      lut.ab[base + e * 2 + 1] = slp;
      lut.cd[base + e * 2 + 0] = off;
      lut.cd[base + e * 2 + 1] = slp;
      // chunk copy 1 (offset 8 int16 = 16 bytes ahead)
      lut.ab[base + 8 + e * 2 + 0] = off;
      lut.ab[base + 8 + e * 2 + 1] = slp;
      lut.cd[base + 8 + e * 2 + 0] = off;
      lut.cd[base + 8 + e * 2 + 1] = slp;
    }
  }
#else
#error "must -DLAYOUT_VARIANT=1..7"
#endif

  using lut_t = aie::lut<4, int16, int16>;
#if LAYOUT_VARIANT == 8 || LAYOUT_VARIANT == 9 || LAYOUT_VARIANT == 11
  int16_t __aie_dm_resource_a *p_ab =
      (int16_t __aie_dm_resource_a *)FLAT_LUT_AB.data;
  int16_t __aie_dm_resource_b *p_cd =
      (int16_t __aie_dm_resource_b *)FLAT_LUT_CD.data;
  lut_t my_lut(256, (const void *)p_ab, (const void *)p_cd);
#elif LAYOUT_VARIANT == 13
  int16_t __aie_dm_resource_a *p_ab =
      (int16_t __aie_dm_resource_a *)RUNTIME_LUT_AB;
  int16_t __aie_dm_resource_b *p_cd =
      (int16_t __aie_dm_resource_b *)RUNTIME_LUT_CD;
  lut_t my_lut(256, (const void *)p_ab, (const void *)p_cd);
#else
  lut_t my_lut(256, (uint16_t *)lut.ab, (uint16_t *)lut.cd);
#endif
#ifndef STEP_BITS
#define STEP_BITS 0
#endif
  // Per lut_example.h softmax: use uint16 input type for parallel_lookup
  // (the int8 input path doesn't work the same way on AIE2P).
  aie::parallel_lookup<uint16, lut_t, aie::lut_oor_policy::truncate> lookup(
      my_lut, STEP_BITS);

  // 32-lane fetch: see if both halves return unique results.
  for (int i = 0; i < n_bytes; i += 32) {
    aie::vector<int8, 32> v_in = aie::load_v<32>((int8_t *)(in + i));
    aie::vector<int16, 32> v_in_i16 = aie::unpack(v_in);
    aie::vector<int16, 32> bias = aie::broadcast<int16, 32>((int16_t)128);
    aie::vector<int16, 32> v_idx_i16 = aie::add(v_in_i16, bias);
    aie::vector<int16, 32> v_res = lookup.fetch(v_idx_i16.cast_to<uint16>());
    for (int j = 0; j < 32; ++j)
      out[i + j] = (int8_t)v_res[j];
  }
}
#endif

} // extern "C"
