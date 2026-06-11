//===- llama_layer_pt.cc ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Phase 1.9 dataflow stubs for the full single-layer integration test.
// IRON treats each Kernel() as a fresh MLIR func decl, so each call
// site needs its own C symbol. One .cc, many shape-specific symbols.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>

// Llama 3.2 1B decode dimensions.
static constexpr int32_t kD = 2048;
static constexpr int32_t kQD = 2048;
static constexpr int32_t kKVD = 512;
static constexpr int32_t kHD = 8192;

// out[0..min(n_in, n_out)] = in[..]; rest zero.
static inline void copy_impl(const int8_t *in, int8_t *out, int32_t n_in,
                             int32_t n_out) {
  int32_t n = (n_in < n_out) ? n_in : n_out;
  for (int32_t i = 0; i < n; i++)
    out[i] = in[i];
  for (int32_t i = n; i < n_out; i++)
    out[i] = 0;
}

// out[i] = in[i % n_in].
static inline void tile_impl(const int8_t *in, int8_t *out, int32_t n_in,
                             int32_t n_out) {
  for (int32_t i = 0; i < n_out; i++)
    out[i] = in[i % n_in];
}

// out[i] = in1[i] + in2[i] (int8 wrap).
static inline void add_impl(const int8_t *in1, const int8_t *in2, int8_t *out,
                            int32_t n) {
  for (int32_t i = 0; i < n; i++)
    out[i] = (int8_t)(in1[i] + in2[i]);
}

// out[i] = in1[i] (in2 dropped).
static inline void first_impl(const int8_t *in1, const int8_t *in2, int8_t *out,
                              int32_t n) {
  (void)in2;
  for (int32_t i = 0; i < n; i++)
    out[i] = in1[i];
}

extern "C" {

// Copies (each its own symbol so IRON can declare each call-site signature).
void llama_pt_copy_D_to_D(int8_t *in, int8_t *out) {
  copy_impl(in, out, kD, kD);
}
void llama_pt_copy_D_to_QD(int8_t *in, int8_t *out) {
  copy_impl(in, out, kD, kQD);
}
void llama_pt_copy_D_to_KVD(int8_t *in, int8_t *out) {
  copy_impl(in, out, kD, kKVD);
}
void llama_pt_copy_QD_to_QD(int8_t *in, int8_t *out) {
  copy_impl(in, out, kQD, kQD);
}
void llama_pt_copy_QD_to_D(int8_t *in, int8_t *out) {
  copy_impl(in, out, kQD, kD);
}
void llama_pt_copy_KVD_to_KVD(int8_t *in, int8_t *out) {
  copy_impl(in, out, kKVD, kKVD);
}
void llama_pt_copy_HD_to_D(int8_t *in, int8_t *out) {
  copy_impl(in, out, kHD, kD);
}

// Tile (upsize).
void llama_pt_tile_D_to_HD(int8_t *in, int8_t *out) {
  tile_impl(in, out, kD, kHD);
}

// Add (in-place size).
void llama_pt_add_D(int8_t *in1, int8_t *in2, int8_t *out) {
  add_impl(in1, in2, out, kD);
}
void llama_pt_add_HD(int8_t *in1, int8_t *in2, int8_t *out) {
  add_impl(in1, in2, out, kHD);
}

// First (drop second input).
void llama_pt_first_QD_KVD(int8_t *in1, int8_t *in2, int8_t *out) {
  first_impl(in1, in2, out, kQD);
}
void llama_pt_first_QD_QD(int8_t *in1, int8_t *in2, int8_t *out) {
  first_impl(in1, in2, out, kQD);
}

} // extern "C"
