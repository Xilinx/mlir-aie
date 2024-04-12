#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

void vector(int32_t *restrict in, int32_t *restrict out) {

  v16int32 massive = broadcast_to_v16int32((int32_t)2147483647);
  int32_t input_size = 1024;
  int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_min = massive;
  for (int32_t i = 0; i < input_size; i += vector_size)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 next = *(v16int32 *)(in + i);
      v16int32 test = min(running_min, next);
      running_min = test;
    }
  after_vector = running_min;
  v16int32 first = shift_bytes(after_vector, after_vector, 32);
  v16int32 second = min(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16);
  v16int32 third = min(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8);
  v16int32 fourth = min(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4);
  v16int32 fifth = min(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0);
  *(int32_t *)out = last;
  return;
}

void scalar(int32_t *restrict in, int32_t *restrict out) {
  size_t input_size = 1024;
  int32_t running_min = (int32_t)2147483647;
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] < running_min)
      running_min = in[i];
  }
  *(int32_t *)out = running_min;

  return;
}

extern "C" {

void vector_min(int32_t *a_in, int32_t *c_out) { vector(a_in, c_out); }

void scalar_min(int32_t *a_in, int32_t *c_out) { scalar(a_in, c_out); }

} // extern "C"
