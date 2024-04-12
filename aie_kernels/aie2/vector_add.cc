#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>

void _add_reduce_scalar(int32_t *restrict in, int32_t *restrict out, const int32_t input_size) {
  int32_t running_total = 0;
  for (int32_t i = 0; i < input_size; i++) {
      running_total = running_total + in[i];
  }
  *out = running_total;
  return;
}

void _add_reduce_vector(int32_t *restrict in, int32_t *restrict out, const int32_t input_size) {
  v16int32 zero = broadcast_to_v16int32((int32_t)0);
  int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_total = zero;
  for (int32_t i = 0; i < input_size; i += vector_size)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      v16int32 next = *(v16int32 *)(in + i);
      v16int32 test = add(running_total, next);
      running_total = test;
    }
  after_vector = running_total;
  v16int32 first = shift_bytes(after_vector, after_vector, 32);
  v16int32 second = add(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16);
  v16int32 third = add(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8);
  v16int32 fourth = add(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4);
  v16int32 fifth = add(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0);
  *(int32_t *)out = last;
  return;
}

extern "C" {
    void add_reduce_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) { _add_reduce_vector(a_in, c_out, input_size); }
    void add_reduce_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) { _add_reduce_scalar(a_in, c_out, input_size); }
} // extern "C"