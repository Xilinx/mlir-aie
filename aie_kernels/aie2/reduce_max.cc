#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

void _reduce_max_vector(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {

  v16int32 tiny = broadcast_to_v16int32((int32_t)INT32_MIN);
  const int32_t vector_size = 16;
  v16int32 after_vector;
  v16int32 running_max = tiny;
  for (int32_t i = 0; i < input_size; i += vector_size)
    chess_prepare_for_pipelining chess_loop_range(8, ) {
      v16int32 next = *(v16int32 *)(in + i);
      v16int32 test = max(running_max, next);
      running_max = test;
    }
  after_vector = running_max;
  v16int32 first = shift_bytes(after_vector, after_vector, 32U);
  v16int32 second = max(after_vector, first);
  v16int32 second_shift = shift_bytes(second, second, 16U);
  v16int32 third = max(second, second_shift);
  v16int32 third_shift = shift_bytes(third, third, 8U);
  v16int32 fourth = max(third, third_shift);
  v16int32 fourth_shift = shift_bytes(fourth, fourth, 4U);
  v16int32 fifth = max(fourth, fourth_shift);
  int32_t last = extract_elem(fifth, 0U);
  *(int32_t *)out = last;
  return;
}

void _reduce_max_scalar(int32_t *restrict in, int32_t *restrict out,
                        const int32_t input_size) {
  int32_t running_max = (int32_t)INT32_MIN;
  for (int32_t i = 0; i < input_size; i++) {
    if (in[i] > running_max)
      running_max = in[i];
  }
  *(int32_t *)out = running_max;

  return;
}

extern "C" {

void reduce_max_vector(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_vector(a_in, c_out, input_size);
}

void reduce_max_scalar(int32_t *a_in, int32_t *c_out, int32_t input_size) {
  _reduce_max_scalar(a_in, c_out, input_size);
}

} // extern "C"
