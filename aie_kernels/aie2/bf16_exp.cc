#include <lut_based_ops.h>

template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {

  int vec_size = 16;
  for (int i = 0; i < N; i += vec_size)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 vec_in = *(v16bfloat16 *)(in + i);
      v16accfloat acc_exp = getExpBf16(vec_in);
      v16bfloat16 bf16_exp = to_v16bfloat16(acc_exp);
      *(v16bfloat16 *)(out + i) = bf16_exp;
    }
  return;
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  exp_bf16_func<1024>(a_in, c_out);
}

} // extern "C"
