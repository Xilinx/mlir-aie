void dut(float *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16float v8 = *(v16float *)(v1 + v7);
      v16accfloat v9 = v16accfloat(v8);
      v16bfloat16 v10 = to_v16bfloat16(v9);
      *(v16bfloat16 *)(v2 + v7) = v10;
    }
  return;
}
