void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v8 = *(v16bfloat16 *)(v1 + v7);
      v16accfloat v9 = ups_to_v16accfloat(v8);
      v16accfloat v10 = neg(v9);
      v16bfloat16 v11 = to_v16bfloat16(v10);
      *(v16bfloat16 *)(v2 + v7) = v11;
    }
  return;
}
