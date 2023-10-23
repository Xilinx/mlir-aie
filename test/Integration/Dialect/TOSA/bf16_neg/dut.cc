void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 16;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v7 = *(v16bfloat16 *)(v1 + v6);
      v16accfloat v8 = ups_to_v16accfloat(v7);
      v16accfloat v9 = neg(v8);
      v16bfloat16 v10 = to_v16bfloat16(v9);
      *(v16bfloat16 *)(v2 + v6) = v10;
    }
  return;
}
