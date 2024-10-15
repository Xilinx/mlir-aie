void dut(bfloat16 *restrict v1, bfloat16 *restrict v2, bfloat16 *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v8 = *(v16bfloat16 *)(v1 + v7);
      v16bfloat16 v9 = *(v16bfloat16 *)(v2 + v7);
      v16accfloat v10 = ups_to_v16accfloat(v8);
      v16accfloat v11 = ups_to_v16accfloat(v9);
      v16accfloat v12 = add(v10, v11);
      v16bfloat16 v13 = to_v16bfloat16(v12);
      *(v16bfloat16 *)(v3 + v7) = v13;
    }
  return;
}
