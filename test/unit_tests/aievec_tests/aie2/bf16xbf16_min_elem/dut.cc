void dut(bfloat16 *restrict v1, bfloat16 *restrict v2, bfloat16 *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v32bfloat16 v8 = *(v32bfloat16 *)(v1 + v7);
      v32bfloat16 v9 = *(v32bfloat16 *)(v2 + v7);
      v32bfloat16 v10 = min(v8, v9);
      *(v32bfloat16 *)(v3 + v7) = v10;
    }
  return;
}
