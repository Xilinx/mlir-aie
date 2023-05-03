void dut(int32_t *restrict v1, int32_t *restrict v2, int32_t *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 32;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int32 v8 = *(v32int32 *)(v1 + v7);
      v32int32 v9 = *(v32int32 *)(v2 + v7);
      v32acc32 v10 = v32acc32(v8);
      v32acc32 v11 = v32acc32(v9);
      v32acc32 v12 = sub(v10, v11);
      v32int32 v13 = v32int32(v12);
      *(v32int32 *)(v3 + v7) = v13;
    }
  return;
}
