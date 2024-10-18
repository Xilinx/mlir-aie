void dut(int16_t *restrict v1, int16_t *restrict v2, int16_t *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 32;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int16 v8 = *(v32int16 *)(v1 + v7);
      v32int16 v9 = *(v32int16 *)(v2 + v7);
      uint32_t v10 = le(v8, v9);
      v32int16 v11 = sel(v9, v8, v10);
      *(v32int16 *)(v3 + v7) = v11;
    }
  return;
}
