void dut(int16_t *restrict v1, int16_t *restrict v2, int16_t *restrict v3) {
  int32_t v4 = 0;
  size_t v5 = 0;
  v32int16 v6 = *(v32int16 *)(v2 + v5);
  v32int16 v7 = broadcast_elem(v6, 0);
  int16_t v8 = extract_elem(v7, v4);
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 32;
  for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int16 v13 = *(v32int16 *)(v1 + v12);
      v32acc32 v14 = ups_to_v32acc32(v13, 0);
      v32int16 v15 = srs_to_v32int16(v14, (int32_t)v8);
      *(v32int16 *)(v3 + v12) = v15;
    }
  return;
}
