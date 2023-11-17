void dut(int8_t *restrict v1, int8_t *restrict v2, int8_t *restrict v3) {
  int32_t v4 = 0;
  size_t v5 = 0;
  v64int8 v6 = *(v64int8 *)(v2 + v5);
  v64int8 v7 = broadcast_elem(v6, 0);
  int8_t v8 = extract_elem(v7, v4);
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 64;
  for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      v64int8 v13 = *(v64int8 *)(v1 + v12);
      v32int8 v14 = extract_v32int8(v13, 0);
      v32int8 v15 = extract_v32int8(v13, 1);
      v32acc32 v16 = ups_to_v32acc32(v14, 0);
      v32int8 v17 = srs_to_v32int8(v16, (int32_t)v8);
      v32acc32 v18 = ups_to_v32acc32(v15, 0);
      v32int8 v19 = srs_to_v32int8(v18, (int32_t)v8);
      v64int8 v20 = concat(v17, v19);
      *(v64int8 *)(v3 + v12) = v20;
    }
  return;
}
