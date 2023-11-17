void dut(int32_t *restrict v1, int32_t *restrict v2, int32_t *restrict v3) {
  int32_t v4 = 0;
  size_t v5 = 0;
  v16int32 v6 = *(v16int32 *)(v2 + v5);
  v16int32 v7 = broadcast_elem(v6, 0);
  int32_t v8 = extract_elem(v7, v4);
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 16;
  for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v13 = *(v16int32 *)(v1 + v12);
      v16acc64 v14 = ups_to_v16acc64(v13, 0);
      v16int32 v15 = srs_to_v16int32(v14, v8);
      *(v16int32 *)(v3 + v12) = v15;
    }
  return;
}
