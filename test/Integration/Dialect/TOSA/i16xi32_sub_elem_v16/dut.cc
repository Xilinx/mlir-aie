void dut(int16_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  int32_t v4 = 0;
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 16;
  for (size_t v8 = v5; v8 < v6; v8 += v7)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int16 v9 = *(v16int16 *)(v1 + v8);
      v16int32 v10 = *(v16int32 *)(v2 + v8);
      v16acc64 v11 = ups_to_v16acc64(v9, 0);
      v16acc64 v12 = ups_to_v16acc64(v10, 0);
      v16acc64 v13 = sub(v11, v12);
      v16int32 v14 = srs_to_v16int32(v13, v4);
      *(v16int32 *)(v3 + v8) = v14;
    }
  return;
}
