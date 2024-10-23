void conv2d(int16_t *restrict v4, size_t m1, int16_t *restrict v5, size_t m2,
            int16_t *restrict v6, size_t m3) {
  size_t v7 = 0;
  v16int16 v8 = *(v16int16 *)(v5 + v7);
  v32int16 v9 = concat(v8, v8);
  size_t v10 = 0;
  size_t v11 = 16;
  size_t v12 = 1;
  for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      size_t v14 = 0;
      size_t v15 = 256;
      size_t v16 = 16;
      for (size_t v17 = v14; v17 < v15; v17 += v16)
        chess_prepare_for_pipelining chess_loop_range(16, 16) {
          v32int16 v18 = *(v32int16 *)(v4 + 288 * v13 + v17);
          v16acc64 v19 = mul_conv_16x4(v18, v9);
          v16int16 v20 = srs_to_v16int16(v19, 10);
          *(v16int16 *)(v6 + 256 * v13 + v17) = v20;
        }
    }
  return;
}
