void conv2d(int16_t *restrict v1, int16_t *restrict v2, int16_t *restrict v3) {
  size_t v4 = 0;
  v16int16 v5 = *(v16int16 *)(v2 + v4);
  v32int16 v6 = concat(v5, v5);
  v32int16 v7 = shift_bytes(v6, undef_v32int16(), 8);
  v32int16 v8 = shift_bytes(v6, undef_v32int16(), 16);
  size_t v9 = 0;
  size_t v10 = 16;
  size_t v11 = 1;
  for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      size_t v13 = 1;
      size_t v14 = v12 + v13;
      size_t v15 = 2;
      size_t v16 = v12 + v15;
      size_t v17 = 0;
      size_t v18 = 256;
      size_t v19 = 16;
      for (size_t v20 = v17; v20 < v18; v20 += v19)
        chess_prepare_for_pipelining chess_loop_range(16, 16) {
          v32int16 v21 = *(v32int16 *)(v1 + 288 * v12 + v20);
          v16acc64 v22 = mul_conv_16x4(v21, v6);
          v32int16 v23 = *(v32int16 *)(v1 + 288 * v14 + v20);
          v22 = mac_conv_16x4(v23, v7, v22);
          v32int16 v24 = *(v32int16 *)(v1 + 288 * v16 + v20);
          v22 = mac_conv_16x4(v24, v8, v22);
          v16int16 v25 = srs_to_v16int16(v22, 10);
          *(v16int16 *)(v3 + 256 * v12 + v20) = v25;
        }
    }
  return;
}
