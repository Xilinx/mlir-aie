void conv2d(int16_t *restrict v1, int16_t *restrict v2, int16_t *restrict v3) {
  size_t v4 = 0;
  v32int16 v5 = *(v32int16 *)(v2 + v4);
  v32int16 v6 = shift_bytes(v5, undef_v32int16(), 8);
  v32int16 v7 = shift_bytes(v5, undef_v32int16(), 16);
  size_t v8 = 0;
  size_t v9 = 16;
  size_t v10 = 1;
  for (size_t v11 = v8; v11 < v9; v11 += v10)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      size_t v12 = 1;
      size_t v13 = v11 + v12;
      size_t v14 = 2;
      size_t v15 = v11 + v14;
      size_t v16 = 0;
      size_t v17 = 256;
      size_t v18 = 16;
      for (size_t v19 = v16; v19 < v17; v19 += v18)
        chess_prepare_for_pipelining chess_loop_range(16, 16) {
          v32int16 v20 = *(v32int16 *)(v1 + 288 * v11 + v19);
          v16acc64 v21 = mul_conv_16x4(v20, v5);
          v32int16 v22 = *(v32int16 *)(v1 + 288 * v13 + v19);
          v21 = mac_conv_16x4(v22, v6, v21);
          v32int16 v23 = *(v32int16 *)(v1 + 288 * v15 + v19);
          v21 = mac_conv_16x4(v23, v7, v21);
          v16int16 v24 = srs_to_v16int16(v21, 10);
          *(v16int16 *)(v3 + 256 * v11 + v19) = v24;
        }
    }
  return;
}
