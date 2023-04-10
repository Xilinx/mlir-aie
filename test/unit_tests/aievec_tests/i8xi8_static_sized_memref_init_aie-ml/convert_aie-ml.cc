void conv2d(int8_t *restrict v1, int8_t *restrict v2, int8_t *restrict v3) {
  size_t v4 = 0;
  v64int8 v5 = *(v64int8 *)(v2 + v4);
  v64int8 v6 = shuffle(v5, 0);
  v64int8 v7 = shift_bytes(v6, undef_v64int8(), 8);
  v64int8 v8 = shift_bytes(v6, undef_v64int8(), 16);
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
      size_t v19 = 32;
      for (size_t v20 = v17; v20 < v18; v20 += v19)
        chess_prepare_for_pipelining chess_loop_range(8, 8) {
          v32int8 v21 = *(v32int8 *)(v3 + 256 * v12 + v20);
          v64int8 v22 = *(v64int8 *)(v1 + 288 * v12 + v20);
          v32acc32 v23 = ups_to_v32acc32(v21, 0);
          v23 = mac_conv_32x8(v22, v6, v23);
          v64int8 v24 = *(v64int8 *)(v1 + 288 * v14 + v20);
          v23 = mac_conv_32x8(v24, v7, v23);
          v64int8 v25 = *(v64int8 *)(v1 + 288 * v16 + v20);
          v23 = mac_conv_32x8(v25, v8, v23);
          v32int8 v26 = srs_to_v32int8(v23, 0);
          *(v32int8 *)(v3 + 256 * v12 + v20) = v26;
        }
    }
  return;
}
