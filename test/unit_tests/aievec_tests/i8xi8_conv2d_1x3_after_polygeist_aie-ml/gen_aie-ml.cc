void conv2d(int8_t *restrict v4, size_t m1, int8_t *restrict v5, size_t m2,
            int32_t *restrict v6, size_t m3) {
  size_t v7 = 0;
  v64int8 v8 = *(v64int8 *)(v5 + v7);
  v64int8 v9 = shuffle(v8, 0);
  size_t v10 = 0;
  size_t v11 = 16;
  size_t v12 = 1;
  for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      size_t v14 = 0;
      size_t v15 = 256;
      size_t v16 = 32;
      for (size_t v17 = v14; v17 < v15; v17 += v16)
        chess_prepare_for_pipelining chess_loop_range(8, 8) {
          v64int8 v18 = *(v64int8 *)(v4 + 288 * v13 + v17);
          v32acc32 v19 = mul_conv_32x8(v18, v9);
          v32int32 v20 = v32int32(v19);
          *(v32int32 *)(v6 + 256 * v13 + v17) = v20;
        }
    }
  return;
}
