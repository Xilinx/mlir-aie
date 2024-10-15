void dut(int8_t *restrict v1, int8_t *restrict v2, int8_t *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 64;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      v64int8 v8 = *(v64int8 *)(v1 + v7);
      v64int8 v9 = *(v64int8 *)(v2 + v7);
      v64int8 v10 = add(v8, v9);
      *(v64int8 *)(v3 + v7) = v10;
    }
  return;
}
