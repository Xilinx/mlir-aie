// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  size_t v4 = 0;
  size_t v5 = 0;
  size_t v6 = 16;
  size_t v7 = 1;
  for (size_t v8 = v5; v8 < v6; v8 += v7)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v9 = 0;
    size_t v10 = 1024;
    size_t v11 = 64;
    for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining
    chess_loop_range(16, 16)
    {
      v64int8 v13 = *(v64int8 *)(v1 + 1024*v8+v12);
      v64int8 v14 = *(v64int8 *)(v2 + 1024*v4+v12);
      v64int8 v15 = sub(v13, v14);
      *(v64int8 *)(v3 + 1024*v8+v12) = v15;
    }
  }
  return;
}
// clang-format on
