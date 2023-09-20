// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  size_t v4 = 0;
  int8_t * restrict v5 = v2;
  size_t v6 = 0;
  size_t v7 = 16;
  size_t v8 = 1;
  for (size_t v9 = v6; v9 < v7; v9 += v8)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v10 = 0;
    size_t v11 = 1024;
    size_t v12 = 64;
    for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining
    chess_loop_range(16, 16)
    {
      v64int8 v14 = *(v64int8 *)(v1 + 1024*v9+v13);
      v64int8 v15 = *(v64int8 *)(v5 + 1024*v4+v13);
      v64int8 v16 = sub(v14, v15);
      *(v64int8 *)(v3 + 1024*v9+v13) = v16;
    }
  }
  return;
}
// clang-format on
