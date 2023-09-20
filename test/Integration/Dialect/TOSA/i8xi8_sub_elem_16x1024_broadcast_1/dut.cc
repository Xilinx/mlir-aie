// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  size_t v4 = 0;
  int8_t * restrict v5 = v2;
  v64int8 v6 = *(v64int8 *)(v5 + v4+v4);
  v64int8 v7 = broadcast_elem(v6, 0);
  size_t v8 = 0;
  size_t v9 = 16;
  size_t v10 = 1;
  for (size_t v11 = v8; v11 < v9; v11 += v10)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v12 = 0;
    size_t v13 = 1024;
    size_t v14 = 64;
    for (size_t v15 = v12; v15 < v13; v15 += v14)
    chess_prepare_for_pipelining
    chess_loop_range(16, 16)
    {
      v64int8 v16 = *(v64int8 *)(v1 + 1024*v11+v15);
      v64int8 v17 = sub(v16, v7);
      *(v64int8 *)(v3 + 1024*v11+v15) = v17;
    }
  }
  return;
}
// clang-format on
