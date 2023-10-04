// clang-format off
void dut(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
  size_t v4 = 0;
  int16_t * restrict v5 = v2;
  v32int16 v6 = *(v32int16 *)(v5 + v4+v4);
  v32int16 v7 = broadcast_elem(v6, 0);
  size_t v8 = 0;
  size_t v9 = 16;
  size_t v10 = 1;
  for (size_t v11 = v8; v11 < v9; v11 += v10)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v12 = 0;
    size_t v13 = 1024;
    size_t v14 = 32;
    for (size_t v15 = v12; v15 < v13; v15 += v14)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int16 v16 = *(v32int16 *)(v1 + 1024*v11+v15);
      v32int16 v17 = sub(v16, v7);
      *(v32int16 *)(v3 + 1024*v11+v15) = v17;
    }
  }
  return;
}
// clang-format on
