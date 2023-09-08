// clang-format off
void dut(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
  size_t v4 = 0;
  v32int16 v5 = *(v32int16 *)(v2 + v4);
  v32int16 v6 = broadcast_elem(v5, 0);
  size_t v7 = 0;
  size_t v8 = 16;
  size_t v9 = 1;
  for (size_t v10 = v7; v10 < v8; v10 += v9)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v11 = 0;
    size_t v12 = 1024;
    size_t v13 = 32;
    for (size_t v14 = v11; v14 < v12; v14 += v13)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int16 v15 = *(v32int16 *)(v1 + 1024*v10+v14);
      v32int16 v16 = sub(v15, v6);
      *(v32int16 *)(v3 + 1024*v10+v14) = v16;
    }
  }
  return;
}
// clang-format on
