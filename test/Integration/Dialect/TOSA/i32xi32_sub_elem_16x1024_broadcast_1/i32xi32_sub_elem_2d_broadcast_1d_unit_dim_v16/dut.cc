// Cycle count: 2131
// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  size_t v4 = 0;
  int32_t * restrict v5 = v2;
  v16int32 v6 = *(v16int32 *)(v5 + v4+v4);
  v16int32 v7 = broadcast_elem(v6, 0);
  size_t v8 = 0;
  size_t v9 = 16;
  size_t v10 = 1;
  for (size_t v11 = v8; v11 < v9; v11 += v10)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v12 = 0;
    size_t v13 = 1024;
    size_t v14 = 16;
    for (size_t v15 = v12; v15 < v13; v15 += v14)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16int32 v16 = *(v16int32 *)(v1 + 1024*v11+v15);
      v16int32 v17 = sub(v16, v7);
      *(v16int32 *)(v3 + 1024*v11+v15) = v17;
    }
  }
  return;
}
// clang-format on
