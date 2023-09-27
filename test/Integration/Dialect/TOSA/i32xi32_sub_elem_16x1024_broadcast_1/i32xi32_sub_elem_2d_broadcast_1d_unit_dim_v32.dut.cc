// Cycle count: 2148
// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  size_t v4 = 0;
  int32_t * restrict v5 = v2;
  v16int32 v6 = *(v16int32 *)(v5 + v4+v4);
  v16int32 v7 = broadcast_elem(v6, 0);
  v32int32 v8 = concat(v7, v7);
  v32acc32 v9 = v32acc32(v8);
  size_t v10 = 0;
  size_t v11 = 16;
  size_t v12 = 1;
  for (size_t v13 = v10; v13 < v11; v13 += v12)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v14 = 0;
    size_t v15 = 1024;
    size_t v16 = 32;
    for (size_t v17 = v14; v17 < v15; v17 += v16)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int32 v18 = *(v32int32 *)(v1 + 1024*v13+v17);
      v32acc32 v19 = v32acc32(v18);
      v32acc32 v20 = sub(v19, v9);
      v32int32 v21 = v32int32(v20);
      *(v32int32 *)(v3 + 1024*v13+v17) = v21;
    }
  }
  return;
}
// clang-format on
