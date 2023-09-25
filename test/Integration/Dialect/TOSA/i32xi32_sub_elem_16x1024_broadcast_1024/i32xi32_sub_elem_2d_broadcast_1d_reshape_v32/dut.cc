// Cycle count: 3177
// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  size_t v4 = 0;
  int32_t * restrict v5 = v2;
  size_t v6 = 0;
  size_t v7 = 16;
  size_t v8 = 1;
  for (size_t v9 = v6; v9 < v7; v9 += v8)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v10 = 0;
    size_t v11 = 1024;
    size_t v12 = 32;
    for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int32 v14 = *(v32int32 *)(v1 + 1024*v9+v13);
      v32int32 v15 = *(v32int32 *)(v5 + 1024*v4+v13);
      v32acc32 v16 = v32acc32(v14);
      v32acc32 v17 = v32acc32(v15);
      v32acc32 v18 = sub(v16, v17);
      v32int32 v19 = v32int32(v18);
      *(v32int32 *)(v3 + 1024*v9+v13) = v19;
    }
  }
  return;
}
// clang-format on
