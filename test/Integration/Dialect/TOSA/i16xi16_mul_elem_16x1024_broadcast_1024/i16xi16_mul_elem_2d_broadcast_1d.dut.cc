// clang-format off
void dut(int16_t * restrict v1, int16_t * restrict v2, int16_t * restrict v3) {
  size_t v4 = 0;
  size_t v5 = 16;
  size_t v6 = 1;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v8 = 0;
    size_t v9 = 1024;
    size_t v10 = 32;
    for (size_t v11 = v8; v11 < v9; v11 += v10)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int16 v12 = *(v32int16 *)(v1 + 1024*v7+v11);
      v32int16 v13 = *(v32int16 *)(v2 + v11);
      v32acc32 v14 = mul_elem_32(v13, v12);
      v32int16 v15 = srs_to_v32int16(v14, 0);
      *(v32int16 *)(v3 + 1024*v7+v11) = v15;
    }
  }
  return;
}
// clang-format on
