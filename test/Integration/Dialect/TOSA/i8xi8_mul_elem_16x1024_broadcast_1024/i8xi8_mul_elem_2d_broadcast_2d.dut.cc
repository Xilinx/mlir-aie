// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  int8_t v4 = 0;
  int8_t * restrict v5 = v2;
  v64int8 v6 = broadcast_to_v64int8(v4);
  v32int8 v7 = extract_v32int8(v6, 0);
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
      v32int8 v16 = *(v32int8 *)(v1 + 1024*v11+v15);
      v32int8 v17 = *(v32int8 *)(v5 + v15);
      v64int8 v18 = concat(v16, v7);
      v64int8 v19 = concat(v17, v7);
      v32acc32 v20 = mul_elem_32_2(v19, v18);
      v32int8 v21 = srs_to_v32int8(v20, 0);
      *(v32int8 *)(v3 + 1024*v11+v15) = v21;
    }
  }
  return;
}
// clang-format on
