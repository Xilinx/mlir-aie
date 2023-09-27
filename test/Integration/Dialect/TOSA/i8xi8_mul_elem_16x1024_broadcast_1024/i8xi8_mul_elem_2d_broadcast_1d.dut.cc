// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  int8_t v4 = 0;
  v64int8 v5 = broadcast_to_v64int8(v4);
  v32int8 v6 = extract_v32int8(v5, 0);
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
      v32int8 v15 = *(v32int8 *)(v1 + 1024*v10+v14);
      v32int8 v16 = *(v32int8 *)(v2 + v14);
      v64int8 v17 = concat(v15, v6);
      v64int8 v18 = concat(v16, v6);
      v32acc32 v19 = mul_elem_32_2(v18, v17);
      v32int8 v20 = srs_to_v32int8(v19, 0);
      *(v32int8 *)(v3 + 1024*v10+v14) = v20;
    }
  }
  return;
}
// clang-format on
