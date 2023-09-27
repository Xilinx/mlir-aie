// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  size_t v4 = 0;
  int8_t v5 = 0;
  int8_t * restrict v6 = v2;
  v32int8 v7 = *(v32int8 *)(v6 + v4);
  v64int8 v8 = concat(v7, v7);
  v64int8 v9 = broadcast_elem(v8, 0);
  v32int8 v10 = extract_v32int8(v9, 0);
  v64int8 v11 = broadcast_to_v64int8(v5);
  v32int8 v12 = extract_v32int8(v11, 0);
  v64int8 v13 = concat(v10, v12);
  size_t v14 = 0;
  size_t v15 = 16;
  size_t v16 = 1;
  for (size_t v17 = v14; v17 < v15; v17 += v16)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v18 = 0;
    size_t v19 = 1024;
    size_t v20 = 32;
    for (size_t v21 = v18; v21 < v19; v21 += v20)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int8 v22 = *(v32int8 *)(v1 + 1024*v17+v21);
      v64int8 v23 = concat(v22, v12);
      v32acc32 v24 = mul_elem_32_2(v13, v23);
      v32int8 v25 = srs_to_v32int8(v24, 0);
      *(v32int8 *)(v3 + 1024*v17+v21) = v25;
    }
  }
  return;
}
// clang-format on
