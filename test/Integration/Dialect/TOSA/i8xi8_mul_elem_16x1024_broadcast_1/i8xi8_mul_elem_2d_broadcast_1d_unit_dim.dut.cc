// clang-format off
void dut(int8_t * restrict v1, int8_t * restrict v2, int8_t * restrict v3) {
  size_t v4 = 0;
  int8_t v5 = 0;
  v32int8 v6 = *(v32int8 *)(v2 + v4);
  v64int8 v7 = concat(v6, v6);
  v64int8 v8 = broadcast_elem(v7, 0);
  v32int8 v9 = extract_v32int8(v8, 0);
  v64int8 v10 = broadcast_to_v64int8(v5);
  v32int8 v11 = extract_v32int8(v10, 0);
  v64int8 v12 = concat(v9, v11);
  size_t v13 = 0;
  size_t v14 = 16;
  size_t v15 = 1;
  for (size_t v16 = v13; v16 < v14; v16 += v15)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v17 = 0;
    size_t v18 = 1024;
    size_t v19 = 32;
    for (size_t v20 = v17; v20 < v18; v20 += v19)
    chess_prepare_for_pipelining
    chess_loop_range(32, 32)
    {
      v32int8 v21 = *(v32int8 *)(v1 + 1024*v16+v20);
      v64int8 v22 = concat(v21, v11);
      v32acc32 v23 = mul_elem_32_2(v12, v22);
      v32int8 v24 = srs_to_v32int8(v23, 0);
      *(v32int8 *)(v3 + 1024*v16+v20) = v24;
    }
  }
  return;
}
// clang-format on
