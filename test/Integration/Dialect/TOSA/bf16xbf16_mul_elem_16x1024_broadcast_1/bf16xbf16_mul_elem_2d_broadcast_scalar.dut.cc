// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  size_t v4 = 0;
  bfloat16 v5 = 0.0e+00;
  bfloat16 * restrict v6 = v2;
  v16bfloat16 v7 = *(v16bfloat16 *)(v6 + v4);
  v32bfloat16 v8 = concat(v7, v7);
  v32bfloat16 v9 = broadcast_elem(v8, 0);
  v16bfloat16 v10 = extract_v16bfloat16(v9, 0);
  v32bfloat16 v11 = broadcast_to_v32bfloat16(v5);
  v16bfloat16 v12 = extract_v16bfloat16(v11, 0);
  v32bfloat16 v13 = concat(v10, v12);
  size_t v14 = 0;
  size_t v15 = 16;
  size_t v16 = 1;
  for (size_t v17 = v14; v17 < v15; v17 += v16)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v18 = 0;
    size_t v19 = 1024;
    size_t v20 = 16;
    for (size_t v21 = v18; v21 < v19; v21 += v20)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16bfloat16 v22 = *(v16bfloat16 *)(v1 + 1024*v17+v21);
      v32bfloat16 v23 = concat(v22, v12);
      v16accfloat v24 = mul_elem_16_2(v13, v23);
      v16bfloat16 v25 = to_v16bfloat16(v24);
      *(v16bfloat16 *)(v3 + 1024*v17+v21) = v25;
    }
  }
  return;
}
// clang-format on
