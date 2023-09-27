// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  size_t v4 = 0;
  bfloat16 v5 = 0.0e+00;
  v16bfloat16 v6 = *(v16bfloat16 *)(v2 + v4);
  v32bfloat16 v7 = concat(v6, v6);
  v32bfloat16 v8 = broadcast_elem(v7, 0);
  v16bfloat16 v9 = extract_v16bfloat16(v8, 0);
  v32bfloat16 v10 = broadcast_to_v32bfloat16(v5);
  v16bfloat16 v11 = extract_v16bfloat16(v10, 0);
  v32bfloat16 v12 = concat(v9, v11);
  size_t v13 = 0;
  size_t v14 = 16;
  size_t v15 = 1;
  for (size_t v16 = v13; v16 < v14; v16 += v15)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v17 = 0;
    size_t v18 = 1024;
    size_t v19 = 16;
    for (size_t v20 = v17; v20 < v18; v20 += v19)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16bfloat16 v21 = *(v16bfloat16 *)(v1 + 1024*v16+v20);
      v32bfloat16 v22 = concat(v21, v11);
      v16accfloat v23 = mul_elem_16_2(v12, v22);
      v16bfloat16 v24 = to_v16bfloat16(v23);
      *(v16bfloat16 *)(v3 + 1024*v16+v20) = v24;
    }
  }
  return;
}
// clang-format on
