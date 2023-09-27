// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  bfloat16 v4 = 0.0e+00;
  bfloat16 * restrict v5 = v2;
  v32bfloat16 v6 = broadcast_to_v32bfloat16(v4);
  v16bfloat16 v7 = extract_v16bfloat16(v6, 0);
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
      v16bfloat16 v16 = *(v16bfloat16 *)(v1 + 1024*v11+v15);
      v16bfloat16 v17 = *(v16bfloat16 *)(v5 + v15);
      v32bfloat16 v18 = concat(v16, v7);
      v32bfloat16 v19 = concat(v17, v7);
      v16accfloat v20 = mul_elem_16_2(v19, v18);
      v16bfloat16 v21 = to_v16bfloat16(v20);
      *(v16bfloat16 *)(v3 + 1024*v11+v15) = v21;
    }
  }
  return;
}
// clang-format on
