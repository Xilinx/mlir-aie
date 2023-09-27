// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  bfloat16 v4 = 0.0e+00;
  v32bfloat16 v5 = broadcast_to_v32bfloat16(v4);
  v16bfloat16 v6 = extract_v16bfloat16(v5, 0);
  size_t v7 = 0;
  size_t v8 = 16;
  size_t v9 = 1;
  for (size_t v10 = v7; v10 < v8; v10 += v9)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v11 = 0;
    size_t v12 = 1024;
    size_t v13 = 16;
    for (size_t v14 = v11; v14 < v12; v14 += v13)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16bfloat16 v15 = *(v16bfloat16 *)(v1 + 1024*v10+v14);
      v16bfloat16 v16 = *(v16bfloat16 *)(v2 + v14);
      v32bfloat16 v17 = concat(v15, v6);
      v32bfloat16 v18 = concat(v16, v6);
      v16accfloat v19 = mul_elem_16_2(v18, v17);
      v16bfloat16 v20 = to_v16bfloat16(v19);
      *(v16bfloat16 *)(v3 + 1024*v10+v14) = v20;
    }
  }
  return;
}
// clang-format on
