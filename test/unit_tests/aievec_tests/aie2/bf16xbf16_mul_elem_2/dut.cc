// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, float * restrict v3) {
  bfloat16 v4 = 0.0e+00;
  v32bfloat16 v5 = broadcast_to_v32bfloat16(v4);
  v16bfloat16 v6 = extract_v16bfloat16(v5, 0);
  size_t v7 = 0;
  size_t v8 = 1024;
  size_t v9 = 16;
  for (size_t v10 = v7; v10 < v8; v10 += v9)
  chess_prepare_for_pipelining
  chess_loop_range(64, 64)
  {
    v16bfloat16 v11 = *(v16bfloat16 *)(v1 + v10);
    v16bfloat16 v12 = *(v16bfloat16 *)(v2 + v10);
    v32bfloat16 v13 = concat(v11, v6);
    v32bfloat16 v14 = concat(v12, v6);
    v16accfloat v15 = mul_elem_16_2(v14, v13);
    v16float v16 = v16float(v15);
    *(v16float *)(v3 + v10) = v16;
  }
  return;
}
// clang-format on
