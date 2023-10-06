// clang-format off
void dut(bfloat16 * restrict v7, size_t m1, size_t m2, bfloat16 * restrict v8, size_t m3, size_t m4, bfloat16 * restrict v9, size_t m5, size_t m6) {
  size_t v10 = 0;
  size_t v11 = 1;
  for (size_t v12 = v10; v12 < m1; v12 += v11)
  chess_prepare_for_pipelining
  {
    size_t v13 = 0;
    size_t v14 = 16;
    for (size_t v15 = v13; v15 < m2; v15 += v14)
    chess_prepare_for_pipelining
    {
      v16bfloat16 v16 = *(v16bfloat16 *)(v7 + m2*v12+v15);
      v16bfloat16 v17 = *(v16bfloat16 *)(v8 + m4*v12+v15);
      v16accfloat v18 = ups_to_v16accfloat(v16);
      v16accfloat v19 = ups_to_v16accfloat(v17);
      v16accfloat v20 = sub(v18, v19);
      v16bfloat16 v21 = to_v16bfloat16(v20);
      *(v16bfloat16 *)(v9 + m6*v12+v15) = v21;
    }
  }
  return;
}
// clang-format on
