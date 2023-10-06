// clang-format off
void dut(bfloat16 * restrict v4, size_t m1, bfloat16 * restrict v5, size_t m2, bfloat16 * restrict v6, size_t m3) {
  size_t v7 = 0;
  size_t v8 = 16;
  for (size_t v9 = v7; v9 < m1; v9 += v8)
  chess_prepare_for_pipelining
  {
    v16bfloat16 v10 = *(v16bfloat16 *)(v4 + v9);
    v16bfloat16 v11 = *(v16bfloat16 *)(v5 + v9);
    v16accfloat v12 = ups_to_v16accfloat(v10);
    v16accfloat v13 = ups_to_v16accfloat(v11);
    v16accfloat v14 = sub(v12, v13);
    v16bfloat16 v15 = to_v16bfloat16(v14);
    *(v16bfloat16 *)(v6 + v9) = v15;
  }
  return;
}
// clang-format on
