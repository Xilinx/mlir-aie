// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  size_t v4 = 0;
  size_t v5 = 16;
  size_t v6 = 1;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v8 = 0;
    size_t v9 = 1024;
    size_t v10 = 16;
    for (size_t v11 = v8; v11 < v9; v11 += v10)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16bfloat16 v12 = *(v16bfloat16 *)(v1 + 1024*v7+v11);
      v16bfloat16 v13 = *(v16bfloat16 *)(v2 + v11);
      v16accfloat v14 = ups_to_v16accfloat(v12);
      v16accfloat v15 = ups_to_v16accfloat(v13);
      v16accfloat v16 = sub(v14, v15);
      v16bfloat16 v17 = to_v16bfloat16(v16);
      *(v16bfloat16 *)(v3 + 1024*v7+v11) = v17;
    }
  }
  return;
}
// clang-format on
