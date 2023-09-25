// Cycle count: 1111
// clang-format off
void dut(bfloat16 * restrict v1, bfloat16 * restrict v2, bfloat16 * restrict v3) {
  size_t v4 = 0;
  bfloat16 * restrict v5 = v2;
  v16bfloat16 v6 = *(v16bfloat16 *)(v5 + v4+v4);
  v32bfloat16 v7 = concat(v6, v6);
  v32bfloat16 v8 = broadcast_elem(v7, 0);
  v16bfloat16 v9 = extract_v16bfloat16(v8, 0);
  v16accfloat v10 = ups_to_v16accfloat(v9);
  size_t v11 = 0;
  size_t v12 = 16;
  size_t v13 = 1;
  for (size_t v14 = v11; v14 < v12; v14 += v13)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v15 = 0;
    size_t v16 = 1024;
    size_t v17 = 16;
    for (size_t v18 = v15; v18 < v16; v18 += v17)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16bfloat16 v19 = *(v16bfloat16 *)(v1 + 1024*v14+v18);
      v16accfloat v20 = ups_to_v16accfloat(v19);
      v16accfloat v21 = sub(v20, v10);
      v16bfloat16 v22 = to_v16bfloat16(v21);
      *(v16bfloat16 *)(v3 + 1024*v14+v18) = v22;
    }
  }
  return;
}
// clang-format on
