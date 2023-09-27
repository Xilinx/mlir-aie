// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  size_t v4 = 0;
  v16int32 v5 = *(v16int32 *)(v2 + v4);
  v16int32 v6 = broadcast_elem(v5, 0);
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
      v16int32 v15 = *(v16int32 *)(v1 + 1024*v10+v14);
      v16acc64 v16 = mul_elem_16_2(v6, broadcast_zero_s32(), v15, undef_v16int32());
      v16int32 v17 = srs_to_v16int32(v16, 0);
      *(v16int32 *)(v3 + 1024*v10+v14) = v17;
    }
  }
  return;
}
// clang-format on
