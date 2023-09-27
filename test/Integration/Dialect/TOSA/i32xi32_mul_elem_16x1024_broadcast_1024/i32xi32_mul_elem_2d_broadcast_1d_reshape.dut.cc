// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
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
      v16int32 v12 = *(v16int32 *)(v1 + 1024*v7+v11);
      v16int32 v13 = *(v16int32 *)(v2 + v11);
      v16acc64 v14 = mul_elem_16_2(v13, broadcast_zero_s32(), v12, undef_v16int32());
      v16int32 v15 = srs_to_v16int32(v14, 0);
      *(v16int32 *)(v3 + 1024*v7+v11) = v15;
    }
  }
  return;
}
// clang-format on
