// clang-format off
void dut(int32_t * restrict v1, int32_t * restrict v2, int32_t * restrict v3) {
  int32_t * restrict v4 = v2;
  size_t v5 = 0;
  size_t v6 = 16;
  size_t v7 = 1;
  for (size_t v8 = v5; v8 < v6; v8 += v7)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v9 = 0;
    size_t v10 = 1024;
    size_t v11 = 16;
    for (size_t v12 = v9; v12 < v10; v12 += v11)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16int32 v13 = *(v16int32 *)(v1 + 1024*v8+v12);
      v16int32 v14 = *(v16int32 *)(v4 + v12);
      v16acc64 v15 = mul_elem_16_2(v14, broadcast_zero_s32(), v13, undef_v16int32());
      v16int32 v16 = srs_to_v16int32(v15, 0);
      *(v16int32 *)(v3 + 1024*v8+v12) = v16;
    }
  }
  return;
}
// clang-format on
