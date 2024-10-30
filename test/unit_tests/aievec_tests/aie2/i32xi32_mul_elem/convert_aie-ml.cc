void mul_elem(int32_t *restrict v1, int32_t *restrict v2,
              int32_t *restrict v3) {
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  for (size_t v7 = v4; v7 < v5; v7 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v8 = *(v16int32 *)(v1 + v7);
      v16int32 v9 = *(v16int32 *)(v2 + v7);
      v16acc64 v10 =
          mul_elem_16_2(v9, broadcast_zero_s32(), v8, undef_v16int32());
      v16int32 v11 = srs_to_v16int32(v10, 0);
      *(v16int32 *)(v3 + v7) = v11;
    }
  return;
}
