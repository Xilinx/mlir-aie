void dut(int8_t *restrict v1, int8_t *restrict v2, int32_t *restrict v3) {
  v64int8 v4 = broadcast_zero_to_v64int8();
  v32int8 v5 = extract_v32int8(v4, 0);
  size_t v6 = 0;
  size_t v7 = 1024;
  size_t v8 = 32;
  for (size_t v9 = v6; v9 < v7; v9 += v8)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int8 v10 = *(v32int8 *)(v1 + v9);
      v32int8 v11 = *(v32int8 *)(v2 + v9);
      v64int8 v12 = concat(v10, v5);
      v64int8 v13 = concat(v11, v5);
      v32acc32 v14 = mul_elem_32_2(v13, v12);
      v32int32 v15 = v32int32(v14);
      *(v32int32 *)(v3 + v9) = v15;
    }
  return;
}
