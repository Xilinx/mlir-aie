void dut(int8_t *restrict v1, int8_t *restrict v2, int32_t *restrict v3) {
  int8_t v4 = 0;
  v64int8 v5 = broadcast_to_v64int8(v4);
  v32int8 v6 = extract_v32int8(v5, 0);
  size_t v7 = 0;
  size_t v8 = 1024;
  size_t v9 = 32;
  for (size_t v10 = v7; v10 < v8; v10 += v9)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int8 v11 = *(v32int8 *)(v1 + v10);
      v32int8 v12 = *(v32int8 *)(v2 + v10);
      v64int8 v13 = concat(v11, v6);
      v64int8 v14 = concat(v12, v6);
      v32acc32 v15 = mul_elem_32_2(v14, v13);
      v32int32 v16 = v32int32(v15);
      *(v32int32 *)(v3 + v10) = v16;
    }
  return;
}
