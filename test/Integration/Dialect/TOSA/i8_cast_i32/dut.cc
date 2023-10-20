void dut(int8_t *restrict v1, int32_t *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 32;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(32, 32) {
      v32int8 v7 = *(v32int8 *)(v1 + v6);
      v32acc32 v8 = ups_to_v32acc32(v7, 0);
      v32int32 v9 = v32int32(v8);
      *(v32int32 *)(v2 + v6) = v9;
    }
  return;
}
