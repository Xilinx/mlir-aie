void dut(int8_t *restrict v1, int8_t *restrict v2) {
  v64int8 v3 = broadcast_to_v64int8((int8_t)-128);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 64;
  v64int8 v7;
  v64int8 v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      v64int8 v10 = *(v64int8 *)(v1 + v9);
      v64int8 v11 = max(v8, v10);
      v8 = v11;
    }
  v7 = v8;
  v64int8 v12 = shift_bytes(v7, undef_v64int8(), 32);
  v64int8 v13 = max(v7, v12);
  v64int8 v14 = shift_bytes(v13, undef_v64int8(), 16);
  v64int8 v15 = max(v13, v14);
  v64int8 v16 = shift_bytes(v15, undef_v64int8(), 8);
  v64int8 v17 = max(v15, v16);
  v64int8 v18 = shift_bytes(v17, undef_v64int8(), 4);
  v64int8 v19 = max(v17, v18);
  v64int8 v20 = shift_bytes(v19, undef_v64int8(), 2);
  v64int8 v21 = max(v19, v20);
  v64int8 v22 = shift_bytes(v21, undef_v64int8(), 1);
  v64int8 v23 = max(v21, v22);
  int8_t v24 = extract_elem(v23, 0);
  *(int8_t *)v2 = v24;
  return;
}
