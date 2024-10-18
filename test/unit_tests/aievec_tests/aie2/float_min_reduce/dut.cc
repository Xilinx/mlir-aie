void dut(float *restrict v1, float *restrict v2) {
  int32_t v3 = 0;
  v16float v4 = broadcast_to_v16float(
      (float)340282346638528859811704183484516925440.000000);
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 16;
  v16float v8;
  v16float v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16float v11 = *(v16float *)(v1 + v10);
      v16float v12 = min(v9, v11);
      v9 = v12;
    }
  v8 = v9;
  v16float v13 = shift_bytes(v8, undef_v16float(), 32);
  v16float v14 = min(v8, v13);
  v16float v15 = shift_bytes(v14, undef_v16float(), 16);
  v16float v16 = min(v14, v15);
  v16float v17 = shift_bytes(v16, undef_v16float(), 8);
  v16float v18 = min(v16, v17);
  v16float v19 = shift_bytes(v18, undef_v16float(), 4);
  v16float v20 = min(v18, v19);
  float v21 = extract_elem(v20, v3);
  *(float *)v2 = v21;
  return;
}
