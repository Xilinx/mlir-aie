void dut(float * restrict v1, float * restrict v2) {
  v16float v3 = broadcast_to_v16float((float)340282346638528859811704183484516925440.000000);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  v16float v7;
  v16float v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
  chess_prepare_for_pipelining
  chess_loop_range(64, 64)
  {
    v16float v10 = *(v16float *)(v1 + v9);
    v16float v11 = min(v8, v10);
    v8 = v11;
  }
  v7 = v8;
  v16float v12 = shift_bytes(v7, undef_v16float(), 32);
  v16float v13 = min(v7, v12);
  v16float v14 = shift_bytes(v13, undef_v16float(), 16);
  v16float v15 = min(v13, v14);
  v16float v16 = shift_bytes(v15, undef_v16float(), 8);
  v16float v17 = min(v15, v16);
  v16float v18 = shift_bytes(v17, undef_v16float(), 4);
  v16float v19 = min(v17, v18);
  float v20 = extract_elem(v19, 0);
  *(float *)v2 = v20;
  return;
}


