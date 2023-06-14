void dut(float *restrict v1, float *restrict v2) {
  v16float v3 = broadcast_to_v16float((float)0.000000);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  v16float v7;
  v16float v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16float v10 = *(v16float *)(v1 + v9);
      v16accfloat v11 = v16accfloat(v8);
      v16accfloat v12 = v16accfloat(v10);
      v16accfloat v13 = add(v11, v12);
      v16float v14 = v16float(v13);
      v8 = v14;
    }
  v7 = v8;
  v16float v15 = shift_bytes(v7, undef_v16float(), 32);
  v16accfloat v16 = v16accfloat(v7);
  v16accfloat v17 = v16accfloat(v15);
  v16accfloat v18 = add(v16, v17);
  v16float v19 = v16float(v18);
  v16accfloat v20 = v16accfloat(v19);
  v16accfloat v21 = add(v20, v17);
  v16float v22 = v16float(v21);
  v16accfloat v23 = v16accfloat(v22);
  v16accfloat v24 = add(v23, v17);
  v16float v25 = v16float(v24);
  v16accfloat v26 = v16accfloat(v25);
  v16accfloat v27 = add(v26, v17);
  v16float v28 = v16float(v27);
  float v29 = extract_elem(v28, 0);
  *(float *)v2 = v29;
  return;
}
