void dut(float *restrict v1, float *restrict v2) {
  int32_t v3 = 0;
  v16float v4 = broadcast_zero_float();
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 16;
  v16float v8;
  v16float v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16float v11 = *(v16float *)(v1 + v10);
      v16accfloat v12 = v16accfloat(v9);
      v16accfloat v13 = v16accfloat(v11);
      v16accfloat v14 = add(v12, v13);
      v16float v15 = v16float(v14);
      v9 = v15;
    }
  v8 = v9;
  v16float v16 = shift_bytes(v8, undef_v16float(), 32);
  v16accfloat v17 = v16accfloat(v8);
  v16accfloat v18 = v16accfloat(v16);
  v16accfloat v19 = add(v17, v18);
  v16float v20 = v16float(v19);
  v16float v21 = shift_bytes(v20, undef_v16float(), 16);
  v16accfloat v22 = v16accfloat(v20);
  v16accfloat v23 = v16accfloat(v21);
  v16accfloat v24 = add(v22, v23);
  v16float v25 = v16float(v24);
  v16float v26 = shift_bytes(v25, undef_v16float(), 8);
  v16accfloat v27 = v16accfloat(v25);
  v16accfloat v28 = v16accfloat(v26);
  v16accfloat v29 = add(v27, v28);
  v16float v30 = v16float(v29);
  v16float v31 = shift_bytes(v30, undef_v16float(), 4);
  v16accfloat v32 = v16accfloat(v30);
  v16accfloat v33 = v16accfloat(v31);
  v16accfloat v34 = add(v32, v33);
  v16float v35 = v16float(v34);
  float v36 = extract_elem(v35, v3);
  *(float *)v2 = v36;
  return;
}
