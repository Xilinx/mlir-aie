void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  int32_t v3 = 0;
  v16bfloat16 v4 = extract_v16bfloat16(broadcast_zero_bfloat16(), 0);
  size_t v5 = 0;
  size_t v6 = 1024;
  size_t v7 = 16;
  v16bfloat16 v8;
  v16bfloat16 v9 = v4;
  for (size_t v10 = v5; v10 < v6; v10 += v7)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v11 = *(v16bfloat16 *)(v1 + v10);
      v16accfloat v12 = ups_to_v16accfloat(v9);
      v16accfloat v13 = ups_to_v16accfloat(v11);
      v16accfloat v14 = add(v12, v13);
      v16bfloat16 v15 = to_v16bfloat16(v14);
      v9 = v15;
    }
  v8 = v9;
  v16accfloat v16 = ups_to_v16accfloat(v8);
  v16accfloat v17 = shift_bytes(v16, undef_v16accfloat(), 32);
  v16accfloat v18 = add(v16, v17);
  v16accfloat v19 = shift_bytes(v18, undef_v16accfloat(), 16);
  v16accfloat v20 = add(v18, v19);
  v16accfloat v21 = shift_bytes(v20, undef_v16accfloat(), 8);
  v16accfloat v22 = add(v20, v21);
  v16accfloat v23 = shift_bytes(v22, undef_v16accfloat(), 4);
  v16accfloat v24 = add(v22, v23);
  v16bfloat16 v25 = to_v16bfloat16(v24);
  v32bfloat16 v26 = concat(v25, v25);
  bfloat16 v27 = extract_elem(v26, v3);
  *(bfloat16 *)v2 = v27;
  return;
}
