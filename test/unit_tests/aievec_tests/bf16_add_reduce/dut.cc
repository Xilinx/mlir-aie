void dut(bfloat16 *restrict v1, bfloat16 *restrict v2) {
  v16bfloat16 v3 =
      extract_v16bfloat16(broadcast_to_v32bfloat16((bfloat16)0.000000), 0);
  size_t v4 = 0;
  size_t v5 = 1024;
  size_t v6 = 16;
  v16bfloat16 v7;
  v16bfloat16 v8 = v3;
  for (size_t v9 = v4; v9 < v5; v9 += v6)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16bfloat16 v10 = *(v16bfloat16 *)(v1 + v9);
      v16accfloat v11 = ups_to_v16accfloat(v8);
      v16accfloat v12 = ups_to_v16accfloat(v10);
      v16accfloat v13 = add(v11, v12);
      v16bfloat16 v14 = to_v16bfloat16(v13);
      v8 = v14;
    }
  v7 = v8;
  v16accfloat v15 = ups_to_v16accfloat(v7);
  v16accfloat v16 = shift_bytes(v15, undef_v16accfloat(), 32);
  v16accfloat v17 = add(v15, v16);
  v16accfloat v18 = shift_bytes(v17, undef_v16accfloat(), 16);
  v16accfloat v19 = add(v17, v18);
  v16accfloat v20 = shift_bytes(v19, undef_v16accfloat(), 8);
  v16accfloat v21 = add(v19, v20);
  v16accfloat v22 = shift_bytes(v21, undef_v16accfloat(), 4);
  v16accfloat v23 = add(v21, v22);
  v16bfloat16 v24 = to_v16bfloat16(v23);
  v32bfloat16 v25 = concat(v24, v24);
  bfloat16 v26 = extract_elem(v25, 0);
  *(bfloat16 *)v2 = v26;
  return;
}
