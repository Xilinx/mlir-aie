void dut(float *restrict v1, float *restrict v2) {
  size_t v3 = 0;
  size_t v4 = 1024;
  size_t v5 = 16;
  for (size_t v6 = v3; v6 < v4; v6 += v5)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16float v7 = *(v16float *)(v1 + v6);
      v16accfloat v8 = v16accfloat(v7);
      v16accfloat v9 = neg(v8);
      v16float v10 = v16float(v9);
      *(v16float *)(v2 + v6) = v10;
    }
  return;
}
