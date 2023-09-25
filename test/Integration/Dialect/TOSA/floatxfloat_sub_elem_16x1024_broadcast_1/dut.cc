// clang-format off
void dut(float * restrict v1, float * restrict v2, float * restrict v3) {
  size_t v4 = 0;
  float * restrict v5 = v2;
  v16float v6 = *(v16float *)(v5 + v4+v4);
  v16float v7 = broadcast_elem(v6, 0);
  v16accfloat v8 = v16accfloat(v7);
  size_t v9 = 0;
  size_t v10 = 16;
  size_t v11 = 1;
  for (size_t v12 = v9; v12 < v10; v12 += v11)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v13 = 0;
    size_t v14 = 1024;
    size_t v15 = 16;
    for (size_t v16 = v13; v16 < v14; v16 += v15)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16float v17 = *(v16float *)(v1 + 1024*v12+v16);
      v16accfloat v18 = v16accfloat(v17);
      v16accfloat v19 = sub(v18, v8);
      v16float v20 = v16float(v19);
      *(v16float *)(v3 + 1024*v12+v16) = v20;
    }
  }
  return;
}
// clang-format on
