// clang-format off
void dut(float * restrict v1, float * restrict v2, float * restrict v3) {
  size_t v4 = 0;
  float * restrict v5 = v2;
  size_t v6 = 0;
  size_t v7 = 16;
  size_t v8 = 1;
  for (size_t v9 = v6; v9 < v7; v9 += v8)
  chess_prepare_for_pipelining
  chess_loop_range(16, 16)
  {
    size_t v10 = 0;
    size_t v11 = 1024;
    size_t v12 = 16;
    for (size_t v13 = v10; v13 < v11; v13 += v12)
    chess_prepare_for_pipelining
    chess_loop_range(64, 64)
    {
      v16float v14 = *(v16float *)(v1 + 1024*v9+v13);
      v16float v15 = *(v16float *)(v5 + 1024*v4+v13);
      v16accfloat v16 = v16accfloat(v14);
      v16accfloat v17 = v16accfloat(v15);
      v16accfloat v18 = sub(v16, v17);
      v16float v19 = v16float(v18);
      *(v16float *)(v3 + 1024*v9+v13) = v19;
    }
  }
  return;
}
// clang-format on
