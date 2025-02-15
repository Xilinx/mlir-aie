
void conv2d(float img_in[18][272], float kernel_coeff[3][3],
            float img_out[16][256]) {
#pragma scop
  for (int r = 0; r < 16; r++)
    for (int c = 0; c < 256; c++) {
      float acc = 0;
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
          acc += img_in[r + i][c + j] * kernel_coeff[i][j];
        }
      img_out[r][c] = acc;
    }
#pragma endscop
}
