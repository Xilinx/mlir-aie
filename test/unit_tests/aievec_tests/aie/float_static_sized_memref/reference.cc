#include "float.h"

void reference(float *restrict img_in, float *restrict kernel_coeff,
               float *restrict img_out) {
  v8float *restrict ptr_img_buffer = (v8float *)img_in;
  v8float *restrict ptr_img_out = (v8float *)img_out;

  v16float data_buf;
  v8float acc;

  v8float *restrict ptr_coeff_buffer = (v8float *)kernel_coeff;
  v8float kernel_vec0 = *(ptr_coeff_buffer)++; // 1st 8 kernel values (0 .. 7)
  v8float kernel_vec1 = *(ptr_coeff_buffer);   // last kernel value (8)

  v8float *restrict ptr0 = ptr_img_buffer;
  v8float *restrict ptr1 = ptr_img_buffer + 1 * 272 / PARALLEL_FACTOR_FLOAT;
  v8float *restrict ptr2 = ptr_img_buffer + 2 * 272 / PARALLEL_FACTOR_FLOAT;
  v8float *restrict ptr_out = ptr_img_out;

  for (int i = 0; i < 16; i++)
    chess_prepare_for_pipelining chess_loop_range(16, 16) {
      for (int j = 0; j < 256;
           j += PARALLEL_FACTOR_FLOAT) // 8x samples per loop
        chess_prepare_for_pipelining chess_loop_range(32, 32) {
          // 1st row
          data_buf = upd_w(data_buf, 0, *(ptr0++)); // r1:00++07|_________
          acc = fpmul(data_buf, 0, 0x76543210, kernel_vec0, 0,
                      0);                         // kernel 0 (r1:00..07)
          data_buf = upd_w(data_buf, 1, *(ptr0)); // r1:00..07|r1:08++15
          acc = fpmac(acc, data_buf, 1, 0x76543210, kernel_vec0, 1,
                      0); // kernel 1 (r1:01..08)
          acc = fpmac(acc, data_buf, 2, 0x76543210, kernel_vec0, 2,
                      0); // kernel 2 (r1:02..09)

          // 2nd row
          data_buf = upd_w(data_buf, 0, *(ptr1++)); // r2:00++07|_________
          acc = fpmac(acc, data_buf, 0, 0x76543210, kernel_vec0, 3,
                      0);                         // kernel 3 (r2:00..07)
          data_buf = upd_w(data_buf, 1, *(ptr1)); // r2:00..07|r2:08++15
          acc = fpmac(acc, data_buf, 1, 0x76543210, kernel_vec0, 4,
                      0); // kernel 4 (r2:01..08)
          acc = fpmac(acc, data_buf, 2, 0x76543210, kernel_vec0, 5,
                      0); // kernel 5 (r2:02..09)

          // 3rd row
          data_buf = upd_w(data_buf, 0, *(ptr2++)); // r3:00++07|_________
          acc = fpmac(acc, data_buf, 0, 0x76543210, kernel_vec0, 6,
                      0);                         // kernel 6 (r3:00..07)
          data_buf = upd_w(data_buf, 1, *(ptr2)); // r3:00..07|r3:08++15
          acc = fpmac(acc, data_buf, 1, 0x76543210, kernel_vec0, 7,
                      0); // kernel 7 (r3:01..08)
          acc = fpmac(acc, data_buf, 2, 0x76543210, kernel_vec1, 0,
                      0); // kernel 8 (r3:02..09)

          *(ptr_out++) = acc; // Write compute pixel to output buffer
        }
      // Increment row pointers to next row
      ptr0 = ptr_img_buffer + (i + 1) * 272 / PARALLEL_FACTOR_FLOAT;
      ptr1 = ptr_img_buffer + (i + 2) * 272 / PARALLEL_FACTOR_FLOAT;
      ptr2 = ptr_img_buffer + (i + 3) * 272 / PARALLEL_FACTOR_FLOAT;
    }
}
