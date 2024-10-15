#include "i32xi32.h"

// From lab11
void reference(int32_t *restrict img_in, int32_t *restrict kernel_coeff,
               int32_t *restrict img_out, int image_width, int image_height,
               int stride) {
  v8int32 *restrict ptr_img_buffer = (v8int32 *)img_in;
  v8int32 *restrict ptr_img_out = (v8int32 *)img_out;

  v16int32 data_buf;
  v8int32 data_out;
  v8acc80 acc;

  v8int32 *restrict ptr_coeff_buffer = (v8int32 *)kernel_coeff;
  v8int32 kernel_vec0 = *(ptr_coeff_buffer)++; // 1st 8 kernel values (0 .. 7)
  v8int32 kernel_vec1 = *(ptr_coeff_buffer);   // last kernel value (8)

  v8int32 *restrict ptr0 = ptr_img_buffer;
  v8int32 *restrict ptr1 = ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_32b;
  v8int32 *restrict ptr2 = ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_32b;
  v8int32 *restrict ptr_out = ptr_img_out;

  // 3x3 kernel positions
  //
  // 0 1 2
  // 3 4 5
  // 6 7 8

  for (int i = 0; i < image_height; i++) {
    for (int j = 0; j < image_width;
         j += PARALLEL_FACTOR_32b) // 8x samples per loop
      chess_prepare_for_pipelining {
        // 1st row
        data_buf = upd_w(data_buf, 0, *(ptr0++)); // r1:00++07|_________
        acc = lmul8(data_buf, 0, 0x76543210, kernel_vec0, 0,
                    0);                         // kernel 0 (r1:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr0)); // r1:00..07|r1:08++15
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 1,
                    0); // kernel 1 (r1:01..08)
        acc = lmac8(acc, data_buf, 2, 0x76543210, kernel_vec0, 2,
                    0); // kernel 2 (r1:02..09)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++)); // r2:00++07|_________
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 3,
                    0);                         // kernel 3 (r2:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr1)); // r2:00..07|r2:08++15
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 4,
                    0); // kernel 4 (r2:01..08)
        acc = lmac8(acc, data_buf, 2, 0x76543210, kernel_vec0, 5,
                    0); // kernel 5 (r2:02..09)

        // 3rd row
        data_buf = upd_w(data_buf, 0, *(ptr2++)); // r3:00++07|_________
        acc = lmac8(acc, data_buf, 0, 0x76543210, kernel_vec0, 6,
                    0);                         // kernel 6 (r3:00..07)
        data_buf = upd_w(data_buf, 1, *(ptr2)); // r3:00..07|r3:08++15
        acc = lmac8(acc, data_buf, 1, 0x76543210, kernel_vec0, 7,
                    0); // kernel 7 (r3:01..08)
        acc = lmac8(acc, data_buf, 2, 0x76543210, kernel_vec1, 0,
                    0); // kernel 8 (r3:02..09)

        data_out = srs(acc, SRS_SHIFT); // output 00..07
        *(ptr_out++) = data_out;        // Write compute pixel to output buffer
      }
    // Increment row pointers to next row
    ptr0 = ptr_img_buffer + (i + 1) * stride / PARALLEL_FACTOR_32b;
    ptr1 = ptr_img_buffer + (i + 2) * stride / PARALLEL_FACTOR_32b;
    ptr2 = ptr_img_buffer + (i + 3) * stride / PARALLEL_FACTOR_32b;
  }
}
