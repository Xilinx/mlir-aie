#include "i16xi16.h"

void reference(int16_t *restrict img_in, int16_t *restrict kernel_coeff,
               int16_t *restrict img_out, int image_width, int image_height,
               int stride) {
  v16int16 *restrict ptr_img_buffer = (v16int16 *)img_in;
  v16int16 *restrict ptr_img_out = (v16int16 *)img_out;

  v32int16 data_buf;
  v16int16 data_out;
  v16acc48 acc;

  v16int16 *restrict ptr_coeff_buffer = (v16int16 *)kernel_coeff;
  v16int16 kernel_vec0 =
      *(ptr_coeff_buffer)++; // 1st 16 kernel values (0 .. 15)

  v16int16 *restrict ptr0 = ptr_img_buffer;
  v16int16 *restrict ptr1 = ptr_img_buffer + 1 * stride / PARALLEL_FACTOR_16b;
  v16int16 *restrict ptr2 = ptr_img_buffer + 2 * stride / PARALLEL_FACTOR_16b;
  v16int16 *restrict ptr_out = ptr_img_out;

  // 3x3 kernel positions
  //
  // 0 1 2
  // 3 4 5
  // 6 7 8

  for (int i = 0; i < image_height; i++) {
    for (int j = 0; j < image_width;
         j += PARALLEL_FACTOR_16b) // 16x samples per loop
      chess_prepare_for_pipelining {
        // 1st row
        data_buf = upd_w(data_buf, 0, *(ptr0++)); // r1:00++15|_________
        data_buf = upd_w(data_buf, 1, *(ptr0));   // r1:00..15|r1:16++31
        acc = mul16(data_buf, 0, 0x03020100, 0x07060504, 0x2110, kernel_vec0, 0,
                    0, 0, 1); // kernel 0, 1 (r1:00..16)
        acc = mac16(acc, data_buf, 0, 0x03020100, 0x07060504, 0x3322,
                    kernel_vec0, 2, 0, 0, 1); // kernel 2    (r1:02..17)

        // 2nd row
        data_buf = upd_w(data_buf, 0, *(ptr1++)); // r2:00++15|________
        data_buf = upd_w(data_buf, 1, *(ptr1));   // r2:00..15|r2:16++31
        acc = mac16(acc, data_buf, 0, 0x03020100, 0x07060504, 0x2110,
                    kernel_vec0, 4, 0, 0, 1); // kernel 3, 4 (r2:00..16)
        acc = mac16(acc, data_buf, 0, 0x03020100, 0x07060504, 0x3322,
                    kernel_vec0, 6, 0, 0, 1); // kernel 5    (r2:02..17)

        // 3rd row
        data_buf = upd_w(data_buf, 0, *(ptr2++)); // r3:00++15|________
        data_buf = upd_w(data_buf, 1, *(ptr2));   // r3:00..15|r2:16++31
        acc = mac16(acc, data_buf, 0, 0x03020100, 0x07060504, 0x2110,
                    kernel_vec0, 8, 0, 0, 1); // kernel 6, 7 (r3:00..16)
        acc = mac16(acc, data_buf, 0, 0x03020100, 0x07060504, 0x3322,
                    kernel_vec0, 10, 0, 0, 1); // kernel 8    (r3:02..17)

        // Store result
        data_out = srs(acc, SRS_SHIFT);
        *(ptr_out++) = data_out;
      }
    // Increment row pointers to next row
    ptr0 = ptr_img_buffer + (i + 1) * stride / PARALLEL_FACTOR_16b;
    ptr1 = ptr_img_buffer + (i + 2) * stride / PARALLEL_FACTOR_16b;
    ptr2 = ptr_img_buffer + (i + 3) * stride / PARALLEL_FACTOR_16b;
  }
}
