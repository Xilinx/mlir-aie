#define __AIENGINE__ 1
#define NOCPP

#define h1 32
#define w1 32
#define w2 32

#include "cardano.h"
#include <stdio.h>

extern "C" {
void extern_kernel(int32_t *A, int32_t *B, int32_t *acc, int32_t *C) {
  input_window_int32 matAImpl = {(window_datatype *)A, (window_datatype *)A,
                                 (window_datatype *)A, 4096, 1024};
  input_window_int32 matBImpl = {(window_datatype *)B, (window_datatype *)B,
                                 (window_datatype *)B, 4096, 1024};
  input_window_int32 accImpl = {(window_datatype *)acc, (window_datatype *)acc,
                                (window_datatype *)acc, 4096, 1024};
  output_window_int32 matCImpl = {(window_datatype *)C, (window_datatype *)C,
                                  (window_datatype *)C, 4096, 1024};

  input_window_int32 *matA = &matAImpl;
  input_window_int32 *matB = &matBImpl;
  input_window_int32 *acc_in = &accImpl;
  output_window_int32 *matC = &matCImpl;

  v8acc80 acc0 = null_v8acc80(); // For first output column
  v8acc80 acc1 = null_v8acc80(); // For second output column
  v16int32 buf_matB = undef_v16int32();
  v16int32 buf_matA = undef_v16int32();

  buf_matB = upd_v(buf_matB, 0, window_read_v4(matB));
  window_incr(matB, w1);
  buf_matB = upd_v(buf_matB, 1, window_read_v4(matB));
  window_decr(matB, w1 - 4);

  buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
  window_incr(matA, h1);

  // v8acc80 acc0=null_v8acc80();//For first output column
  // v8acc80 acc1=null_v8acc80();//For second output column
  for (unsigned int i = 0; i < 4; i++)
    chess_prepare_for_pipelining chess_loop_range(4, ) {

      for (unsigned int k = 0; k < 16; k++)
        chess_prepare_for_pipelining chess_loop_range(16, ) {
          int jump = h1;
          if (k == 15) {
            jump = h1 + 8;
          } else {
            jump = h1;
          }
          acc0 = lups(window_read_v8(acc_in), 0);
          window_incr(acc_in, h1);
          acc1 = lups(window_read_v8(acc_in), 0);
          window_incr(acc_in, jump);
          for (unsigned int j = 0; j < 3; j++)
            chess_prepare_for_pipelining chess_loop_range(3, ) {
              acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 0,
                           0x0);
              buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
              window_incr(matA, h1);
              buf_matB = upd_v(buf_matB, 2, window_read_v4(matB));
              window_incr(matB, w1);
              acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 4,
                           0x0);
              buf_matB = upd_v(buf_matB, 3, window_read_v4(matB));
              window_decr(matB, w1 - 4);

              acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 1,
                           0x0);
              buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 5,
                           0x0);

              acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 2,
                           0x0);
              buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 6,
                           0x0);

              acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 3,
                           0x0);
              buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 7,
                           0x0);

              ////////////////////////////////////////////////////////////////////////
              acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 0,
                           0x0);
              buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 4,
                           0x0);

              acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 1,
                           0x0);
              buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 5,
                           0x0);

              acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 2,
                           0x0);
              buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
              window_incr(matA, h1);
              acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 6,
                           0x0);

              acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 3,
                           0x0);
              // window_write(matC,srs(acc0,0));
              // window_incr(matC,h1);
              buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
              window_incr(matA, h1);
              buf_matB = upd_v(buf_matB, 0, window_read_v4(matB));
              window_incr(matB, w1);
              acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 7,
                           0x0);
              // window_write(matC,srs(acc1,0));
              // window_incr(matC,h1+8);
              buf_matB = upd_v(buf_matB, 1, window_read_v4(matB));
              window_decr(matB, w1 - 4);
            }
          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 0, 0x0);
          buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
          window_incr(matA, h1);
          buf_matB = upd_v(buf_matB, 2, window_read_v4(matB));
          window_incr(matB, w1);
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 4, 0x0);
          buf_matB = upd_v(buf_matB, 3, window_read_v4(matB));
          window_incr(matB, 4);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 1, 0x0);
          buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
          window_incr(matA, h1);
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 5, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 2, 0x0);
          buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
          window_incr(matA, h1);
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 6, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 3, 0x0);
          buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
          window_incr(matA, h1);
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 7, 0x0);

          ////////////////////////////////////////////////////////////////////////
          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 0, 0x0);
          buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
          window_incr(matA, h1);
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 4, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 1, 0x0);
          buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
          window_incr(matA, h1);
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 5, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 2, 0x0);
          buf_matA = upd_w(buf_matA, 1, window_read_v8(matA));
          window_incr(matA, jump);
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 6, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 3, 0x0);
          window_write(matC, srs(acc0, 0));
          window_incr(matC, h1);
          buf_matA = upd_w(buf_matA, 0, window_read_v8(matA));
          window_incr(matA, h1);
          buf_matB = upd_v(buf_matB, 0, window_read_v4(matB));
          window_incr(matB, w1);
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 7, 0x0);
          window_write(matC, srs(acc1, 0));
          window_incr(matC, jump);
          buf_matB = upd_v(buf_matB, 1, window_read_v4(matB));
          window_decr(matB, w1 - 4);
        }
    }
}
}