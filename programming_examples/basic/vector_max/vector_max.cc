#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

void vector(int32_t *restrict v1, int32_t *restrict v2) {
  int32_t v3 = 4;
  int32_t v4 = 8;
  int32_t v5 = 16;
  int32_t v6 = 32;
  int32_t v7 = 0;
  v16int32 v8 = broadcast_to_v16int32((int32_t)-2147483648);
  size_t v9 = 0;
  size_t v10 = 1024;
  size_t v11 = 16;
  v16int32 v12;
  v16int32 v13 = v8;
  for (size_t v14 = v9; v14 < v10; v14 += v11)
    chess_prepare_for_pipelining chess_loop_range(64, 64) {
      v16int32 v15 = *(v16int32 *)(v1 + v14);
      v16int32 v16 = max(v13, v15);
      v13 = v16;
    }
  v12 = v13;
  v16int32 v17 = shift_bytes(v12, v12, v6);
  v16int32 v18 = max(v12, v17);
  v16int32 v19 = shift_bytes(v18, v18, v5);
  v16int32 v20 = max(v18, v19);
  v16int32 v21 = shift_bytes(v20, v20, v4);
  v16int32 v22 = max(v20, v21);
  v16int32 v23 = shift_bytes(v22, v22, v3);
  v16int32 v24 = max(v22, v23);
  int32_t v25 = extract_elem(v24, v7);
  *(int32_t *)v2 = v25;
  return;
}

void scalar(int32_t *restrict v1, int32_t *restrict v2) {
  *(int32_t *)v2 = 27; // what could be bigger
  return;
}

extern "C" {

void vector_max(int32_t *a_in, int32_t *c_out) { vector(a_in, c_out); }

void scalar_max(int32_t *a_in, int32_t *c_out) { scalar(a_in, c_out); }

} // extern "C"
