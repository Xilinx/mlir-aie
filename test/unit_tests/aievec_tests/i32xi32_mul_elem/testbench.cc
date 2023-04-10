#include "testbench.h"
#include <math.h>

using namespace std;

#define MAT_SIZE 1024

alignas(32) int32_t mat_a_data[MAT_SIZE];
alignas(32) int32_t mat_b_data[MAT_SIZE];
alignas(32) int32_t mat_c_data[MAT_SIZE];
alignas(32) int32_t ref_c_data[MAT_SIZE];

#ifndef __chess__
int chess_cycle_count() { return 0; }
#endif

int main() {
  int i;
  // Read in matrix_a to local memory
  for (i = 0; i < MAT_SIZE; i++) {
    mat_a_data[i] = i % 100;
  }

  // Read in matrix_b to local memory
  for (i = 0; i < MAT_SIZE; i++) {
    mat_b_data[i] = (i + 11) % 100;
  }

  for (i = 0; i < MAT_SIZE; i++) {
    ref_c_data[i] = mat_a_data[i] * mat_b_data[i];
  }

  // Compute matrix multiplication
  // reference(mat_a_data, mat_b_data, mat_c_data);
  printf("Running MATMUL...\n\n");
  auto cyclesBegin = chess_cycle_count();
  mul_elem(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesEnd = chess_cycle_count();
  printf("Cycle count: %d\n", (int)(cyclesEnd - cyclesBegin));

  printf("Finish MATMUL!\n\n");

  // Compare results with reference result
  // NOTE: There will be some rounding errors in results so we accept absolute
  // value differences < 5

  printf("Compare the results\n\n");
  int errors = 0;
  int max_error = 0;
  int absErrorDiff = 0;
  for (int i = 0; i < MAT_SIZE; i++) {
    if (mat_c_data[i] != ref_c_data[i]) {
      printf("%d got %d expected %d\n", i, mat_c_data[i], ref_c_data[i]);
      absErrorDiff = fabs(mat_c_data[i] - ref_c_data[i]);
      if (absErrorDiff >= 5)
        printf("Delta found: Index %d is %d and should be %d\n", i,
               mat_c_data[i], ref_c_data[i]);
      if (absErrorDiff > max_error)
        printf("max found in index: %d\n", i);
      max_error = max_error < absErrorDiff ? absErrorDiff : max_error;
      errors++;
    }
  }

  if (errors == 0 || max_error < 5) {
    printf("PASSED, Max delta: %d, pixel intensity\n\n", max_error);
  } else {
    printf("FAIL. Number of deltas = %d, Max delta: %d, pixel intensity\n\n",
           errors, max_error);
  }

  return 0;
}
