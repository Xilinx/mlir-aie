#include "testbench.h"
#include <math.h>

using namespace std;

#define MAT_A_SIZE 2048
#define MAT_B_SIZE 2048
#define MAT_C_SIZE 4096
#define N_SIZE 64
#define M_SIZE 64
#define K_SIZE 32

bfloat16 mat_a_data[MAT_A_SIZE];
bfloat16 mat_b_data[MAT_B_SIZE];
float mat_c_data[MAT_C_SIZE];
float ref_c_data[MAT_C_SIZE];

#define INPUT_A_FILE "data/matrix_a_test.txt"
#define INPUT_B_FILE "data/matrix_b_test.txt"
#define OUTPUT_C_FILE "data/matrix_c_test.txt"

#ifndef __chess__
int chess_cycle_count() { return 0; }
#endif

int main() {
  int i = 0, j = 0, k = 0;

  // Open input and ref file
  FILE *fpin_a = NULL, *fpin_b = NULL, *fpin_c = NULL;
  fpin_a = fopen(INPUT_A_FILE, "r");
  if (fpin_a == NULL) {
    printf("failure opening file %s for reading\n", INPUT_A_FILE);
    return -1;
  }

  fpin_b = fopen(INPUT_B_FILE, "r");
  if (fpin_b == NULL) {
    printf("failure opening file %s for reading\n", INPUT_B_FILE);
    return -1;
  }

  fpin_c = fopen(OUTPUT_C_FILE, "r");
  if (fpin_c == NULL) {
    printf("failure opening file %s for reading\n", OUTPUT_C_FILE);
    return -1;
  }

  // Print matrix size
  printf("N: %d, M: %d, K: %d\n", N_SIZE, M_SIZE, K_SIZE);

  // Read in matrix_a to local memory
  int index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (k = 0; k < K_SIZE; k++) {
      float val;
      fscanf(fpin_a, "%f", &val);
      int32_t ival = *reinterpret_cast<int32_t *>(&val);
      int16_t bfval = (ival & 0xFFFF0000) >> 16;
      mat_a_data[index++] = *reinterpret_cast<bfloat16 *>(&bfval);
    }
  }

  // Read in matrix_b to local memory
  index = 0;
  for (k = 0; k < K_SIZE; k++) {
    for (j = 0; j < M_SIZE; j++) {
      float val;
      fscanf(fpin_b, "%f", &val);
      int32_t ival = *reinterpret_cast<int32_t *>(&val);
      int16_t bfval = (ival & 0xFFFF0000) >> 16;
      mat_b_data[index++] = *reinterpret_cast<bfloat16 *>(&bfval);
    }
  }

  // Initialize matrix_c to local memory
  index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < M_SIZE; j++) {
      mat_c_data[index++] = 0.f;
    }
  }

  // Compute matrix multiplication
  // reference(mat_a_data, mat_b_data, mat_c_data);
  printf("Running MATMUL...\n\n");

  auto cyclesBegin = chess_cycle_count();
  gemm_64x32x64_bf16_packed_4x8x4(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesEnd = chess_cycle_count();
  printf("Cycle count: %d\n", (int)(cyclesEnd - cyclesBegin));

  printf("Finish MATMUL!\n\n");

  index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < M_SIZE; j++) {
      float val;
      fscanf(fpin_c, "%f", &val);
      ref_c_data[index++] = val;
    }
  }

  // Compare results with reference result
  // NOTE: There will be some rounding errors in results so we accept absolute
  // value differences < 5

  printf("Compare the results\n\n");
  int errors = 0;
  float max_error = 0.f;
  float absErrorDiff = 0.f;
  for (int i = 0; i < MAT_C_SIZE; i++) {
    if (mat_c_data[i] != ref_c_data[i]) {
      printf("%d got %f expected %f\n", i, mat_c_data[i], ref_c_data[i]);
      absErrorDiff = fabs(mat_c_data[i] - ref_c_data[i]);
      if (absErrorDiff >= 5.f)
        printf("Delta found: Index %d is %f and should be %f\n", i,
               mat_c_data[i], ref_c_data[i]);
      if (absErrorDiff > max_error)
        printf("max found in index: %d\n", i);
      max_error = max_error < absErrorDiff ? absErrorDiff : max_error;
      errors++;
    }
  }

  if (errors == 0 || max_error < 5.f) {
    printf("PASSED, Max delta: %f, pixel intensity\n\n", max_error);
  } else {
    printf("FAIL. Number of deltas = %d, Max delta: %f, pixel intensity\n\n",
           errors, max_error);
  }

  fclose(fpin_a);
  fclose(fpin_b);
  fclose(fpin_c);
  return 0;
}
