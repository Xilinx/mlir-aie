#include "testbench.h"
#include <math.h>

using namespace std;

#define MAT_A_SIZE 4096
#define MAT_B_SIZE 4096
#define MAT_C_SIZE 4096
#define N_SIZE 64
#define M_SIZE 64
#define K_SIZE 64

int16_t mat_a_data[MAT_A_SIZE];
int16_t mat_b_data[MAT_B_SIZE];
int16_t mat_c_data[MAT_C_SIZE];
int16_t ref_c_data[MAT_C_SIZE];

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
      int val;
      fscanf(fpin_a, "%d", &val);
      mat_a_data[index++] = val;
    }
  }

  // Read in matrix_b to local memory
  index = 0;
  for (k = 0; k < K_SIZE; k++) {
    for (j = 0; j < M_SIZE; j++) {
      int val;
      fscanf(fpin_b, "%d", &val);
      mat_b_data[index++] = val;
    }
  }

  // Initialize matrix_c to local memory
  index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < M_SIZE; j++) {
      mat_c_data[index++] = 0;
    }
  }
  /*
    index = 0;
    for (i = 0; i < N_SIZE; i++) {
      for (j = 0; j < M_SIZE; j++) {
        ref_c_data[index++] = 0;
      }
    }
  */

  // Compute matrix multiplication
  // reference(mat_a_data, mat_b_data, mat_c_data);
  printf("Running MATMUL...\n\n");

  auto cyclesBegin = chess_cycle_count();
  matmul(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesEnd = chess_cycle_count();
  printf("Cycle count: %d\n", (int)(cyclesEnd - cyclesBegin));

  printf("Finish MATMUL!\n\n");

  index = 0;
  for (i = 0; i < N_SIZE; i++) {
    for (j = 0; j < M_SIZE; j++) {
      int val;
      fscanf(fpin_c, "%d", &val);
      ref_c_data[index++] = val;
    }
  }

  /*
    for (i = 0; i < N_SIZE; i++) {
      for (j = 0; j < M_SIZE; j++) {
        for (k = 0; k < K_SIZE; k++) {
          ref_c_data[i * M_SIZE + j] += mat_a_data[i * K_SIZE + k] *
    mat_b_data[k * M_SIZE + j];
        }
        printf("%d\n", ref_c_data[i * M_SIZE + j]);
      }
    }
  */

  // Compare results with reference result
  // NOTE: There will be some rounding errors in results so we accept absolute
  // value differences < 5

  printf("Compare the results\n\n");
  int errors = 0;
  int max_error = 0;
  int absErrorDiff = 0;
  for (int i = 0; i < MAT_C_SIZE; i++) {
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

  fclose(fpin_a);
  fclose(fpin_b);
  fclose(fpin_c);
  return 0;
}
