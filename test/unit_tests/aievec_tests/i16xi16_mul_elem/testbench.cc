#include "testbench.h"
#include <math.h>

using namespace std;

#define MAT_SIZE 1024

alignas(32) int16_t mat_a_data[MAT_SIZE];
alignas(32) int16_t mat_b_data[MAT_SIZE];
alignas(32) int16_t mat_c_data[MAT_SIZE];
alignas(32) int16_t ref_c_data[MAT_SIZE];

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

  // Compute the reference result
  for (i = 0; i < MAT_SIZE; i++) {
    ref_c_data[i] = mat_a_data[i] * mat_b_data[i];
  }

  printf("Running MUL_ELEM...\n\n");
  auto cyclesBegin = chess_cycle_count();
  mul_elem(mat_a_data, mat_b_data, mat_c_data);
  auto cyclesEnd = chess_cycle_count();
  printf("Cycle count: %d\n", (int)(cyclesEnd - cyclesBegin));

  printf("Finish MUL_ELEM!\n\n");

  int errors = 0;
  // Compare results with reference result
  printf("Compare the results\n\n");
  for (int i = 0; i < MAT_SIZE; i++) {
    if (mat_c_data[i] != ref_c_data[i]) {
      errors++;
    }
  }

  if (errors == 0) {
    printf("PASSED.\n\n");
  } else {
    printf("FAIL. Number of erros = %d\n\n", errors);
  }

  return 0;
}
