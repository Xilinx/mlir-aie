#include "../common/testbench.h"
#include "defines.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

void dut(int32_t *restrict in0, int8_t *restrict out0);
void dut_ref(int32_t *in0, int8_t *out0);

alignas(32) int32_t g_in0[IN0_SIZE];
alignas(32) int8_t g_out0[OUT0_SIZE];
alignas(32) int8_t g_out0Ref[OUT0_SIZE];

int main(int argc, char *argv[]) {
  std::string dataDir(TO_STR(DATA_DIR));
  srand(10);
  std::generate(g_in0, g_in0 + IN0_SIZE,
                [&]() { return random_integer<int32_t>(); });

  writeData(g_in0, IN0_SIZE, dataDir + "/in0.txt");

  chess_memory_fence();
  auto cyclesBegin = chess_cycle_count();
  dut(g_in0, g_out0);
  auto cyclesEnd = chess_cycle_count();
  chess_memory_fence();

  auto cycleCount = (int)(cyclesEnd - cyclesBegin);
  reportCycleCount(cycleCount, dataDir + "/cycle_count.txt");

  writeData(g_out0, OUT0_SIZE, dataDir + "/out0.txt");
  cyclesBegin = chess_cycle_count();
  dut_ref(g_in0, g_out0Ref);
  cyclesEnd = chess_cycle_count();
  chess_memory_fence();
  cycleCount = (int)(cyclesEnd - cyclesBegin);
  reportCycleCount(cycleCount, dataDir + "/cycle_count.txt");
  writeData(g_out0Ref, OUT0_SIZE, dataDir + "/out0_ref.txt");

  bool ok = true;
  ok &= checkData(g_out0, g_out0Ref, OUT0_SIZE, 0, 1e-2, 2.4e-2);

  if (ok)
    printf("TEST PASSED\n");
  else
    printf("TEST FAILED\n");

  return ok ? 0 : 1;
}

void dut_ref(int32_t *in0, int8_t *out0) {
  for (unsigned k = 0; k < OUT0_SIZE; k += 1) {
    out0[k] = (int8_t)in0[k];
  }
}
