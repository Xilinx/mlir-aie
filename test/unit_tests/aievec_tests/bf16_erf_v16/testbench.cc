#include "../common/testbench.h"
#include "defines.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdlib.h>
#include <string>

void dut(bfloat16 *restrict in0, bfloat16 *restrict out0);
void dut_ref(bfloat16 *in0, bfloat16 *out0);

alignas(32) bfloat16 g_in0[IN0_SIZE];
alignas(32) bfloat16 g_out0[OUT0_SIZE];
alignas(32) bfloat16 g_out0Ref[OUT0_SIZE];

int main(int argc, char *argv[]) {
  std::string dataDir(TO_STR(DATA_DIR));
  srand(10);
  std::generate(g_in0, g_in0 + IN0_SIZE,
                [&]() { return random_bfloat16(-4, 4, 3); });

  writeData(g_in0, IN0_SIZE, dataDir + "/in0.txt");

  chess_memory_fence();
  auto cyclesBegin = chess_cycle_count();
  dut(g_in0, g_out0);
  auto cyclesEnd = chess_cycle_count();
  chess_memory_fence();

  auto cycleCount = (int)(cyclesEnd - cyclesBegin);
  reportCycleCount(cycleCount, dataDir + "/cycle_count.txt");

  writeData(g_out0, OUT0_SIZE, dataDir + "/out0.txt");

  return 1;
}
