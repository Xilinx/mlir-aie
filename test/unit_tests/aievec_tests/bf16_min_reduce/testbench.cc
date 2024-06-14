#include "../common/testbench.h"
#include "defines.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <float.h>

#ifdef TO_LLVM
extern "C" {
#endif
void dut(bfloat16 *restrict in0, bfloat16 *restrict out0);
#ifdef TO_LLVM
}
#endif

void dut_ref(bfloat16 *in0, bfloat16 *out0);

alignas(32) bfloat16 g_in0[IN0_SIZE];
alignas(32) bfloat16 g_out0[OUT0_SIZE];
alignas(32) bfloat16 g_out0Ref[OUT0_SIZE];

int main(int argc, char *argv[]) {
  // XXX Figure out how to use argv with xca_udm_dbg --aiearch aie-ml -A
  std::string dataDir(TO_STR(DATA_DIR));
  srand(10);
  std::generate(g_in0, g_in0 + IN0_SIZE,
                [&]() { return random_bfloat16(-4, 1, 3); });

  writeData(g_in0, IN0_SIZE, dataDir + "/in0.txt");

  chess_memory_fence();
  auto cyclesBegin = chess_cycle_count();
  dut(g_in0, g_out0);
  auto cyclesEnd = chess_cycle_count();
  chess_memory_fence();

  auto cycleCount = (int)(cyclesEnd - cyclesBegin);
  reportCycleCount(cycleCount, dataDir + "/cycle_count.txt");

  writeData(g_out0, OUT0_SIZE, dataDir + "/out0.txt");

  dut_ref(g_in0, g_out0Ref);
  writeData(g_out0Ref, OUT0_SIZE, dataDir + "/out0_ref.txt");

  bool ok = true;
  ok &= checkData(g_out0, g_out0Ref, OUT0_SIZE, 0, 1e-2, 1e-2);

  if (ok)
    printf("TEST PASSED\n");
  else
    printf("TEST FAILED\n");

  return ok ? 0 : 1;
}

void dut_ref(bfloat16 *in0, bfloat16 *out0) {
  bfloat16 minx = bfloat16(0x1.FEp+127f);
  for (unsigned k = 0; k < IN0_SIZE; k += 1) {
    minx = std::min(minx, in0[k]);
  }
  *out0 = minx;
}
