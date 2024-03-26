#include "../common/testbench.h"
#include "defines.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef TO_CPP
void dut(int16_t *restrict in0, int16_t *restrict in1, int16_t *restrict out0);
#elif TO_LLVM
extern "C" {
void dut(int16_t *in0_allocated, int16_t *in0_aligned, int64_t in0_offset,
         int64_t in0_sizes_0, int64_t in0_strides_0, int16_t *in1_allocated,
         int16_t *in1_aligned, int64_t in1_offset, int64_t in1_sizes_0,
         int64_t in1_strides_0, int16_t *out0_allocated, int16_t *out0_aligned,
         int64_t out0_offset, int64_t out0_sizes_0, int64_t out0_strides_0);
}
#endif

void dut_ref(int16_t *in0, int16_t *in1, int16_t *out0);

alignas(32) int16_t g_in0[IN0_SIZE];
alignas(32) int16_t g_in1[IN1_SIZE];
alignas(32) int16_t g_out0[OUT0_SIZE];
alignas(32) int16_t g_out0Ref[OUT0_SIZE];

int main(int argc, char *argv[]) {
  std::string dataDir(TO_STR(DATA_DIR));
  srand(10);
  std::generate(g_in0, g_in0 + IN0_SIZE,
                [&]() { return random_integer<int16_t>(); });
  std::generate(g_in1, g_in1 + IN1_SIZE,
                [&]() { return random_integer<int16_t>(); });

  writeData(g_in0, IN0_SIZE, dataDir + "/in0.txt");
  writeData(g_in1, IN1_SIZE, dataDir + "/in1.txt");

  chess_memory_fence();
  auto cyclesBegin = chess_cycle_count();
#ifdef TO_CPP
  dut(g_in0, g_in1, g_out0);
#elif TO_LLVM
  dut(nullptr, g_in0, 0, 0, 0, nullptr, g_in1, 0, 0, 0, nullptr, g_out0, 0, 0,
      0);
#endif
  auto cyclesEnd = chess_cycle_count();
  chess_memory_fence();

  auto cycleCount = (int)(cyclesEnd - cyclesBegin);
  reportCycleCount(cycleCount, dataDir + "/cycle_count.txt");

  writeData(g_out0, OUT0_SIZE, dataDir + "/out0.txt");

  dut_ref(g_in0, g_in1, g_out0Ref);
  writeData(g_out0Ref, OUT0_SIZE, dataDir + "/out0_ref.txt");

  bool ok = true;
  ok &= checkData(g_out0, g_out0Ref, OUT0_SIZE, 1);

  if (ok)
    printf("TEST PASSED\n");
  else
    printf("TEST FAILED\n");

  return ok ? 0 : 1;
}

void dut_ref(int16_t *in0, int16_t *in1, int16_t *out0) {
  for (unsigned k = 0; k < OUT0_SIZE; k += 1) {
    out0[k] = (int16_t)(in0[k] * in1[k]);
  }
}
