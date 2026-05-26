// PL layout discovery host driver.
// Feeds bufIn[i] = (int8_t)(i - 128). Expected: bufOut[i] == bufIn[i] (sentinel
// LUT[k] = k - 128 → identity over signed-input range).
// Prints all (input, output) mismatches up to 32 to reveal layout pattern.

#include "xrt_test_wrapper.h"
#include <cstdint>

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE_IN1 = std::int8_t;
using DATATYPE_OUT = std::int8_t;
#endif

void initialize_bufIn1(DATATYPE_IN1 *bufIn1, int SIZE) {
  for (int i = 0; i < SIZE; i++)
    bufIn1[i] = (int8_t)(i - 128);
}

void initialize_bufOut(DATATYPE_OUT *bufOut, int SIZE) {
  memset(bufOut, 0, SIZE);
}

int verify_passthrough_kernel(DATATYPE_IN1 *bufIn1, DATATYPE_OUT *bufOut,
                              int SIZE, int verbosity) {
  int errors = 0;
  int printed = 0;
  for (int i = 0; i < SIZE; i++) {
    int32_t in_v = bufIn1[i];
    int32_t out_v = bufOut[i];
    int32_t expect = in_v;
    if (out_v != expect) {
      if (printed < 64) {
        std::cout << "MISMATCH idx=" << i << " in=" << in_v
                  << " expect=" << expect << " got=" << out_v << std::endl;
        printed++;
      }
      errors++;
    }
  }
  std::cout << "TOTAL MISMATCHES: " << errors << " / " << SIZE << std::endl;
  // Also dump first 32 outputs raw for analysis
  std::cout << "FIRST 32 IO PAIRS:" << std::endl;
  for (int i = 0; i < 32 && i < SIZE; i++) {
    std::cout << "  [" << i << "] in=" << (int)bufIn1[i]
              << " out=" << (int)bufOut[i] << std::endl;
  }
  return errors;
}

int main(int argc, const char *argv[]) {
  constexpr int IN1_VOLUME = IN1_SIZE / sizeof(DATATYPE_IN1);
  constexpr int OUT_VOLUME = OUT_SIZE / sizeof(DATATYPE_OUT);

  args myargs = parse_args(argc, argv);

  int res = setup_and_run_aie<DATATYPE_IN1, DATATYPE_OUT, initialize_bufIn1,
                              initialize_bufOut, verify_passthrough_kernel>(
      IN1_VOLUME, OUT_VOLUME, myargs);
  return res;
}
