//===- target_model_unknown_device.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// getTargetModel() used to fall out of its switch and return the VC1902 model
// for any value outside AIEDevice, so a bad device silently answered with an
// unrelated device family instead of failing. aieGetTargetModel() casts an
// unchecked uint32_t, so such a value is reachable from the C API and from the
// Python get_target_model() binding built on it.
//
// report_fatal_error would abort the process, which CTest reports as a failure
// regardless of PASS_REGULAR_EXPRESSION, so intercept it with a fatal-error
// handler and check the message in-process. That keeps this a plain exit-status
// test like its siblings here, and keeps it portable to MSVC.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>

using namespace xilinx;

static void expectUnknownDevice(void *, const char *reason, bool) {
  // Exit from the handler: returning from it lets report_fatal_error abort.
  std::exit(llvm::StringRef(reason).contains("unknown AIEDevice value 16") ? 0
                                                                          : 1);
}

int main() {
  llvm::install_fatal_error_handler(expectUnknownDevice);

  // One past the last enumerator (npu2_7col). Deriving the value from the
  // enumerator keeps the test out of range as devices are added. Zero is out of
  // range too -- the cases start at 1 -- so a default-initialized device value
  // reaches this path as well.
  auto bad = static_cast<AIE::AIEDevice>(
      static_cast<int>(AIE::AIEDevice::npu2_7col) + 1);
  AIE::getTargetModel(bad);

  // Only reached if getTargetModel returned a model for an unknown device,
  // which is the regression under test.
  return 1;
}
