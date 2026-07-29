//===- target_model_unknown_device.cpp --------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// getTargetModel(AIEDevice) used to answer an out-of-range device with the
// VC1902 model instead of failing. Reachable from Python, since
// aieGetTargetModel() casts an unchecked uint32_t.
//
// Checked through a fatal-error handler, not CTest's PASS_REGULAR_EXPRESSION,
// which does not override the failure CTest reports for a signal-aborted child.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>

using namespace xilinx;

static void expectUnknownDevice(void *, const char *reason, bool) {
  bool named = llvm::StringRef(reason).contains("unknown AIEDevice value 0");
  // Returning from the handler would let report_fatal_error abort.
  std::exit(named ? 0 : 1);
}

int main() {
  llvm::install_fatal_error_handler(expectUnknownDevice);

  // 0 stays out of range whatever devices are added: the enumerators start
  // at 1. It is also what a default-initialized device carries.
  AIE::getTargetModel(static_cast<AIE::AIEDevice>(0));

  return 1; // Only reached if an unknown device resolved to a model.
}
