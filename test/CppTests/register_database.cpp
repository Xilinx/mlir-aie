//===- register_database.cpp ------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIERegisterDatabase.h"

#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace xilinx::AIE;
using namespace mlir;

void test_load_database() {
  std::cout << "Test: Load AIE2 Register Database\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load AIE2 register database");
  }

  std::cout << "  ✓ Database loaded successfully\n";
}

void test_lookup_core_events() {
  std::cout << "Test: Lookup Core Events\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Test INSTR_EVENT_0 in core module
  auto instr0 = db->lookupEvent("INSTR_EVENT_0", "core");
  if (!instr0) {
    throw std::runtime_error("Failed to find INSTR_EVENT_0 in core module");
  }
  std::cout << "  ✓ INSTR_EVENT_0 = " << *instr0 << "\n";

  // Test INSTR_EVENT_1 in core module
  auto instr1 = db->lookupEvent("INSTR_EVENT_1", "core");
  if (!instr1) {
    throw std::runtime_error("Failed to find INSTR_EVENT_1 in core module");
  }
  std::cout << "  ✓ INSTR_EVENT_1 = " << *instr1 << "\n";

  // Test INSTR_VECTOR in core module
  auto instrvec = db->lookupEvent("INSTR_VECTOR", "core");
  if (!instrvec) {
    throw std::runtime_error("Failed to find INSTR_VECTOR in core module");
  }
  std::cout << "  ✓ INSTR_VECTOR = " << *instrvec << "\n";

  // Test MEMORY_STALL in core module
  auto memstall = db->lookupEvent("MEMORY_STALL", "core");
  if (!memstall) {
    throw std::runtime_error("Failed to find MEMORY_STALL in core module");
  }
  std::cout << "  ✓ MEMORY_STALL = " << *memstall << "\n";
}

void test_lookup_pl_events() {
  std::cout << "Test: Lookup PL (Shim) Events\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Test DMA events in pl module
  auto dma_s2mm_0_start = db->lookupEvent("DMA_S2MM_0_START_TASK", "pl");
  if (!dma_s2mm_0_start) {
    throw std::runtime_error(
        "Failed to find DMA_S2MM_0_START_TASK in pl module");
  }
  std::cout << "  ✓ DMA_S2MM_0_START_TASK = " << *dma_s2mm_0_start << "\n";

  auto dma_mm2s_0_start = db->lookupEvent("DMA_MM2S_0_START_TASK", "pl");
  if (!dma_mm2s_0_start) {
    throw std::runtime_error(
        "Failed to find DMA_MM2S_0_START_TASK in pl module");
  }
  std::cout << "  ✓ DMA_MM2S_0_START_TASK = " << *dma_mm2s_0_start << "\n";

  auto dma_s2mm_0_finished = db->lookupEvent("DMA_S2MM_0_FINISHED_TASK", "pl");
  if (!dma_s2mm_0_finished) {
    throw std::runtime_error(
        "Failed to find DMA_S2MM_0_FINISHED_TASK in pl module");
  }
  std::cout << "  ✓ DMA_S2MM_0_FINISHED_TASK = " << *dma_s2mm_0_finished
            << "\n";

  auto dma_starvation = db->lookupEvent("DMA_S2MM_0_STREAM_STARVATION", "pl");
  if (!dma_starvation) {
    throw std::runtime_error(
        "Failed to find DMA_S2MM_0_STREAM_STARVATION in pl module");
  }
  std::cout << "  ✓ DMA_S2MM_0_STREAM_STARVATION = " << *dma_starvation << "\n";
}

void test_nonexistent_event() {
  std::cout << "Test: Lookup Non-existent Event\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Should return nullopt for non-existent event
  auto result = db->lookupEvent("NONEXISTENT_EVENT", "core");
  if (result.has_value()) {
    throw std::runtime_error("Expected nullopt for non-existent event");
  }
  std::cout << "  ✓ Correctly returns nullopt for non-existent event\n";
}

void test_wrong_module() {
  std::cout << "Test: Lookup Event in Wrong Module\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // INSTR_EVENT_0 exists in core, but not in pl
  auto result = db->lookupEvent("INSTR_EVENT_0", "pl");
  if (result.has_value()) {
    throw std::runtime_error(
        "Expected nullopt when looking up core event in pl module");
  }
  std::cout << "  ✓ Correctly returns nullopt when module is wrong\n";
}

void test_trace_event_attr() {
  std::cout << "Test: TraceEventAttr getName() Method\n";

  MLIRContext context;
  context.loadDialect<AIEDialect>();

  // Create a TraceEventAttr using StringRef
  llvm::StringRef eventNameRef = "DMA_S2MM_0_START_TASK";
  auto eventAttr = TraceEventAttr::get(&context, eventNameRef);

  // Get the name
  std::string name = eventAttr.getName().str();

  std::cout << "  ✓ TraceEventAttr name: '" << name << "'\n";

  // Now test with the database
  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // This is what the pass does - lookup using the name
  auto eventNum = db->lookupEvent(name, "pl");
  if (!eventNum) {
    throw std::runtime_error("Failed to lookup event from TraceEventAttr name");
  }

  std::cout << "  ✓ Successfully looked up event: " << name << " = "
            << *eventNum << "\n";
}

int main() {
  try {
    std::cout << "==============================================\n";
    std::cout << "Register Database Unit Tests\n";
    std::cout << "==============================================\n\n";

    test_load_database();
    test_lookup_core_events();
    test_lookup_pl_events();
    test_nonexistent_event();
    test_wrong_module();
    test_trace_event_attr();

    std::cout << "\n==============================================\n";
    std::cout << "All tests passed! ✓\n";
    std::cout << "==============================================\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed: " << e.what() << "\n";
    return 1;
  }
}
