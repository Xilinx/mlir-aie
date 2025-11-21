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
#include "aie/Dialect/AIE/Util/AIERegisterDatabase.h"

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

void test_lookup_register() {
  std::cout << "Test: Lookup Registers\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Test Core_Control register in core module
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control in core module");
  }
  std::cout << "  ✓ Core_Control found\n";
  std::cout << "    - offset: 0x" << std::hex << coreControl->offset << std::dec
            << "\n";
  std::cout << "    - width: " << coreControl->width << " bits\n";

  // Verify offset is 0x32000
  if (coreControl->offset != 0x32000) {
    throw std::runtime_error("Core_Control offset mismatch");
  }
  std::cout << "  ✓ Core_Control offset is correct (0x32000)\n";
}

void test_lookup_register_wrong_module() {
  std::cout << "Test: Lookup Register in Wrong Module\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Core_Control exists in core, but not in pl
  auto result = db->lookupRegister("Core_Control", "pl");
  if (result != nullptr) {
    throw std::runtime_error(
        "Expected nullptr when looking up core register in pl module");
  }
  std::cout << "  ✓ Correctly returns nullptr when module is wrong\n";
}

void test_lookup_nonexistent_register() {
  std::cout << "Test: Lookup Non-existent Register\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  auto result = db->lookupRegister("NONEXISTENT_REGISTER", "core");
  if (result != nullptr) {
    throw std::runtime_error("Expected nullptr for non-existent register");
  }
  std::cout << "  ✓ Correctly returns nullptr for non-existent register\n";
}

void test_register_bit_fields() {
  std::cout << "Test: Register Bit Fields\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Core_Control register which has Enable and Reset fields
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control");
  }

  // Check that bit_fields are populated
  if (coreControl->bit_fields.empty()) {
    throw std::runtime_error("Core_Control has no bit fields");
  }
  std::cout << "  ✓ Core_Control has " << coreControl->bit_fields.size()
            << " bit fields\n";

  // Look for the Enable field (bit 0)
  const BitFieldInfo *enableField = coreControl->getField("Enable");
  if (!enableField) {
    throw std::runtime_error("Failed to find Enable bit field");
  }
  std::cout << "  ✓ Enable field found at bits [" << enableField->bit_start
            << ":" << enableField->bit_end << "]\n";

  // Look for the Reset field (bit 1)
  const BitFieldInfo *resetField = coreControl->getField("Reset");
  if (!resetField) {
    throw std::runtime_error("Failed to find Reset bit field");
  }
  std::cout << "  ✓ Reset field found at bits [" << resetField->bit_start << ":"
            << resetField->bit_end << "]\n";
}

void test_encode_field_value() {
  std::cout << "Test: Encode Field Values\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Performance_Control0 which has multi-bit fields
  auto perfControl = db->lookupRegister("Performance_Control0", "core");
  if (!perfControl) {
    throw std::runtime_error("Failed to find Performance_Control0");
  }

  // Test encoding Cnt0_Start_Event (bits 6:0)
  // Value 0x25 (event 37) should stay at bits 0-6
  const BitFieldInfo *cnt0Start = perfControl->getField("Cnt0_Start_Event");
  if (!cnt0Start) {
    throw std::runtime_error("Failed to find Cnt0_Start_Event field");
  }

  uint32_t encoded0 = db->encodeFieldValue(*cnt0Start, 0x25);
  if (encoded0 != 0x25) {
    std::cerr << "Expected 0x25, got 0x" << std::hex << encoded0 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Start_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Start_Event(0x25) = 0x" << std::hex << encoded0
            << std::dec << "\n";

  // Test encoding Cnt0_Stop_Event (bits 14:8)
  // Value 0x21 should be shifted to bits 8-14, resulting in 0x2100
  const BitFieldInfo *cnt0Stop = perfControl->getField("Cnt0_Stop_Event");
  if (!cnt0Stop) {
    throw std::runtime_error("Failed to find Cnt0_Stop_Event field");
  }

  uint32_t encoded1 = db->encodeFieldValue(*cnt0Stop, 0x21);
  if (encoded1 != 0x2100) {
    std::cerr << "Expected 0x2100, got 0x" << std::hex << encoded1 << std::dec
              << "\n";
    throw std::runtime_error("Cnt0_Stop_Event encoding failed");
  }
  std::cout << "  ✓ Cnt0_Stop_Event(0x21) = 0x" << std::hex << encoded1
            << std::dec << "\n";
}

void test_encode_single_bit_field() {
  std::cout << "Test: Encode Single-Bit Field Values\n";

  auto db = RegisterDatabase::loadAIE2();
  if (!db) {
    throw std::runtime_error("Failed to load database");
  }

  // Get Core_Control which has single-bit fields
  auto coreControl = db->lookupRegister("Core_Control", "core");
  if (!coreControl) {
    throw std::runtime_error("Failed to find Core_Control");
  }

  // Test Enable field (bit 0)
  const BitFieldInfo *enableField = coreControl->getField("Enable");
  if (!enableField) {
    throw std::runtime_error("Failed to find Enable field");
  }

  uint32_t enableEncoded = db->encodeFieldValue(*enableField, 1);
  if (enableEncoded != 0x1) {
    std::cerr << "Expected 0x1, got 0x" << std::hex << enableEncoded << std::dec
              << "\n";
    throw std::runtime_error("Enable encoding failed");
  }
  std::cout << "  ✓ Enable(1) = 0x" << std::hex << enableEncoded << std::dec
            << "\n";

  // Test Reset field (bit 1)
  const BitFieldInfo *resetField = coreControl->getField("Reset");
  if (!resetField) {
    throw std::runtime_error("Failed to find Reset field");
  }

  uint32_t resetEncoded = db->encodeFieldValue(*resetField, 1);
  if (resetEncoded != 0x2) {
    std::cerr << "Expected 0x2, got 0x" << std::hex << resetEncoded << std::dec
              << "\n";
    throw std::runtime_error("Reset encoding failed");
  }
  std::cout << "  ✓ Reset(1) = 0x" << std::hex << resetEncoded << std::dec
            << "\n";
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
    test_lookup_register();
    test_lookup_register_wrong_module();
    test_lookup_nonexistent_register();
    test_register_bit_fields();
    test_encode_field_value();
    test_encode_single_bit_field();

    std::cout << "\n==============================================\n";
    std::cout << "All tests passed! ✓\n";
    std::cout << "==============================================\n";

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "\n❌ Test failed: " << e.what() << "\n";
    return 1;
  }
}
