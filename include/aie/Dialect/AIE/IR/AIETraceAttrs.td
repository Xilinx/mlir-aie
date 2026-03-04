//===- AIETraceAttrs.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines attributes for AIE trace operations
//===----------------------------------------------------------------------===//

#ifndef AIE_TRACE_ATTRS
#define AIE_TRACE_ATTRS

include "aie/Dialect/AIE/IR/AIE.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Trace Mode Enumeration
//===----------------------------------------------------------------------===//

def TraceModeEventTime : I32EnumAttrCase<"EventTime", 0, "Event-Time">;
def TraceModeEventPC   : I32EnumAttrCase<"EventPC", 1, "Event-PC">;
def TraceModeExecution : I32EnumAttrCase<"Execution", 2, "Execution">;

def TraceModeAttr : I32EnumAttr<"TraceMode",
    "Trace capture mode",
    [
      TraceModeEventTime,
      TraceModeEventPC,
      TraceModeExecution
    ]> {
  let cppNamespace = "::xilinx::AIE";
  let description = [{
    Specifies the trace mode:
    - Event-Time (00): Captures event occurrence with timestamp
    - Event-PC (01): Captures program counter when event occurs
    - Execution (10): Instruction-level execution trace
  }];
}

//===----------------------------------------------------------------------===//
// Packet Type Enumeration
//===----------------------------------------------------------------------===//

def TracePacketTypeCore     : I32EnumAttrCase<"Core", 0, "core">;
def TracePacketTypeMem      : I32EnumAttrCase<"Mem", 1, "mem">;
def TracePacketTypeShimTile : I32EnumAttrCase<"ShimTile", 2, "shimtile">;
def TracePacketTypeMemTile  : I32EnumAttrCase<"MemTile", 3, "memtile">;

def TracePacketTypeAttr : I32EnumAttr<"TracePacketType",
    "Packet type identifier for parsing",
    [
      TracePacketTypeCore,
      TracePacketTypeMem,
      TracePacketTypeShimTile,
      TracePacketTypeMemTile
    ]> {
  let cppNamespace = "::xilinx::AIE";
  let description = [{
    Packet Type Convention:
    - 0: Core tile (CORE_MODULE)
    - 1: Core tile (MEMORY_MODULE)
    - 2: Shim tile
    - 3: Mem tile
  }];
}

//===----------------------------------------------------------------------===//
// Trace Event Attribute
//===----------------------------------------------------------------------===//

def TraceEventAttr : AttrDef<AIE_Dialect, "TraceEvent"> {
  let mnemonic = "trace_event";
  let summary = "Reference to a trace event by name or enum";

  let description = [{
    References an event by name string or typed enum. Strings are used for
    generic references that are resolved during lowering. Enums provide
    compile-time type checking.

    Validated against:
    - Tile type (core events only valid for core tiles, etc.)
    - Architecture (AIE/AIE2/AIE2P/AIE4)

    Examples:
      "INSTR_EVENT_0"                // String (resolved during lowering)
      CoreEventAIE2::INSTR_EVENT_0   // Enum
  }];

  let parameters = (ins
    "Attribute":$value
  );

  let assemblyFormat = "`<` custom<TraceEventValue>($value) `>`";

  let extraClassDeclaration = [{
    // Helper to get event name as string for lookups
    std::string getEventName() const;

    // Check if this is a string attribute (needs promotion)
    bool isStringAttr() const;

    // Get the underlying enum value if this is an enum
    std::optional<int64_t> getEnumValue() const;
  }];
}

//===----------------------------------------------------------------------===//
// Combo Event Logic Enumeration
//===----------------------------------------------------------------------===//

def ComboLogicAND     : I32EnumAttrCase<"AND", 0, "AND">;
def ComboLogicAND_NOT : I32EnumAttrCase<"AND_NOT", 1, "AND_NOT">;
def ComboLogicOR      : I32EnumAttrCase<"OR", 2, "OR">;
def ComboLogicOR_NOT  : I32EnumAttrCase<"OR_NOT", 3, "OR_NOT">;

def ComboLogicAttr : I32EnumAttr<"ComboLogic",
    "Combo event logic function",
    [
      ComboLogicAND,
      ComboLogicAND_NOT,
      ComboLogicOR,
      ComboLogicOR_NOT
    ]> {
  let cppNamespace = "::xilinx::AIE";
  let description = [{
    Logical operation for combining two events:
    - AND (00): eventA AND eventB
    - AND_NOT (01): eventA AND NOT eventB
    - OR (10): eventA OR eventB
    - OR_NOT (11): eventA OR NOT eventB
  }];
}

//===----------------------------------------------------------------------===//
// Edge Detection Trigger Mode Enumeration
//===----------------------------------------------------------------------===//

def EdgeTriggerRISING  : I32EnumAttrCase<"RISING", 1, "RISING">;
def EdgeTriggerFALLING : I32EnumAttrCase<"FALLING", 2, "FALLING">;
def EdgeTriggerBOTH    : I32EnumAttrCase<"BOTH", 3, "BOTH">;

def EdgeTriggerAttr : I32EnumAttr<"EdgeTrigger",
    "Edge detection trigger mode",
    [
      EdgeTriggerRISING,
      EdgeTriggerFALLING,
      EdgeTriggerBOTH
    ]> {
  let cppNamespace = "::xilinx::AIE";
  let description = [{
    Edge detection trigger mode:
    - RISING (01): Trigger on rising edge (0→1 transition)
    - FALLING (10): Trigger on falling edge (1→0 transition)
    - BOTH (11): Trigger on both edges
  }];
}

#endif // AIE_TRACE_ATTRS
