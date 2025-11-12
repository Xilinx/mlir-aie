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
  let summary = "Reference to a trace event by name";
  
  let description = [{
    References an event by name. Event code is looked up from the
    architecture-specific event database (utils/events_database.json).
    
    Validated against:
    - Tile type (core events only valid for core tiles, etc.)
    - Architecture (AIE/AIE2/AIE2P/AIE4)
    
    Examples: INSTR_EVENT_0, LOCK_STALL, PORT_RUNNING_0
  }];
  
  let parameters = (ins
    StringRefParameter<>:$name
  );
  
  let assemblyFormat = "`<` $name `>`";
}

#endif // AIE_TRACE_ATTRS
