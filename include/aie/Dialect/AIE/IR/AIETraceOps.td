//===- AIETraceOps.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines operations for AIE trace configuration
//===----------------------------------------------------------------------===//

#ifndef AIE_TRACE_OPS
#define AIE_TRACE_OPS

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIETraceAttrs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AIE_TraceOp - Top-Level Trace Configuration (Symbol)
//===----------------------------------------------------------------------===//

def AIE_TraceOp : AIE_Op<"trace", [
    Symbol,
    HasParent<"DeviceOp">,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Declare a named trace configuration for a tile";
  
  let description = [{
    Declares a named trace configuration that can be invoked from a runtime
    sequence. The trace operation is a symbol that defines what events to
    monitor and how to configure the trace hardware, but does not actually
    apply the configuration. Configuration is applied via `aie.trace.start_config`
    within a `aiex.runtime_sequence` block.
    
    Example:
    ```mlir
    %tile02 = aie.tile(0, 2)
    
    aie.trace @trace_tile02(%tile02) {
      aie.trace.mode Event-Time
      aie.trace.packet id=1 type=core
      aie.trace.event<INSTR_EVENT_0>
      aie.trace.event<INSTR_VECTOR>
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    ```
  }];
  
  let arguments = (ins
    Index:$tile,
    SymbolNameAttr:$sym_name,
    OptionalAttr<AIEI32Attr>:$buffer_size
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name `(` $tile `)` (`buffer_size` `=` $buffer_size^)? 
    $body attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceModeOp - Trace Mode Selection
//===----------------------------------------------------------------------===//

def AIE_TraceModeOp : AIE_Op<"trace.mode", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Set trace mode (Event-Time, Event-PC, or Execution)";
  
  let description = [{
    Specifies the trace mode. Default: Event-Time
    
    Example:
    ```mlir
    aie.trace.mode Event-Time
    ```
  }];
  
  let arguments = (ins TraceModeAttr:$mode);
  
  let assemblyFormat = [{
    $mode attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceEventOp - Event Slot Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceEventOp : AIE_Op<"trace.event", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Monitor a specific event in this trace unit";
  
  let description = [{
    Adds an event to be monitored by the trace unit. Each tile type
    (core/mem/memtile/shimtile) has architecture-specific events.
    
    Maximum 8 events per trace unit (verified at compile time).
    
    Example:
    ```mlir
    aie.trace.event<INSTR_EVENT_0>
    aie.trace.event<LOCK_STALL>
    ```
  }];
  
  let arguments = (ins
    TraceEventAttr:$event,
    OptionalAttr<StrAttr>:$label
  );
  
  let assemblyFormat = [{
    $event (`label` `=` $label^)? attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TracePacketOp - Packet Routing Configuration
//===----------------------------------------------------------------------===//

def AIE_TracePacketOp : AIE_Op<"trace.packet", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Enable packet-switched trace routing";
  
  let description = [{
    Enables packet routing for trace data. Assigns packet ID (1-31) and
    packet type to differentiate tile sources during parsing.
    
    Example:
    ```mlir
    aie.trace.packet id=1 type=core
    ```
  }];
  
  let arguments = (ins
    ConfinedAttr<AIEI32Attr, [IntMinValue<1>, IntMaxValue<31>]>:$id,
    TracePacketTypeAttr:$type
  );
  
  let assemblyFormat = [{
    `id` `=` $id `type` `=` $type attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TracePortOp - Stream Switch Port Event Configuration
//===----------------------------------------------------------------------===//

def AIE_TracePortOp : AIE_Op<"trace.port", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure stream switch port for event monitoring";
  
  let description = [{
    Configures a stream switch port monitoring slot (0-7) to track events
    from a specific port. The port is specified by type (NORTH, DMA, etc.),
    channel number, and direction (S2MM=master, MM2S=slave).
    
    Direction semantics:
    - S2MM (Stream-to-Memory-Mapped) = master port (receiving from stream)
    - MM2S (Memory-Mapped-to-Stream) = slave port (sending to stream)
    
    After configuration, the port can be monitored using:
    - PORT_RUNNING_<slot>
    - PORT_IDLE_<slot>
    - PORT_STALLED_<slot>
    - PORT_TLAST_<slot>
    
    Example:
    ```mlir
    aie.trace.port<0> port = North channel = 1 direction = S2MM
    aie.trace.port<1> port = DMA channel = 0 direction = MM2S
    aie.trace.event<"PORT_RUNNING_0">  // Monitor NORTH:1 (S2MM/master)
    aie.trace.event<"PORT_IDLE_1">     // Monitor DMA:0 (MM2S/slave)
    ```
  }];
  
  let arguments = (ins
    ConfinedAttr<AIEI32Attr, [IntMinValue<0>, IntMaxValue<7>]>:$slot,
    WireBundle:$port,
    AIEI32Attr:$channel,
    DMAChannelDir:$direction
  );
  
  let assemblyFormat = [{
    `<` $slot `>` `port` `=` $port `channel` `=` $channel `direction` `=` $direction
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceStartEventOp - Trace Start Event Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceStartEventOp : AIE_Op<"trace.start", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure trace start event";
  
  let description = [{
    Specifies the event that triggers trace capture to begin.
    For multi-tile tracing, typically uses a broadcast event (default: 15).
    
    Example:
    ```mlir
    aie.trace.start broadcast=15
    aie.trace.start event=<TRUE>
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<AIEI32Attr>:$broadcast,
    OptionalAttr<TraceEventAttr>:$event
  );
  
  let assemblyFormat = [{
    (`broadcast` `=` $broadcast^)?
    (`event` `=` $event^)?
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceStopEventOp - Trace Stop Event Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceStopEventOp : AIE_Op<"trace.stop", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure trace stop event";
  
  let description = [{
    Specifies the event that triggers trace capture to end.
    For multi-tile tracing, typically uses a broadcast event (default: 14).
    
    Example:
    ```mlir
    aie.trace.stop broadcast=14
    aie.trace.stop event=<USER_EVENT_0>
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<AIEI32Attr>:$broadcast,
    OptionalAttr<TraceEventAttr>:$event
  );
  
  let assemblyFormat = [{
    (`broadcast` `=` $broadcast^)?
    (`event` `=` $event^)?
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceConfigOp - Configuration Container (Intermediate, Symbol)
//===----------------------------------------------------------------------===//

def AIE_TraceConfigOp : AIE_Op<"trace.config", [
    Symbol,
    HasParent<"DeviceOp">,
    SingleBlockImplicitTerminator<"EndOp">
  ]> {
  let summary = "Intermediate representation of lowered trace configuration";
  
  let description = [{
    Generated by AIETraceToConfigPass. Contains calculated register
    addresses and values needed to configure trace hardware.
    
    This is an intermediate representation that makes it easier to perform
    optimizations before final lowering to NPU register writes.
    
    Example (generated, not user-written):
    ```mlir
    aie.trace.config @trace_tile02_config(%tile02) {
      aie.trace.reg register="Trace_Control0" field="Trace_Start_Event" value=15
      aie.trace.reg register="Trace_Control0" field="Mode" value="Event-Time"
    }
    ```
  }];
  
  let arguments = (ins
    Index:$tile,
    SymbolNameAttr:$sym_name,
    OptionalAttr<TracePacketTypeAttr>:$packet_type
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name `(` $tile `)` (`packet_type` `=` $packet_type^)? $body attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceRegOp - Logical Register/Bitfield Specification
//===----------------------------------------------------------------------===//

def AIE_TraceRegOp : AIE_Op<"trace.reg", [
    HasParent<"TraceConfigOp">
  ]> {
  let summary = "Specify a trace register field write by logical name";
  
  let description = [{
    Specifies a single register field write needed to configure trace hardware.
    Uses logical register names (e.g., "Trace_Control0"), field names
    (e.g., "Trace_Start_Event"), and values (integers or event names). If field
    is omitted, entire register is written.
    
    The mask field (optional) allows partial writes. When present, only the bits
    set in the mask are written, and the value is assumed to already be shifted.
    This enables combining multiple field writes into a single register write.
    
    Example:
    ```mlir
    aie.trace.reg register="Trace_Control0" field="Trace_Start_Event" value=15
    aie.trace.reg register="Trace_Control0" field="Mode" value=0
    aie.trace.reg register="Trace_Control0" value=0x12345678 mask=0xFF00
    ```
  }];
  
  let arguments = (ins
    StrAttr:$reg_name,
    OptionalAttr<StrAttr>:$field,
    AnyAttr:$value,
    OptionalAttr<AIEI32Attr>:$mask,
    OptionalAttr<StrAttr>:$comment
  );
  
  let assemblyFormat = [{
    `register` `=` $reg_name
    (`field` `=` $field^)?
    `value` `=` $value
    (`mask` `=` $mask^)?
    (`comment` `=` $comment^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceComboEventOp - Combo Event Configuration (within trace)
//===----------------------------------------------------------------------===//

def AIE_TraceComboEventOp : AIE_Op<"trace.combo_event", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure a combo event for logical event combinations";
  
  let description = [{
    Configures a combo event that creates a derived event from logical
    combinations of other events. Combo events are evaluated by hardware
    Event Logic and produce new events (COMBO_EVENT_0/1/2/3) that can be
    traced using aie.trace.event operations.
    
    The combo event is configured with two input events and a logic function
    (AND, AND_NOT, OR, OR_NOT). AIE2 supports hierarchical combinations
    through slot 2 (combo of combo0 and combo1 results).
    
    Slots:
    - Slot 0 (combo0): uses eventA, eventB
    - Slot 1 (combo1): uses eventC, eventD  
    - Slot 2 (combo2): hierarchical, uses COMBO_EVENT_0 and COMBO_EVENT_1
    - Slot 3 (combo3): architecture-specific state machine (no configuration)
    
    Example:
    ```mlir
    aie.trace @my_trace(%tile02) {
      // Combo 0: lock stalled AND NOT DMA active
      aie.trace.combo_event<0> "LOCK_STALL" AND_NOT "DMA_S2MM_0_STALLED"
      
      // Combo 1: instruction event OR vector operation
      aie.trace.combo_event<1> "INSTR_EVENT_0" OR "INSTR_VECTOR"
      
      // Combo 2: (combo0) AND (combo1)
      aie.trace.combo_event<2> "COMBO_EVENT_0" AND "COMBO_EVENT_1"
      
      // Trace the combo results
      aie.trace.event<"COMBO_EVENT_0">
      aie.trace.event<"COMBO_EVENT_1">
      aie.trace.event<"COMBO_EVENT_2">
      ...
    }
    ```
  }];
  
  let arguments = (ins
    AIEI32Attr:$slot,           // 0, 1, or 2
    TraceEventAttr:$eventA,     // First input event
    ComboLogicAttr:$logic,      // Logic function
    TraceEventAttr:$eventB      // Second input event
  );
  
  let assemblyFormat = [{
    `<` $slot `>` $eventA $logic $eventB attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceEdgeEventOp - Edge Detection Event Configuration (within trace)
//===----------------------------------------------------------------------===//

def AIE_TraceEdgeEventOp : AIE_Op<"trace.edge_event", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure an edge detection event";
  
  let description = [{
    Configures an edge detection event that triggers on signal transitions
    (rising edge, falling edge, or both) rather than signal levels.
    
    Edge detection is useful for counting event occurrences rather than
    measuring duration. For example, counting the number of lock stalls
    vs. the total cycles spent stalled.
    
    Each module has 2 edge detectors producing EDGE_DETECTION_EVENT_0/1
    that can be traced using aie.trace.event operations.
    
    Example:
    ```mlir
    aie.trace @my_trace(%tile02) {
      // Edge detector 0: count lock stalls (rising edges)
      aie.trace.edge_event<0> event="LOCK_STALL" trigger=RISING
      
      // Edge detector 1: count DMA transitions (both edges)
      aie.trace.edge_event<1> event="DMA_MM2S_0_FINISHED_BD" trigger=BOTH
      
      // Trace the edge-detected events
      aie.trace.event<"EDGE_DETECTION_EVENT_0">
      aie.trace.event<"EDGE_DETECTION_EVENT_1">
      ...
    }
    ```
  }];
  
  let arguments = (ins
    AIEI32Attr:$slot,           // 0 or 1
    TraceEventAttr:$event,      // Source event to monitor
    EdgeTriggerAttr:$trigger    // RISING, FALLING, or BOTH
  );
  
  let assemblyFormat = [{
    `<` $slot `>` `event` `=` $event `trigger` `=` $trigger attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceStartConfigOp - Runtime Configuration Invocation
//===----------------------------------------------------------------------===//

def AIE_TraceStartConfigOp : AIE_Op<"trace.start_config", []> {
  let summary = "Apply a trace configuration in runtime sequence";
  
  let description = [{
    Invokes a named trace configuration within a runtime sequence.
    During lowering, this operation is replaced by the actual register
    writes (npu_write32) that implement the trace configuration.
    
    Example:
    ```mlir
    aiex.runtime_sequence @seq {
      aie.trace.start_config @my_trace
    }
    ```
  }];
  
  let arguments = (ins
    FlatSymbolRefAttr:$trace_config
  );
  
  let assemblyFormat = [{
    $trace_config attr-dict
  }];
  
  let hasVerifier = 1;
}

#endif // AIE_TRACE_OPS
