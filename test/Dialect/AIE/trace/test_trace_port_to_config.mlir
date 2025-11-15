// RUN: aie-opt %s -aie-trace-to-config | FileCheck %s

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @port_trace(%tile_0_2) {
      aie.trace.mode "Event-Time"
      aie.trace.packet id=1 type="core"
      aie.trace.port<0> port=North channel=1 direction=S2MM
      aie.trace.port<1> port=DMA channel=0 direction=MM2S
      aie.trace.event<"PORT_RUNNING_0">
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    
    aiex.runtime_sequence @seq(%arg0: memref<32xi32>) {
      aie.trace.start_config @port_trace
    }
  }
}

// CHECK: aie.device(npu1_1col)
// CHECK: %[[TILE:.*]] = aie.tile(0, 2)
// CHECK: aie.trace.config @port_trace_config(%[[TILE]])
// CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Start_Event" value = 15
// CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Trace_Stop_Event" value = 14
// CHECK-DAG: aie.trace.reg register = "Trace_Control0" field = "Mode" value = 0
// CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "ID" value = 1
// CHECK-DAG: aie.trace.reg register = "Trace_Control1" field = "Packet_Type" value = 0
// CHECK-DAG: aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" field = "Port_0_ID" value = "North:1"
// CHECK-DAG: aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" field = "Port_0_Master_Slave" value = 1
// CHECK-DAG: aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" field = "Port_1_ID" value = "DMA:0"
// CHECK-DAG: aie.trace.reg register = "Stream_Switch_Event_Port_Selection_0" field = "Port_1_Master_Slave" value = 0
