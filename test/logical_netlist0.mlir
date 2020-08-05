// RUN: aie-opt %s | FileCheck %s

// CHECK-LABEL: module @logical_netlist0 {
// CHECK:       }

// This is similar to Cardano's SDF graph
module @logical_netlist0 {
  %k0 = AIE.compute()
  %k1 = AIE.compute()
  %k2 = AIE.compute()
  %k3 = AIE.compute()

  %m0 = AIE.memory()

  AIE.net(%k0, %m0)
  AIE.net(%m0, %k1)
  AIE.net(%m0, %k2)
  AIE.net(%k1, %k3)
  AIE.net(%k2, %k3)
}
