// RUN: aie-opt %s -split-input-file -verify-diagnostics

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @bad_slot(%tile_0_2) {
      // expected-error @+1 {{attribute 'slot' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0 whose maximum value is 7}}
      aie.trace.port<8> port=North channel=1 master=true
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @duplicate_slot(%tile_0_2) {
      // expected-error @+1 {{duplicate port slot 0 in trace duplicate_slot}}
      aie.trace.port<0> port=North channel=1 master=true
      aie.trace.port<0> port=DMA channel=0 master=false
    }
  }
}

// -----

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    
    aie.trace @negative_channel(%tile_0_2) {
      // expected-error @+1 {{invalid stream switch port configuration for tile (0, 2)}}
      aie.trace.port<0> port=North channel=-1 master=true
    }
  }
}
