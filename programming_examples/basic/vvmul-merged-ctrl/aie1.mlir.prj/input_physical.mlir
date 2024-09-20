module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, North : 3>
    }
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      %0 = aie.amsel<2> (3)
      %1 = aie.amsel<3> (3)
      %2 = aie.amsel<4> (3)
      %3 = aie.amsel<5> (3)
      %4 = aie.masterset(South : 0, %2) {keep_pkt_header = true}
      %5 = aie.masterset(North : 1, %0)
      %6 = aie.masterset(North : 4, %1)
      %7 = aie.masterset(Ctrl : 0, %3) {keep_pkt_header = true}
      aie.packet_rules(South : 3) {
        aie.rule(31, 27, %0)
        aie.rule(31, 26, %1)
        aie.rule(31, 15, %3)
      }
      aie.packet_rules(Ctrl : 0) {
        aie.rule(31, 15, %2)
      }
    }
    %tile_0_1 = aie.tile(0, 1) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 26>}
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      %0 = aie.amsel<4> (3)
      %1 = aie.amsel<5> (3)
      %2 = aie.masterset(North : 1, %1)
      %3 = aie.masterset(Ctrl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %1)
      }
      aie.packet_rules(South : 4) {
        aie.rule(31, 26, %0)
      }
    }
    %tile_0_2 = aie.tile(0, 2) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 27>}
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<5> (3)
      %1 = aie.masterset(Ctrl : 0, %0) {keep_pkt_header = true}
      aie.packet_rules(South : 1) {
        aie.rule(31, 27, %0)
      }
    }
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, Ctrl : 0>
      aie.packet_dest<%tile_0_0, South : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(15) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_0, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @ctrlpkt_col0_mm2s_chan0(MM2S, 0, 0)
    memref.global "public" @ctrlpkt_col0_mm2s_chan0 : memref<2048xi32>
    aie.packet_flow(26) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.packet_flow(27) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, Ctrl : 0>
    } {keep_pkt_header = true, priority_route = true}
    aie.wire(%shim_mux_0_0 : North, %switchbox_0_0 : South)
    aie.wire(%tile_0_0 : DMA, %shim_mux_0_0 : DMA)
    aie.wire(%tile_0_1 : Core, %switchbox_0_1 : Core)
    aie.wire(%tile_0_1 : DMA, %switchbox_0_1 : DMA)
    aie.wire(%switchbox_0_0 : North, %switchbox_0_1 : South)
    aie.wire(%tile_0_2 : Core, %switchbox_0_2 : Core)
    aie.wire(%tile_0_2 : DMA, %switchbox_0_2 : DMA)
    aie.wire(%switchbox_0_1 : North, %switchbox_0_2 : South)
  }
}

