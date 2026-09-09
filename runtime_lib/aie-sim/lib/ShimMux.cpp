//===- ShimMux.cpp ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The shim tile's PL-interface mux and demux: what the stream switch's south
// ports connect to on the outside.
//
// This is a separate block from the stream switch, not a corner of it, and all
// three authorities we have agree on that: aie-rt gives it its own XAie_PlIfMod
// (ShimNocMux / ShimNocDeMux, xaie2pgbl_reginit.c:2359-2362) distinct from the
// XAie_StrmMod; mlir-aie gives it its own op, aie.shim_mux, distinct from
// aie.switchbox; and its registers live in the NOC module at 0x1F000 rather
// than in the switch's 0x3F000 window.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Components.h"

#include <cstdint>

using namespace aiesim;

namespace {

// xaie2pgbl_params.h:19155,19175.
constexpr uint32_t kMuxConfig = 0x0001F000;
constexpr uint32_t kDemuxConfig = 0x0001F004;

// Which south port each two-bit field covers, in the order aie-rt indexes
// them. _XAie_ConfigShimNocMux maps port -> field as {2,3}->{0,1} and
// {6,7}->{2,3} (xaie_plif.c:626-632); _XAie_ConfigShimNocDeMux maps
// {2,3,4,5}->{0,1,2,3} (:687). The field LSBs are then the ShimNocMux /
// ShimNocDeMux tables at xaie2pgbl_reginit.c:2268-2283 -- which are simply
// ascending pairs, so the LSB is 2*fieldIndex plus a per-register base.
constexpr uint8_t kMuxPorts[] = {2, 3, 6, 7};
constexpr uint8_t kDemuxPorts[] = {2, 3, 4, 5};
constexpr uint32_t kMuxLsb0 = 8;   // MUX_CONFIG_SOUTH2_LSB
constexpr uint32_t kDemuxLsb0 = 4; // DEMUX_CONFIG_SOUTH2_LSB
constexpr uint32_t kFieldWidth = 2;

ShimPortEndpoint decodeField(uint32_t word, uint32_t lsb) {
  switch ((word >> lsb) & 0x3u) {
  case 1:
    return ShimPortEndpoint::ShimDma;
  case 2:
    return ShimPortEndpoint::NoC;
  default:
    // 0 is PL. 3 is not a documented encoding; the reset state of both
    // registers is 0 across every field (all the *_DEFVAL are 0x0), and PL is
    // what that reset state means -- XAie_EnablePlToAieStrmPort's own note
    // says AIE<->PL is what a device reset leaves enabled. Treating an
    // undocumented 3 as PL rather than faulting keeps a design that writes
    // one running as the hardware default; nothing we have says what else it
    // could be.
    return ShimPortEndpoint::PL;
  }
}

class ShimMuxImpl final : public ShimMuxModule {
public:
  ShimPortEndpoint slaveEndpoint(uint32_t index) const override {
    for (uint32_t f = 0; f < std::size(kMuxPorts); ++f)
      if (kMuxPorts[f] == index)
        return decodeField(mux, kMuxLsb0 + kFieldWidth * f);
    // A south port with no field in the mux register is hard-wired to the PL
    // and is not something a design can steer. Saying PL here is the same
    // answer the register gives for the ports it does cover at reset.
    return ShimPortEndpoint::PL;
  }

  ShimPortEndpoint masterEndpoint(uint32_t index) const override {
    for (uint32_t f = 0; f < std::size(kDemuxPorts); ++f)
      if (kDemuxPorts[f] == index)
        return decodeField(demux, kDemuxLsb0 + kFieldWidth * f);
    return ShimPortEndpoint::PL;
  }

  void onRegWrite(uint32_t off, uint32_t value) {
    if (off == kMuxConfig)
      mux = value;
    else if (off == kDemuxConfig)
      demux = value;
  }

private:
  uint32_t mux = 0;
  uint32_t demux = 0;
};

} // namespace

int aiesim::shimDmaSouthPort(DmaDirection dir, uint32_t channel) {
  // aie-rt names a PORT and never a channel: XAie_EnableShimDmaToAieStrmPort
  // accepts only south 3 and 7 (xaie_plif.c:723-727) and
  // XAie_EnableAieToShimDmaStrmPort only south 2 and 3 (:753-757). So the
  // hardware fixes WHICH TWO ports the shim DMA can reach, and leaves which
  // of the two belongs to which channel unstated.
  //
  // mlir-aie's lowering assigns them in ascending order, and it is what
  // programs the designs this model replays: MM2S 0 -> south 3, MM2S 1 ->
  // south 7, S2MM 0 -> south 2, S2MM 1 -> south 3
  // (AIECreatePathFindFlows.cpp:1219-1229 for the mux side, :1253-1263 for the
  // demux side; those emit ShimMuxOp connects that AIERT.cpp:772-790 turns
  // into exactly the two calls above). A design using only channel 0 cannot
  // distinguish this convention from the opposite one.
  //
  // Note the two directions overlap on south 3: it is a MASTER port for S2MM
  // channel 1 and a SLAVE port for MM2S channel 0, which are different wires.
  if (dir == DmaDirection::MM2S)
    return channel == 0 ? 3 : channel == 1 ? 7 : -1;
  return channel == 0 ? 2 : channel == 1 ? 3 : -1;
}

const char *aiesim::shimPortEndpointName(ShimPortEndpoint endpoint) {
  switch (endpoint) {
  case ShimPortEndpoint::PL:
    return "PL";
  case ShimPortEndpoint::ShimDma:
    return "shim DMA";
  case ShimPortEndpoint::NoC:
    return "NoC";
  }
  return "?";
}

void aiesim::installShimMux(Tile &tile) {
  if (tile.getType() != TileType::Shim)
    return;

  auto module = std::make_unique<ShimMuxImpl>();
  ShimMuxImpl *raw = module.get();

  // One claim over both words. aie-rt reaches these through XAie_MaskWrite32
  // (xaie_plif.c:641, 700), which READS before it writes, so claiming only the
  // write side would still fault the read -- and an unclaimed read is the fatal
  // direction. That read is why the claim is a range rather than two handlers:
  // both words must read back what was last written to them.
  tile.regs().onWrite(kMuxConfig, kDemuxConfig + 4,
                      [raw](uint32_t off, uint32_t value) {
                        raw->onRegWrite(off, value);
                      });

  tile.setShimMux(std::move(module));
}
