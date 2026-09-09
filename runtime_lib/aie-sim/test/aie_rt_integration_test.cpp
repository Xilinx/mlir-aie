//===- aie_rt_integration_test.cpp -------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Every other test in this suite configures the model by hand-writing
// registers using this model's OWN layout tables -- derived from the same
// reading of aie-rt that produced the model. If a layout was misread, the
// model and its test are wrong together and agree: self-consistency, not
// correctness.
//
// This file instead links the real vendored aie-rt driver (built as
// aie_rt_sim, see the parent CMakeLists.txt) and calls its PUBLIC API. The
// driver generates the register traffic through the ess_* ABI
// (docs/AIESimulator.md section 2); if this model misread a layout, the
// driver's writes land somewhere it does not expect and an assertion here
// fails. Where the driver and the model disagree, that is a finding to
// report, not a test to adjust.
//
// The XAie_Config field values below are copied from the one place mlir-aie
// itself builds this struct, lib/Targets/AIERT.cpp:262-277, so this test
// drives the model with the same configuration a real aiecc-built host
// program would.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"

#include "TestSupport.h"

extern "C" {
#include "xaiengine.h"
}

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// aie-rt-side device config, AIE2P/npu2, one column. Field values and their
// source lines:
//   AieGen            XAIE_DEV_GEN_AIE2P_STRIX_B0   AIERT.cpp:259
//   BaseAddr          XAIE_BASE_ADDR (0x40000000)   AIERT.cpp:187,264
//   ColShift/RowShift 25 / 20                       AIERT.cpp:265-266,
//                                                    AIETargetModel.h:738-739
//   ShimRowNum        0                              AIERT.cpp:188,269
//   MemTileRowStart   1                              AIERT.cpp:189,270
//   MemTileNumRows    1                              AIETargetModel.h:894-906
//                                                    (BaseNPU2TargetModel)
//   AieTileRowStart/  2 / 4  (MemTileRowStart +      AIERT.cpp:272-275
//   AieTileNumRows    MemTileNumRows; rows()=6 total)
// NumRows/NumCols (6/1) and the model-side DeviceModel below both come from
// aiesim::makeAIE2PDevice(1), so the two sides of the seam share one
// geometry rather than two independently-typed copies of it.
//
// Backend: the patch (third_party/patches/aie-rt/
// 0001-cdo-sim-defork-fixes.patch) adds this field and changes
// XAie_CfgInitialize to pass it to XAie_IOInit, because the shipping
// xaiengine host library is built with BOTH __AIECDO__ and __AIESIM__
// (runtime_lib/xaiengine/lib/CMakeLists.txt:57) and needs to pick a backend
// at run time rather than compile time. This library defines only
// __AIESIM__ (see the aie_rt_sim target), so XAIE_IO_BACKEND_SIM is also
// what XAie_IOInit would pick by itself
// (third_party/aie-rt/driver/src/io_backend/xaie_io.c:34-35); it is spelled
// out here rather than relied on implicitly.
constexpr uint8_t kAieTileRowStart = 2;
constexpr uint8_t kAieTileNumRows = 4;

// A location on the one core tile used by each of the three sections below,
// so a driver/model disagreement caught in one section cannot be masked by
// state a different section left behind on the same tile.
constexpr uint8_t kLockTestRow = 2;
constexpr uint8_t kDmaTestRow = 3;
constexpr uint8_t kStreamSwitchTestRow = 4;

// Collects Array::error() calls instead of the default abort, matching the
// convention in lock_test.cpp / dma_test.cpp: a driver/model disagreement
// most often surfaces as a fault (an out-of-range BD id, an unclaimed
// register) rather than a silently wrong value, so every section checks this
// stays empty except where a fault is the point.
struct ErrorSink {
  std::vector<std::string> messages;
  DiagnosticHandler handler() {
    return [this](const std::string &m) { messages.push_back(m); };
  }
  bool empty() const { return messages.empty(); }
};

// Replicates what a real aiecc-built HOST program does, which is the only
// path this simulator ever sees: the generated mlir_aie_init_libxaie() fills an
// XAie_Config and calls XAie_CfgInitialize (via mlir_aie_init_device,
// runtime_lib/test_lib/test_library.cpp:42) and never calls
// XAie_SetupPartitionConfig -- there are zero occurrences of it in
// lib/Targets/AIETargetXAIEV2.cpp. That matters: DevInst->NumCols is therefore
// still 0, so the copy at xaiegbl.c:198-202 DOES take effect and the live base
// address is the config's.
//
// Do not "fix" this by adding XAie_SetupPartitionConfig. That is the
// COMPILER-side CDO sequence (AIERT.cpp:280-284), it takes the other branch,
// and copying it here is what made an earlier version of this test agree with
// a wrong base address.
XAie_DevInst makeDevInst(XAie_Config &cfg) {
  XAie_InstDeclare(devInst, &cfg);
  AIESIM_CHECK(XAie_CfgInitialize(&devInst, &cfg) == XAIE_OK);
  return devInst;
}

//===----------------------------------------------------------------------===//
// LOCKS: XAie_LockSetValue / XAie_LockAcquire / XAie_LockRelease against a
// real LockModule, checking both the register layout (Lock.cpp's citations)
// and the acquire polarity (Components.h: negative == AcquireGreaterEqual).
//===----------------------------------------------------------------------===//

void testLocks(XAie_DevInst &dev, Array &array, ErrorSink &err) {
  Tile *core = array.tile(0, kLockTestRow);
  AIESIM_CHECK(core != nullptr);
  LockModule *lm = core->locks();
  AIESIM_CHECK(lm != nullptr);

  XAie_LocType loc = XAie_TileLoc(/*col=*/0, kLockTestRow);
  const uint8_t lockId = 3;

  // XAie_LockSetValue: the direct-write path (Lock.cpp's LockN_Value
  // register), bypassing acquire semantics. Sets lock 3's counter to 5.
  AIESIM_CHECK(XAie_LockSetValue(&dev, loc, XAie_Lock{lockId, 5}) == XAIE_OK);
  AIESIM_CHECK_EQ(lm->value(lockId), int32_t{5});

  // Plain Acquire(5) (LockVal >= 0, exact match): succeeds against the value
  // just set, and Lock.cpp documents that a successful plain acquire also
  // subtracts the matched value (the file's "one corner not fully nailed
  // down by a single source line" -- this is one of the few ways to observe
  // it against real driver traffic instead of only the model's own opinion
  // of itself).
  AIESIM_CHECK(XAie_LockAcquire(&dev, loc, XAie_Lock{lockId, 5},
                                /*TimeOut=*/1000) == XAIE_OK);
  AIESIM_CHECK_EQ(lm->value(lockId), int32_t{0});

  // XAie_LockRelease increments (AIEOps.td: "In AIE2, increment the lock by
  // value"); release 4, so the counter reads 4.
  AIESIM_CHECK(XAie_LockRelease(&dev, loc, XAie_Lock{lockId, 4},
                                /*TimeOut=*/1000) == XAIE_OK);
  AIESIM_CHECK_EQ(lm->value(lockId), int32_t{4});

  // AcquireGreaterEqual is a NEGATIVE LockVal (Components.h, Lock.cpp,
  // AIERT.cpp:322-333 all agree on this polarity -- see Lock.cpp's header).
  // Counter is 4: -5 (need >= 5) must fail, -4 (need >= 4) must succeed and
  // drop the counter to 0 by the full threshold, matching
  // AIEOps.td's "decremented by value" for this mode.
  AIESIM_CHECK(XAie_LockAcquire(&dev, loc, XAie_Lock{lockId, -5},
                                /*TimeOut=*/0) == XAIE_LOCK_RESULT_FAILED);
  AIESIM_CHECK_EQ(lm->value(lockId), int32_t{4});
  AIESIM_CHECK(XAie_LockAcquire(&dev, loc, XAie_Lock{lockId, -4},
                                /*TimeOut=*/1000) == XAIE_OK);
  AIESIM_CHECK_EQ(lm->value(lockId), int32_t{0});

  AIESIM_CHECK(err.empty());
}

//===----------------------------------------------------------------------===//
// DMA + CHANNEL CONTROL: XAie_DmaDescInit / XAie_DmaSetMultiDimAddr /
// XAie_DmaEnableBd / XAie_DmaWriteBd build a 2-D BD; XAie_DmaChannelEnable +
// XAie_DmaChannelPushBdToQueue start it, and XAie_DmaGetPendingBdCount
// (itself decoding the model's status register through the driver's own
// field masks, not this test's) confirms the model's DmaModule queued it
// before any simulated cycle has run. The transfer is then drained through a
// stream-switch connection the driver also configured
// (XAie_StrmConnCctEnable), and the delivered word order is checked against
// the n-D shape the driver was asked for -- the highest-value assertion
// here, since BD packing is what Dma.cpp itself calls "the most intricate
// thing the model re-derived".
//===----------------------------------------------------------------------===//

// Interleaves stepping the array with draining `port`, because the real
// stream switch's FIFOs are only 2 deep (StreamSwitch.cpp's kFifoDepth): a
// single runUntilQuiescent() would stall the moment the DMA fills the
// switch and never finish a 12-word transfer. Mirrors dma_test.cpp's
// drainInterleaved.
void drainInterleaved(Array &array, StreamPort &port, size_t count,
                      std::vector<uint32_t> &words, int maxCycles) {
  for (int cyc = 0; cyc < maxCycles && words.size() < count; ++cyc) {
    array.advance(1);
    while (port.canPop()) {
      uint32_t w;
      bool tlast;
      port.pop(w, tlast);
      words.push_back(w);
    }
  }
}

void testDmaAndChannelControl(XAie_DevInst &dev, Array &array,
                              ErrorSink &err) {
  Tile *core = array.tile(0, kDmaTestRow);
  AIESIM_CHECK(core != nullptr);
  XAie_LocType loc = XAie_TileLoc(/*col=*/0, kDmaTestRow);

  // Fill tile memory so the value at word offset w is 5000 + w: the pushed
  // sequence can then be checked directly against a hand-computed offset
  // sequence without a second copy of the memory contents.
  std::vector<uint32_t> mem(64);
  for (uint32_t i = 0; i < mem.size(); ++i)
    mem[i] = 5000 + i;
  AIESIM_CHECK(core->memory()->write(0, mem.data(), mem.size() * 4));

  // Route DMA0's MM2S output to a local Ctrl master through the real
  // stream switch, configured entirely by the driver: XAie_StrmConnCctEnable
  // writes both the master's Config field (the physical slave index,
  // _XAie_GetSlaveIdx) and the slave's enable bit in one call
  // (third_party/aie-rt/driver/src/stream_switch/xaie_ss.c:224-260).
  AIESIM_CHECK(XAie_StrmConnCctEnable(&dev, loc, DMA, /*SlvPortNum=*/0, CTRL,
                                      /*MstrPortNum=*/0) == XAIE_OK);
  StreamPort *sink = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  AIESIM_CHECK(sink != nullptr);

  // A 2-D BD: D0 (fast, index 0) wrap 4 step 3; D1 (slow, index 1) wrap 3
  // step 1 -- Tensor->Dim[i] maps onto DimDesc[i] unchanged
  // (third_party/aie-rt/driver/src/dma/xaie_dma_aieml.c:168-193,
  // _XAieMl_DmaSetMultiDim), so index 0 is the innermost/fastest dimension,
  // matching Dma.cpp's computeAddress. Base word offset 8 (byte address 32,
  // 4-byte aligned per Aie2PTileDmaProp.AddrAlignShift == 2,
  // xaie2pgbl_reginit.c:1804) rather than 0, so a decode bug that silently
  // drops the address field cannot pass by coincidence.
  XAie_DmaDesc desc;
  AIESIM_CHECK(XAie_DmaDescInit(&dev, &desc, loc) == XAIE_OK);
  XAie_DmaDimDesc dims[2] = {};
  dims[0].AieMlDimDesc = {/*StepSize=*/3, /*Wrap=*/4};
  dims[1].AieMlDimDesc = {/*StepSize=*/1, /*Wrap=*/3};
  XAie_DmaTensor tensor = {/*NumDim=*/2, dims};
  const uint32_t numWords = 4 * 3; // D0.Wrap * D1.Wrap.
  const uint64_t baseWordOffset = 8;
  AIESIM_CHECK(XAie_DmaSetMultiDimAddr(&desc, &tensor, baseWordOffset * 4,
                                       numWords * 4) == XAIE_OK);
  AIESIM_CHECK(XAie_DmaEnableBd(&desc) == XAIE_OK);
  const uint8_t bdId = 2;
  AIESIM_CHECK(XAie_DmaWriteBd(&dev, &desc, loc, bdId) == XAIE_OK);

  // CHANNEL CONTROL: enable the channel, then push the BD onto its
  // start-queue (XAie_DmaChannelPushBdToQueue writes only BdNum to the
  // queue word, third_party/aie-rt/driver/src/dma/xaie_dma.c:114-155 --
  // RepeatCount defaults to the encoded 0, i.e. one shot).
  // Channel N's data plane is stream-switch DMA-bundle port N
  // (DmaModuleImpl::effectivePort, lib/Dma.cpp), so this must match the
  // SlvPortNum wired to CTRL above.
  const uint8_t ch = 0;
  AIESIM_CHECK(XAie_DmaChannelEnable(&dev, loc, ch, DMA_MM2S) == XAIE_OK);
  AIESIM_CHECK(XAie_DmaChannelPushBdToQueue(&dev, loc, ch, DMA_MM2S, bdId) ==
              XAIE_OK);

  // Before any simulated cycle runs: the model's DmaModule must already
  // have the task queued. XAie_DmaGetPendingBdCount decodes the model's own
  // status-register bit layout through the driver's masks
  // (_XAieMl_DmaGetPendingBdCount, xaie_dma_aieml.c:1087-1120), so this
  // checks the status word's TaskQSize/ChannelRunning fields agree with
  // Dma.cpp's onStatusRead in both directions, not just that some nonzero
  // bit came back.
  uint8_t pending = 0;
  AIESIM_CHECK(XAie_DmaGetPendingBdCount(&dev, loc, ch, DMA_MM2S, &pending) ==
              XAIE_OK);
  AIESIM_CHECK(pending >= 1);

  std::vector<uint32_t> words;
  drainInterleaved(array, *sink, numWords, words, /*maxCycles=*/500);

  AIESIM_CHECK(err.empty());
  AIESIM_CHECK_EQ(words.size(), static_cast<size_t>(numWords));
  for (uint32_t l = 0; l < numWords && l < words.size(); ++l) {
    uint32_t d0 = l % 4, d1 = (l / 4) % 3;
    uint32_t expected = 5000 + static_cast<uint32_t>(baseWordOffset) +
                        d0 * 3 + d1 * 1;
    AIESIM_CHECK_EQ(words[l], expected);
  }
  AIESIM_CHECK_EQ(core->dma()->completedBds(DmaDirection::MM2S, ch), 1u);
}

//===----------------------------------------------------------------------===//
// STREAM SWITCH: XAie_StrmConnCctEnable on its own, isolated from the DMA
// section's traffic by using a different tile. aie-rt has no API for a raw
// stream word (that is below the register-level ABI it models), so the
// model's own StreamPort::push/pop stands in for a producer/consumer,
// exactly as stream_switch_test.cpp does; what is under test is whether the
// driver's write lands on the connection the model expects.
//===----------------------------------------------------------------------===//

void testStreamSwitch(XAie_DevInst &dev, Array &array, ErrorSink &err) {
  Tile *core = array.tile(0, kStreamSwitchTestRow);
  AIESIM_CHECK(core != nullptr);
  XAie_LocType loc = XAie_TileLoc(/*col=*/0, kStreamSwitchTestRow);

  AIESIM_CHECK(XAie_StrmConnCctEnable(&dev, loc, DMA, /*SlvPortNum=*/0, CTRL,
                                      /*MstrPortNum=*/0) == XAIE_OK);

  StreamPort *slave = core->streamSwitch()->slavePort(PortBundle::DMA, 0);
  StreamPort *master = core->streamSwitch()->masterPort(PortBundle::Ctrl, 0);
  AIESIM_CHECK(slave != nullptr);
  AIESIM_CHECK(master != nullptr);

  slave->push(0xC0FFEEu, /*tlast=*/true);
  bool delivered = false;
  for (int cyc = 0; cyc < 10 && !delivered; ++cyc) {
    array.advance(1);
    delivered = master->canPop();
  }
  AIESIM_CHECK(delivered);
  if (delivered) {
    uint32_t word;
    bool tlast;
    master->pop(word, tlast);
    AIESIM_CHECK_EQ(word, 0xC0FFEEu);
    AIESIM_CHECK(tlast);
  }
  AIESIM_CHECK(err.empty());
}

} // namespace

int main() {
  DeviceModel dev = makeAIE2PDevice(/*numCols=*/1);

  // No baseAddr override any more. This tier FOUND that Device.cpp had
  // 0x40000000 here, cited from AIERT.cpp's XAie_Config -- a value that is
  // wrong for both paths: the host path uses 0x20000000000
  // (AIETargetXAIEV2.cpp:383), and on the compiler path AIERT.cpp:280 calls
  // XAie_SetupPartitionConfig first so the config value never reaches DevInst
  // at all and the live base is 0x0 (AIERT.cpp:190). Every other test in this
  // suite passed regardless, because they hand-add dev.baseAddr themselves --
  // exactly the self-consistency this file exists to break.

  setCurrentArray(std::make_unique<Array>(dev, nullptr));
  Array &array = currentArray();
  ErrorSink err;
  array.setDiagnosticHandler(err.handler());

  XAie_Config cfg{
      /*AieGen*/ XAIE_DEV_GEN_AIE2P_STRIX_B0,
      /*BaseAddr*/ 0x20000000000, // lib/Targets/AIETargetXAIEV2.cpp:383
      /*ColShift*/ static_cast<uint8_t>(dev.colShift),
      /*RowShift*/ static_cast<uint8_t>(dev.rowShift),
      /*NumRows*/ static_cast<uint8_t>(dev.numRows),
      /*NumCols*/ static_cast<uint8_t>(dev.numCols),
      /*ShimRowNum*/ 0,
      /*MemTileRowStart*/ 1,
      /*MemTileNumRows*/ static_cast<uint8_t>(dev.numMemTileRows),
      /*AieTileRowStart*/ kAieTileRowStart,
      /*AieTileNumRows*/ kAieTileNumRows,
      /*PartProp*/ {},
      /*Backend*/ XAIE_IO_BACKEND_SIM};
  XAie_DevInst devInst = makeDevInst(cfg);
  // The point of this tier in one assertion: the address the driver will put
  // on every ess_Write32 is the one the model claims it decodes.
  AIESIM_CHECK_EQ(devInst.BaseAddr, uint64_t{0x20000000000});
  AIESIM_CHECK_EQ(devInst.BaseAddr, dev.baseAddr);

  testLocks(devInst, array, err);
  testDmaAndChannelControl(devInst, array, err);
  testStreamSwitch(devInst, array, err);

  return aiesim_test::summarize("aie_rt_integration");
}
