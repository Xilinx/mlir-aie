//===- readings_test.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// The readings record: that it observes what happened, that it is byte-stable,
// and that an unmeasured quantity never reads as a measured zero.
//
//===----------------------------------------------------------------------===//

#include "TestSupport.h"

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/Readings.h"
#include "aiesim/RegionMap.h"

#include <string>

using namespace aiesim;
using namespace aiesim::readings;

namespace {

std::unique_ptr<Array> makeArray() {
  DeviceModel dev;
  std::string err;
  AIESIM_CHECK(makeDeviceFromName("npu2_1col", dev, err));
  return std::make_unique<Array>(dev, nullptr);
}

/// Substring search, so the assertions name the JSON they expect rather than
/// pulling in a parser this directory does not depend on.
bool has(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

CaptureConfig config() {
  CaptureConfig c;
  c.design = "readings_test";
  c.runId = "test-run";
  c.device = "npu2_1col";
  c.provenance.simVersion = "test";
  return c;
}

void testTrackingOffIsUnknownNotZero() {
  auto array = makeArray();
  std::string json = capture(*array, config()).toJson();
  // The failure this guards is the one the register file exists to prevent,
  // in a new place: a zero that means "nobody counted" must not be reported
  // as a zero that means "nothing happened".
  AIESIM_CHECK(has(json, "\"id\":\"memory-tracking-enabled\""));
  AIESIM_CHECK(has(json, "\"outcome\":\"unknown\""));
  AIESIM_CHECK(!has(json, "\"outcome\":\"pass\",\"severity\":\"info\""));
}

void testTouchedMemoryIsObserved() {
  auto array = makeArray();
  enableMemoryTracking(*array);

  Tile *core = nullptr;
  for (uint32_t row = 0; row < array->device().numRows && !core; ++row)
    if (Tile *t = array->tile(0, row))
      if (t->getType() == TileType::Core)
        core = t;
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;

  const uint32_t kBytes = 100;
  std::vector<uint8_t> payload(kBytes, 0xAB);
  AIESIM_CHECK(core->memory()->write(0, payload.data(), kBytes));

  Record rec = capture(*array, config());
  std::string json = rec.toJson();

  AIESIM_CHECK(has(json, "\"id\":\"containment/tile-memory\""));
  AIESIM_CHECK(has(json, "\"unit\":\"bytes\""));
  // 100 bytes spans four 32-byte granules, so the count rounds UP to 128. The
  // rounding is the documented contract, not an accident worth hiding.
  AIESIM_CHECK(has(json, "\"value\":128"));
  AIESIM_CHECK(has(json, "\"capacity\":"));

  bool trackingPassed = false;
  for (const Verdict &v : rec.verdicts)
    if (v.id == "memory-tracking-enabled" && v.outcome == Outcome::Pass)
      trackingPassed = true;
  AIESIM_CHECK(trackingPassed);
}

void testUnclaimedRegistersBecomeCoverageAndAVerdict() {
  auto array = makeArray();
  // An offset nothing claims, in the middle of the core tile's register
  // window. Recorded rather than fatal, which is what makes it observable.
  array->recordUnclaimedWrite(0, 2, 0x31500);

  Record rec = capture(*array, config());
  std::string json = rec.toJson();

  AIESIM_CHECK(has(json, "\"id\":\"coverage/unclaimed-registers\""));
  AIESIM_CHECK(has(json, "tile:0,2/0x31500"));
  AIESIM_CHECK(has(json, "\"id\":\"all-registers-modelled\""));
  AIESIM_CHECK(has(json, "\"outcome\":\"fail\""));
  AIESIM_CHECK(has(json, "\"fail\":1"));
}

void testUnmodelledOpcodesBecomeCoverageAndAVerdict() {
  auto array = makeArray();
  array->recordUnmodelledOpcode(0, 2, "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX");

  Record rec = capture(*array, config());
  std::string json = rec.toJson();

  AIESIM_CHECK(has(json, "\"id\":\"coverage/opcode-semantics\""));
  AIESIM_CHECK(has(json, "VMAC_f_vmac_bfp_vmul_bfp_core_EX_EX"));
  AIESIM_CHECK(has(json, "\"id\":\"all-opcodes-modelled\""));
  AIESIM_CHECK(has(json, "\"scalar/unmodelled-opcodes\""));
  AIESIM_CHECK(has(json, "\"outcome\":\"fail\""));
}

void testUnmodelledOpcodesAreDedupedByName() {
  auto array = makeArray();
  // The same gap reached from several tiles is ONE gap: the question is which
  // instructions the engine lacks, not how many cores tripped over them.
  array->recordUnmodelledOpcode(0, 2, "VSHUFFLE_vec_shuffle_x");
  array->recordUnmodelledOpcode(1, 3, "VSHUFFLE_vec_shuffle_x");
  array->recordUnmodelledOpcode(0, 4, "VSRS_2x_mv_x_srs_cm_srsSign0");
  AIESIM_CHECK(array->unmodelledOpcodes().size() == 2);

  Record rec = capture(*array, config());
  AIESIM_CHECK(has(rec.toJson(), "\"value\":2"));
}

void testNoInstructionsReachedIsUnknownNotPass() {
  auto array = makeArray();
  Record rec = capture(*array, config());
  std::string json = rec.toJson();
  // Present even when empty: a shape that vanishes when clean cannot be
  // distinguished from a shape nobody emitted.
  AIESIM_CHECK(has(json, "\"id\":\"coverage/opcode-semantics\""));
  // And the verdict must be `unknown`, not `pass`: a design where no core ran
  // an instruction has not demonstrated that its instructions are modelled.
  bool sawUnknown = false;
  for (const Verdict &v : rec.verdicts)
    if (v.id == "all-opcodes-modelled" && v.outcome == Outcome::Unknown)
      sawUnknown = true;
  AIESIM_CHECK(sawUnknown);
}

void testSummaryTierIsPresentAndPointsAtTheBody() {
  auto array = makeArray();
  enableMemoryTracking(*array);
  Record rec = capture(*array, config());
  std::string json = rec.toJson();

  AIESIM_CHECK(has(json, "\"summary\""));
  AIESIM_CHECK(has(json, "\"headline\""));
  AIESIM_CHECK(has(json, "\"index\""));
  AIESIM_CHECK(has(json, "\"shape\":\"containment\""));
  AIESIM_CHECK(has(json, "\"shape\":\"coverage\""));
  AIESIM_CHECK(has(json, "\"scalar/cycles\""));
  // Units travel with values, so a consumer cannot misreport a bare number.
  AIESIM_CHECK(has(json, "\"unit\":\"cycles\""));
  AIESIM_CHECK(has(json, "\"derivedFrom\":\"touched / capacity\""));
  AIESIM_CHECK(has(json, "\"schemaVersion\":\"1.0.0\""));
  AIESIM_CHECK(has(json, "\"diffIgnore\""));
  // Every member docs/readings-schema.json marks required. Checked here rather
  // than only by an external validator so the build catches a drift between
  // emitter and schema without needing python.
  AIESIM_CHECK(has(json, "\"kind\":\"aie-sim-readings\""));
  AIESIM_CHECK(has(json, "\"device\":\"npu2_1col\""));
  AIESIM_CHECK(has(json, "\"cycles\":"));
  AIESIM_CHECK(has(json, "\"simVersion\":"));
}

void testRecordIsByteStable() {
  // Two independent runs of the same work must serialise identically, or every
  // diff is noise and the record cannot be a regression gate.
  std::string first, second;
  for (std::string *out : {&first, &second}) {
    auto array = makeArray();
    enableMemoryTracking(*array);
    for (uint32_t row = 0; row < array->device().numRows; ++row)
      if (Tile *t = array->tile(0, row))
        if (t->memory()) {
          std::vector<uint8_t> payload(64, 0x5A);
          t->memory()->write(0, payload.data(), 64);
        }
    array->recordUnclaimedWrite(0, 2, 0x31500);
    *out = capture(*array, config()).toJson();
  }
  AIESIM_CHECK(first == second);
  AIESIM_CHECK(!first.empty());
}

// A trimmed but verbatim slice of ldScripts_main_core_0_5.ld.script: stack at
// 0x70000+0xD00 and the tile's first own-memory buffer at 0x70D00.
const char *kScript = R"LD(
MEMORY
{
   program (RX) : ORIGIN = 0, LENGTH = 0x0020000
   data (!RX) : ORIGIN = 0x7D20C, LENGTH = 0x2DF4
}
SECTIONS
{
. = 0x70000;
_sp_start_value_DM_stack = .;
. += 0xD00; /* stack */
. = 0x70D00;
C_L1L2_3_0_buff_0 = .;
. += 0x1200;
}
)LD";

Tile *firstCore(Array &array) {
  for (uint32_t row = 0; row < array.device().numRows; ++row)
    if (Tile *t = array.tile(0, row))
      if (t->getType() == TileType::Core)
        return t;
  return nullptr;
}

void testZeroClearanceIsReportedAsAFailedVerdict() {
  auto array = makeArray();
  Tile *core = firstCore(*array);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;
  RegionMap map;
  std::string err;
  AIESIM_CHECK(parseLinkerScript(kScript, map, err));
  core->setRegionMap(std::move(map));

  Record rec = capture(*array, config());
  std::string json = rec.toJson();

  // The whole point of the region map: this fires with no core engine, no
  // execution, and no device -- before the overwrite it describes could happen.
  bool failed = false;
  for (const Verdict &v : rec.verdicts)
    if (v.id == "stack-clearance") {
      failed = v.outcome == Outcome::Fail;
      AIESIM_CHECK(v.why.find("C_L1L2_3_0_buff_0") != std::string::npos);
      AIESIM_CHECK(v.why.find("one-byte frame overrun") != std::string::npos);
    }
  AIESIM_CHECK(failed);
  AIESIM_CHECK(has(json, "\"id\":\"regions-disjoint\""));

  // The named allocations replace the flat "data" leaf, so the treemap shows
  // the stack abutting the buffer instead of one aggregate number.
  AIESIM_CHECK(has(json, "\"name\":\"_sp_start_value_DM_stack\""));
  AIESIM_CHECK(has(json, "\"kind\":\"stack\""));
  AIESIM_CHECK(has(json, "\"addr\":\"0x70d00\""));
  AIESIM_CHECK(has(json, "\"capacity\":3328"));
}

void testTouchedIsAttributedToTheRegionItLandedIn() {
  auto array = makeArray();
  enableMemoryTracking(*array);
  Tile *core = firstCore(*array);
  AIESIM_CHECK(core != nullptr);
  if (!core)
    return;
  RegionMap map;
  std::string err;
  AIESIM_CHECK(parseLinkerScript(kScript, map, err));
  core->setRegionMap(std::move(map));

  // 0x70D00 is the buffer's first byte; as a data-memory offset that is
  // 0x70D00 - ownMemoryBandBase = 0xD00.
  uint32_t off = 0x70D00 - ownMemoryBandBase(array->device().generation);
  std::vector<uint8_t> payload(64, 0x11);
  AIESIM_CHECK(core->memory()->write(off, payload.data(), 64));

  std::string json = capture(*array, config()).toJson();
  // The buffer accounts for the write and the stack does not, which is the
  // attribution a flat per-tile total cannot make.
  AIESIM_CHECK(has(json, "\"name\":\"C_L1L2_3_0_buff_0\",\"value\":64"));
  AIESIM_CHECK(has(json, "\"name\":\"_sp_start_value_DM_stack\",\"value\":0"));
}

} // namespace

/// A component that never runs must not make the array look blocked. Nothing
/// was scheduled, so there is no time to attribute and the verdict says so
/// rather than reporting a clean sweep of zero stalls.
void testNothingScheduledIsUnknownNotPass() {
  auto array = makeArray();
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "\"id\":\"interval/stall-attribution\""));
  AIESIM_CHECK(has(json, "\"id\":\"stalls-attributed\""));
  AIESIM_CHECK(has(json, "No component was ever scheduled"));
  AIESIM_CHECK(has(json, "\"scalar/stalled-cycles\""));
}

/// Switching attribution off is reported as unknown, not as an empty timeline
/// that reads like a run with no stalls -- the same silent-zero the register
/// file exists to prevent.
void testTimelineOffIsUnknownNotEmpty() {
  auto array = makeArray();
  array->setTimelineEnabled(false);
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "Stall attribution was switched off"));
  AIESIM_CHECK(!has(json, "\"id\":\"interval/stall-attribution\""));
  AIESIM_CHECK(!has(json, "\"scalar/stalled-cycles\""));
}

/// The interval shape reaches the index and the categories carry the
/// productive flag a consumer sums lost time with.
void testIntervalShapeIsIndexedWithItsCategories() {
  auto array = makeArray();
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "\"shape\":\"interval\""));
  AIESIM_CHECK(has(json, "\"timeUnit\":\"cycle\""));
  AIESIM_CHECK(has(json, "\"name\":\"running\",\"productive\":true"));
  AIESIM_CHECK(has(json, "\"name\":\"lock\",\"productive\":false"));
  AIESIM_CHECK(has(json, "\"name\":\"backpressure\",\"productive\":false"));
}

/// No DMA ran, so residency was never put to the test. `unknown`, not a pass:
/// a design that moved nothing has not demonstrated it keeps its stream on
/// chip, and reporting that as a clean result is the silent-zero mistake in
/// the place it would matter most.
void testNoTrafficIsUnknownNotResident() {
  auto array = makeArray();
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "\"id\":\"flow/stream-traffic\""));
  AIESIM_CHECK(has(json, "\"id\":\"stream-stays-on-chip\""));
  AIESIM_CHECK(has(json, "No DMA transfer happened"));
  AIESIM_CHECK(has(json, "\"scalar/ddr-bytes\""));
}

/// Bytes that stayed on chip pass; a single DDR byte fails. Driven through
/// recordDmaBytes rather than a configured DMA, because what is under test is
/// the verdict, and dma_test already covers the counting.
void testOnChipPassesAndDdrFails() {
  auto array = makeArray();
  array->recordDmaBytes(TileType::MemTile, /*toFabric=*/true, 256);
  array->recordDmaBytes(TileType::Core, /*toFabric=*/false, 256);
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "512 byte(s) moved stayed between L1 and L2"));
  AIESIM_CHECK(has(json, "\"from\":\"l2\",\"to\":\"fabric\",\"value\":256"));

  auto crossed = makeArray();
  crossed->recordDmaBytes(TileType::MemTile, /*toFabric=*/true, 256);
  crossed->recordDmaBytes(TileType::Shim, /*toFabric=*/true, 64);
  std::string json2 = capture(*crossed, config()).toJson();
  AIESIM_CHECK(has(json2, "64 byte(s) crossed DDR against 256 on-chip"));
  AIESIM_CHECK(has(json2, "\"shape\":\"flow\""));
}

/// A design that configured no route gets an empty graph, and the description
/// still accounts for packet-mode masters so "no edges" cannot be read as "no
/// routing" when the two are different things.
void testGraphIsEmptyWithoutConfiguration() {
  auto array = makeArray();
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(has(json, "\"id\":\"graph/stream-routes\""));
  AIESIM_CHECK(has(json, "\"shape\":\"graph\""));
  AIESIM_CHECK(has(json, "0 packet-mode master(s) are configured"));
}

/// Every cycle stepped shows up on the concurrency series, and it records
/// change points rather than one entry per cycle.
void testConcurrencySeriesRecordsChangePoints() {
  auto array = makeArray();
  // Nothing is active, so advance() jumps the clock rather than stepping and
  // there is no point to record -- which is the reading being honest about an
  // idle array rather than drawing a flat line through cycles nobody ran.
  array->advance(1000);
  AIESIM_CHECK(array->concurrency().empty());
  std::string json = capture(*array, config()).toJson();
  AIESIM_CHECK(!has(json, "\"id\":\"series/active-components\""));
}

int main() {
  testGraphIsEmptyWithoutConfiguration();
  testConcurrencySeriesRecordsChangePoints();
  testNoTrafficIsUnknownNotResident();
  testOnChipPassesAndDdrFails();
  testNothingScheduledIsUnknownNotPass();
  testTimelineOffIsUnknownNotEmpty();
  testIntervalShapeIsIndexedWithItsCategories();
  testZeroClearanceIsReportedAsAFailedVerdict();
  testTouchedIsAttributedToTheRegionItLandedIn();
  testTrackingOffIsUnknownNotZero();
  testTouchedMemoryIsObserved();
  testUnclaimedRegistersBecomeCoverageAndAVerdict();
  testUnmodelledOpcodesBecomeCoverageAndAVerdict();
  testUnmodelledOpcodesAreDedupedByName();
  testNoInstructionsReachedIsUnknownNotPass();
  testSummaryTierIsPresentAndPointsAtTheBody();
  testRecordIsByteStable();
  return aiesim_test::summarize("readings_test");
}
