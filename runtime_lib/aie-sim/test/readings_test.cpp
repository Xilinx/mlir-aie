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
#include "aiesim/Readings.h"

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

} // namespace

int main() {
  testTrackingOffIsUnknownNotZero();
  testTouchedMemoryIsObserved();
  testUnclaimedRegistersBecomeCoverageAndAVerdict();
  testSummaryTierIsPresentAndPointsAtTheBody();
  testRecordIsByteStable();
  return aiesim_test::summarize("readings_test");
}
