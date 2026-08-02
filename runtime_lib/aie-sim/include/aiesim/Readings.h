//===- Readings.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Observability records: what a run saw, as data. Schema and the shape-to-
// viewer mapping are in docs/readings-schema.json and docs/Readings.md.
//
// Recording never perturbs the result -- there is no real-time constraint and
// nothing here touches the active set -- so it can be exhaustive rather than
// sampled. That is the property device trace does not have.
//
//===----------------------------------------------------------------------===//

#ifndef AIESIM_READINGS_H
#define AIESIM_READINGS_H

#include <cstdint>
#include <string>
#include <vector>

namespace aiesim {

class Array;

namespace readings {

/// Schema version this emitter writes. Bump the major when a consumer that
/// knows the old one would misread the new one.
inline constexpr const char *kSchemaVersion = "1.0.0";

/// A number that cannot be misreported: unit and derivation travel with it.
struct Quantity {
  double value = 0;
  std::string unit;
  std::string derivedFrom;
  std::vector<std::pair<std::string, std::string>> inputs;
};

enum class Better { Neither, Lower, Higher };

struct Scalar {
  std::string id, label, description;
  Quantity quantity;
  Better betterWhen = Better::Neither;
  std::vector<std::string> tags;
};

/// Treemap node. `value` is set on leaves only; a parent's is the sum of its
/// children, so the two can never disagree. `capacity` on a bounded node makes
/// the free gap computable, which is the number a stack/buffer collision shows
/// up in.
struct ContainmentNode {
  std::string name;
  bool hasValue = false;
  double value = 0;
  bool hasCapacity = false;
  double capacity = 0;
  std::vector<std::pair<std::string, std::string>> attrs;
  std::vector<ContainmentNode> children;
};

struct Containment {
  std::string id, label, description, unit;
  ContainmentNode root;
  std::vector<std::string> tags;
};

struct CoverageItem {
  std::string key;
  bool seen = false;
  uint64_t count = 0;
  std::vector<std::pair<std::string, std::string>> attrs;
};

struct Coverage {
  std::string id, label, description;
  /// Set to 0 to omit: a percentage against an invented denominator is worse
  /// than no percentage.
  uint64_t universe = 0;
  std::vector<CoverageItem> items;
  std::vector<std::string> tags;
};

enum class Outcome { Pass, Fail, Unknown };

/// A judgment computed here rather than left to whoever reads the number.
struct Verdict {
  std::string id, label, why;
  Outcome outcome = Outcome::Unknown;
  std::string severity = "warn";
  std::vector<std::string> evidence;
};

struct Provenance {
  std::string simVersion, gitSha, toolchainLock, engine, recordedAt, host;
  bool gitDirty = false;
  std::vector<std::string> argv;
};

struct RunInfo {
  std::string id, design, device, generation;
  uint64_t cycles = 0;
  bool quiescent = true;
  std::vector<std::pair<std::string, std::string>> params;
};

/// One run, observed. Serialises byte-stably for identical inputs: emission
/// order is fixed here and the only wall-clock values sit under the pointers
/// listed in `diffIgnore`.
class Record {
public:
  RunInfo run;
  Provenance provenance;
  std::vector<Scalar> scalars;
  std::vector<Containment> containments;
  std::vector<Coverage> coverages;
  std::vector<Verdict> verdicts;

  /// Ids promoted into summary.headline. Unknown ids are ignored, so a caller
  /// may name a scalar it did not end up emitting.
  std::vector<std::string> headline;

  /// Serialise, summary tier included. The summary is derived here rather than
  /// by the caller so it cannot drift from the body.
  std::string toJson() const;
};

struct CaptureConfig {
  std::string design;
  std::string runId;
  /// Device name as spelled to makeDeviceFromName ("npu2_1col"). Supplied by
  /// the caller because DeviceModel keeps the geometry, not the name it was
  /// parsed from, and a record that cannot say which device it describes is
  /// not worth diffing.
  std::string device;
  Provenance provenance;
  /// Include per-tile nodes whose memory was never touched. Off by default: on
  /// a 48-tile array they are most of the record and none of the signal.
  bool includeUntouchedTiles = false;
};

/// Capture what the array can report today: geometry and touched memory
/// (containment), registers written but modelled by nothing (coverage), and
/// the scalars and verdicts over both.
///
/// Memory write tracking must have been enabled before the run for the
/// containment `touched` values to be meaningful; see enableMemoryTracking.
Record capture(Array &array, const CaptureConfig &config);

/// Turn on write tracking for every memory in the array. Call before the run.
/// Untracked memories report a touched value of 0, which capture() marks
/// rather than silently reporting as "nothing was written".
void enableMemoryTracking(Array &array);

} // namespace readings
} // namespace aiesim

#endif // AIESIM_READINGS_H
