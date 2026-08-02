//===- Readings.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Readings.h"

#include "aiesim/Array.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>

using namespace aiesim;
using namespace aiesim::readings;

namespace {

/// Minimal JSON writer. Hand-written because this directory deliberately
/// depends on nothing but a C++17 compiler, and because emission order is a
/// correctness property here: the record must be byte-stable so a diff is
/// signal rather than key-order churn.
class Json {
public:
  std::string out;

  void beginObj() { punct('{'); }
  void endObj() { close('}'); }
  void beginArr() { punct('['); }
  void endArr() { close(']'); }

  void key(const char *k) {
    comma();
    quote(k);
    out += ':';
    needComma = false;
  }
  void str(const std::string &v) { comma(); quote(v); }
  void raw(const std::string &v) { comma(); out += v; }
  void boolean(bool v) { comma(); out += v ? "true" : "false"; }
  void num(double v) { comma(); out += fmt(v); }
  void u64(uint64_t v) {
    comma();
    char b[24];
    std::snprintf(b, sizeof(b), "%" PRIu64, v);
    out += b;
  }

  void kv(const char *k, const std::string &v) { key(k); str(v); }
  void kv(const char *k, uint64_t v) { key(k); u64(v); }
  void kvBool(const char *k, bool v) { key(k); boolean(v); }
  void kvNum(const char *k, double v) { key(k); num(v); }

  /// Omits the member entirely when empty, so an absent value never reads as a
  /// present-but-blank one.
  void kvOpt(const char *k, const std::string &v) {
    if (!v.empty())
      kv(k, v);
  }

  void strArray(const char *k, const std::vector<std::string> &v) {
    if (v.empty())
      return;
    key(k);
    beginArr();
    for (const std::string &s : v)
      str(s);
    endArr();
  }

  void attrs(const char *k,
             const std::vector<std::pair<std::string, std::string>> &a) {
    if (a.empty())
      return;
    key(k);
    beginObj();
    for (const auto &kv2 : a) {
      comma();
      quote(kv2.first);
      out += ':';
      needComma = false;
      str(kv2.second);
    }
    endObj();
  }

private:
  bool needComma = false;

  // Integral values print as integers so a byte count never renders as
  // "65536.0" in one run and "65536" in another.
  static std::string fmt(double v) {
    char b[40];
    if (std::isfinite(v) && v == std::floor(v) && std::fabs(v) < 1e15)
      std::snprintf(b, sizeof(b), "%lld", (long long)v);
    else
      std::snprintf(b, sizeof(b), "%.10g", v);
    return b;
  }

  void punct(char c) {
    comma();
    out += c;
    needComma = false;
  }
  void close(char c) {
    out += c;
    needComma = true;
  }
  void comma() {
    if (needComma)
      out += ',';
    needComma = true;
  }
  void quote(const std::string &s) {
    out += '"';
    for (char c : s) {
      switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          out += b;
        } else {
          out += c;
        }
      }
    }
    out += '"';
  }
};

void writeQuantity(Json &j, const Quantity &q) {
  j.beginObj();
  j.kvNum("value", q.value);
  j.kv("unit", q.unit);
  j.kvOpt("derivedFrom", q.derivedFrom);
  j.attrs("inputs", q.inputs);
  j.endObj();
}

const char *betterName(Better b) {
  switch (b) {
  case Better::Lower: return "lower";
  case Better::Higher: return "higher";
  case Better::Neither: break;
  }
  return "neither";
}

const char *outcomeName(Outcome o) {
  switch (o) {
  case Outcome::Pass: return "pass";
  case Outcome::Fail: return "fail";
  case Outcome::Unknown: break;
  }
  return "unknown";
}

void writeNode(Json &j, const ContainmentNode &n) {
  j.beginObj();
  j.kv("name", n.name);
  if (n.hasValue)
    j.kvNum("value", n.value);
  if (n.hasCapacity)
    j.kvNum("capacity", n.capacity);
  j.attrs("attrs", n.attrs);
  if (!n.children.empty()) {
    j.key("children");
    j.beginArr();
    for (const ContainmentNode &c : n.children)
      writeNode(j, c);
    j.endArr();
  }
  j.endObj();
}

/// Element count, for summary.index: what a consumer needs to predict the cost
/// of fetching an observation before it fetches it.
uint64_t nodeCount(const ContainmentNode &n) {
  uint64_t total = 1;
  for (const ContainmentNode &c : n.children)
    total += nodeCount(c);
  return total;
}

double nodeSum(const ContainmentNode &n) {
  if (n.children.empty())
    return n.hasValue ? n.value : 0;
  double sum = 0;
  for (const ContainmentNode &c : n.children)
    sum += nodeSum(c);
  return sum;
}

/// Largest leaves first, for summary.topN. Ties break on name so the record
/// stays byte-stable.
void collectLeaves(const ContainmentNode &n, const std::string &path,
                   std::vector<std::pair<std::string, double>> &out) {
  std::string here = path.empty() ? n.name : path + "/" + n.name;
  if (n.children.empty()) {
    if (n.hasValue && n.value > 0)
      out.push_back({here, n.value});
    return;
  }
  for (const ContainmentNode &c : n.children)
    collectLeaves(c, here, out);
}

} // namespace

std::string Record::toJson() const {
  Json j;
  j.beginObj();
  j.kv("$schema", "./readings-schema.json");
  j.kv("schemaVersion", kSchemaVersion);
  j.kv("kind", "aie-sim-readings");

  j.key("diffIgnore");
  j.beginArr();
  j.str("/provenance/recordedAt");
  j.str("/provenance/host");
  j.str("/provenance/argv");
  j.endArr();

  j.key("run");
  j.beginObj();
  j.kv("id", run.id);
  j.kvOpt("design", run.design);
  // Always emitted, never omitted-when-empty: the schema requires a non-empty
  // device, so a caller that forgot one fails validation instead of producing
  // a record nobody can tell the provenance of.
  j.kv("device", run.device);
  j.kvOpt("generation", run.generation);
  j.kv("cycles", run.cycles);
  j.kvBool("quiescent", run.quiescent);
  j.attrs("params", run.params);
  j.endObj();

  j.key("provenance");
  j.beginObj();
  j.kv("simVersion", provenance.simVersion);
  j.kvOpt("gitSha", provenance.gitSha);
  if (!provenance.gitSha.empty())
    j.kvBool("gitDirty", provenance.gitDirty);
  j.kvOpt("toolchainLock", provenance.toolchainLock);
  j.kvOpt("engine", provenance.engine);
  j.kvOpt("recordedAt", provenance.recordedAt);
  j.kvOpt("host", provenance.host);
  j.strArray("argv", provenance.argv);
  j.endObj();

  // --- summary: derived here so it cannot drift from the body it describes.
  uint64_t pass = 0, fail = 0, unknown = 0;
  for (const Verdict &v : verdicts) {
    if (v.outcome == Outcome::Pass)
      ++pass;
    else if (v.outcome == Outcome::Fail)
      ++fail;
    else
      ++unknown;
  }

  j.key("summary");
  j.beginObj();

  j.key("headline");
  j.beginArr();
  for (const std::string &id : headline)
    for (const Scalar &s : scalars)
      if (s.id == id) {
        j.beginObj();
        j.kv("id", s.id);
        j.kv("label", s.label);
        j.key("quantity");
        writeQuantity(j, s.quantity);
        j.kv("betterWhen", betterName(s.betterWhen));
        j.endObj();
      }
  j.endArr();

  j.key("verdicts");
  j.beginObj();
  j.kv("pass", pass);
  j.kv("fail", fail);
  j.kv("unknown", unknown);
  j.endObj();

  j.key("index");
  j.beginArr();
  for (const Scalar &s : scalars) {
    j.beginObj();
    j.kv("id", s.id);
    j.kv("shape", "scalar");
    j.kv("label", s.label);
    j.kv("points", uint64_t(1));
    j.kvOpt("unit", s.quantity.unit);
    j.endObj();
  }
  for (const Containment &c : containments) {
    j.beginObj();
    j.kv("id", c.id);
    j.kv("shape", "containment");
    j.kv("label", c.label);
    j.kv("points", nodeCount(c.root));
    j.kvOpt("unit", c.unit);
    j.endObj();
  }
  for (const Coverage &c : coverages) {
    j.beginObj();
    j.kv("id", c.id);
    j.kv("shape", "coverage");
    j.kv("label", c.label);
    j.kv("points", uint64_t(c.items.size()));
    j.endObj();
  }
  j.endArr();

  j.key("topN");
  j.beginArr();
  for (const Containment &c : containments) {
    std::vector<std::pair<std::string, double>> leaves;
    collectLeaves(c.root, "", leaves);
    std::stable_sort(leaves.begin(), leaves.end(),
                     [](const std::pair<std::string, double> &a,
                        const std::pair<std::string, double> &b) {
                       if (a.second != b.second)
                         return a.second > b.second;
                       return a.first < b.first;
                     });
    if (leaves.empty())
      continue;
    if (leaves.size() > 10)
      leaves.resize(10);
    j.beginObj();
    j.kv("id", c.id);
    j.key("entries");
    j.beginArr();
    for (const auto &leaf : leaves) {
      j.beginObj();
      j.kv("key", leaf.first);
      j.key("quantity");
      writeQuantity(j, Quantity{leaf.second, c.unit, "", {}});
      j.endObj();
    }
    j.endArr();
    j.endObj();
  }
  j.endArr();
  j.endObj(); // summary

  j.key("verdicts");
  j.beginArr();
  for (const Verdict &v : verdicts) {
    j.beginObj();
    j.kv("id", v.id);
    j.kv("label", v.label);
    j.kv("outcome", outcomeName(v.outcome));
    j.kv("severity", v.severity);
    j.kv("why", v.why);
    j.strArray("evidence", v.evidence);
    j.endObj();
  }
  j.endArr();

  j.key("shapes");
  j.beginObj();

  j.key("scalar");
  j.beginArr();
  for (const Scalar &s : scalars) {
    j.beginObj();
    j.kv("id", s.id);
    j.kv("label", s.label);
    j.kvOpt("description", s.description);
    j.strArray("tags", s.tags);
    j.key("quantity");
    writeQuantity(j, s.quantity);
    j.kv("betterWhen", betterName(s.betterWhen));
    j.endObj();
  }
  j.endArr();

  j.key("containment");
  j.beginArr();
  for (const Containment &c : containments) {
    j.beginObj();
    j.kv("id", c.id);
    j.kv("label", c.label);
    j.kvOpt("description", c.description);
    j.strArray("tags", c.tags);
    j.kv("unit", c.unit);
    j.key("root");
    writeNode(j, c.root);
    j.endObj();
  }
  j.endArr();

  j.key("coverage");
  j.beginArr();
  for (const Coverage &c : coverages) {
    j.beginObj();
    j.kv("id", c.id);
    j.kv("label", c.label);
    j.kvOpt("description", c.description);
    j.strArray("tags", c.tags);
    if (c.universe)
      j.kv("universe", c.universe);
    j.key("items");
    j.beginArr();
    for (const CoverageItem &it : c.items) {
      j.beginObj();
      j.kv("key", it.key);
      j.kvBool("seen", it.seen);
      if (it.count)
        j.kv("count", it.count);
      j.attrs("attrs", it.attrs);
      j.endObj();
    }
    j.endArr();
    j.endObj();
  }
  j.endArr();

  j.endObj(); // shapes
  j.endObj();
  j.out += '\n';
  return j.out;
}

void aiesim::readings::enableMemoryTracking(Array &array) {
  const DeviceModel &dev = array.device();
  for (uint32_t row = 0; row < dev.numRows; ++row)
    for (uint32_t col = 0; col < dev.numCols; ++col)
      if (Tile *t = array.tile(col, row)) {
        if (Memory *m = t->memory())
          m->trackWrites();
        if (Memory *p = t->programMemory())
          p->trackWrites();
      }
}

namespace {

const char *tileTypeName(TileType t) {
  switch (t) {
  case TileType::Shim: return "shim";
  case TileType::MemTile: return "memtile";
  case TileType::Core: return "core";
  case TileType::Invalid: break;
  }
  return "invalid";
}

/// One memory of one tile. Emitted as a leaf carrying its capacity, so the
/// viewer can draw touched-against-capacity and the free remainder is
/// arithmetic rather than another number that could disagree.
bool addMemoryLeaf(ContainmentNode &tileNode, const char *name, Memory *mem,
                   bool includeUntouched) {
  if (!mem)
    return false;
  uint32_t touched = mem->tracksWrites() ? mem->touchedBytes() : 0;
  if (!touched && !includeUntouched)
    return false;
  ContainmentNode leaf;
  leaf.name = name;
  leaf.hasValue = true;
  leaf.value = touched;
  leaf.hasCapacity = true;
  leaf.capacity = mem->size();
  if (!mem->tracksWrites())
    // Otherwise a 0 here reads as "nothing was written" rather than "nobody
    // was counting", which is the same silent-zero mistake the register file
    // exists to prevent.
    leaf.attrs.push_back({"tracked", "false"});
  tileNode.children.push_back(std::move(leaf));
  return true;
}

} // namespace

Record aiesim::readings::capture(Array &array, const CaptureConfig &config) {
  Record rec;
  const DeviceModel &dev = array.device();

  rec.run.id = config.runId;
  rec.run.design = config.design;
  rec.run.device = config.device;
  rec.run.generation = dev.generation == Generation::AIE2 ? "AIE2" : "AIE2P";
  rec.run.cycles = array.cycle();
  rec.provenance = config.provenance;
  if (rec.provenance.simVersion.empty())
    rec.provenance.simVersion = kSchemaVersion;

  // --- containment: touched memory, array -> column -> tile -> memory.
  Containment mem;
  mem.id = "containment/tile-memory";
  mem.label = "Tile memory touched";
  mem.description = "Bytes in 32-byte granules written at least once, against "
                    "each memory's capacity.";
  mem.unit = "bytes";
  mem.root.name = "array";

  uint64_t totalTouched = 0, totalCapacity = 0, tilesWithMemory = 0;
  for (uint32_t col = 0; col < dev.numCols; ++col) {
    ContainmentNode colNode;
    colNode.name = "col:" + std::to_string(col);
    for (uint32_t row = 0; row < dev.numRows; ++row) {
      Tile *t = array.tile(col, row);
      if (!t)
        continue;
      ContainmentNode tileNode;
      tileNode.name = "tile:" + std::to_string(col) + "," + std::to_string(row);
      tileNode.attrs.push_back({"type", tileTypeName(t->getType())});
      bool any = addMemoryLeaf(tileNode, "data", t->memory(),
                               config.includeUntouchedTiles);
      any |= addMemoryLeaf(tileNode, "program", t->programMemory(),
                           config.includeUntouchedTiles);
      if (t->memory()) {
        totalCapacity += t->memory()->size();
        totalTouched += t->memory()->tracksWrites()
                            ? t->memory()->touchedBytes()
                            : 0;
        ++tilesWithMemory;
      }
      if (any)
        colNode.children.push_back(std::move(tileNode));
    }
    if (!colNode.children.empty())
      mem.root.children.push_back(std::move(colNode));
  }
  rec.containments.push_back(std::move(mem));

  // --- coverage: registers written that nothing models.
  Coverage unclaimed;
  unclaimed.id = "coverage/unclaimed-registers";
  unclaimed.label = "Registers written but modelled by nothing";
  unclaimed.description =
      "Each distinct (col, row, offset) the design configured that this model "
      "does not implement. A non-empty list means the design asked for "
      "behaviour the run did not provide.";
  for (const Array::UnclaimedWrite &u : array.unclaimedWrites()) {
    char off[16];
    std::snprintf(off, sizeof(off), "0x%x", u.regOff);
    CoverageItem item;
    item.key = "tile:" + std::to_string(u.col) + "," + std::to_string(u.row) +
               "/" + off;
    item.seen = true;
    unclaimed.items.push_back(std::move(item));
  }
  rec.coverages.push_back(std::move(unclaimed));

  // --- scalars.
  double touchedRatio =
      totalCapacity ? double(totalTouched) / double(totalCapacity) : 0.0;
  rec.scalars.push_back(
      {"scalar/cycles", "Simulated cycles", "",
       Quantity{double(array.cycle()), "cycles", "", {}}, Better::Lower, {}});
  rec.scalars.push_back(
      {"scalar/memory-touched", "Data memory touched", "",
       Quantity{double(totalTouched), "bytes",
                "sum of per-tile touched granules", {}},
       Better::Neither, {}});
  rec.scalars.push_back(
      {"scalar/memory-touched-ratio", "Data memory touched, of capacity", "",
       Quantity{touchedRatio, "ratio", "touched / capacity",
                {{"touched", std::to_string(totalTouched)},
                 {"capacity", std::to_string(totalCapacity)},
                 {"tiles", std::to_string(tilesWithMemory)}}},
       Better::Neither, {}});
  rec.scalars.push_back(
      {"scalar/unclaimed-registers", "Unmodelled registers written", "",
       Quantity{double(array.unclaimedWrites().size()), "count", "", {}},
       Better::Lower, {}});

  rec.headline = {"scalar/cycles", "scalar/memory-touched",
                  "scalar/unclaimed-registers"};

  // --- verdicts.
  Verdict modelled;
  modelled.id = "all-registers-modelled";
  modelled.label = "Every register the design wrote is modelled";
  modelled.evidence = {"coverage/unclaimed-registers"};
  if (array.unclaimedWrites().empty()) {
    modelled.outcome = Outcome::Pass;
    modelled.why = "No write landed on an unmodelled register.";
  } else {
    modelled.outcome = Outcome::Fail;
    modelled.severity = "error";
    modelled.why = std::to_string(array.unclaimedWrites().size()) +
                   " distinct register(s) were written but modelled by "
                   "nothing, so the design configured behaviour this run did "
                   "not provide.";
  }
  rec.verdicts.push_back(std::move(modelled));

  Verdict tracked;
  tracked.id = "memory-tracking-enabled";
  tracked.label = "Memory readings are being counted";
  tracked.evidence = {"containment/tile-memory"};
  bool anyTracked = false;
  for (uint32_t col = 0; col < dev.numCols && !anyTracked; ++col)
    for (uint32_t row = 0; row < dev.numRows && !anyTracked; ++row)
      if (Tile *t = array.tile(col, row))
        if (t->memory() && t->memory()->tracksWrites())
          anyTracked = true;
  if (anyTracked) {
    tracked.outcome = Outcome::Pass;
    tracked.why = "Write tracking was on, so touched figures are counts.";
  } else {
    // Deliberately `unknown` and not `fail`: nothing is wrong with the design,
    // the instrument was off. Collapsing the two would report a measurement
    // that was never taken as a measurement of zero.
    tracked.outcome = Outcome::Unknown;
    tracked.severity = "info";
    tracked.why = "Write tracking was off, so every touched figure is 0 "
                  "because nobody counted, not because nothing was written.";
  }
  rec.verdicts.push_back(std::move(tracked));

  return rec;
}
