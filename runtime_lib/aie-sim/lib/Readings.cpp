//===- Readings.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "aiesim/Readings.h"

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/RegionMap.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <map>

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
  for (const Flow &f : flows) {
    j.beginObj();
    j.kv("id", f.id);
    j.kv("shape", "flow");
    j.kv("label", f.label);
    j.kv("points", uint64_t(f.edges.size()));
    j.kvOpt("unit", f.unit);
    j.endObj();
  }
  for (const Graph &g : graphs) {
    j.beginObj();
    j.kv("id", g.id);
    j.kv("shape", "graph");
    j.kv("label", g.label);
    j.kv("points", uint64_t(g.nodes.size() + g.edges.size()));
    j.endObj();
  }
  for (const Series &sr : series) {
    uint64_t pts = 0;
    for (const SeriesTrack &tr : sr.series)
      pts += tr.points.size();
    j.beginObj();
    j.kv("id", sr.id);
    j.kv("shape", "series");
    j.kv("label", sr.label);
    j.kv("points", pts);
    j.kvOpt("unit", sr.unit);
    j.endObj();
  }
  for (const Interval &iv : intervals) {
    uint64_t spans = 0;
    for (const IntervalTrack &t : iv.tracks)
      spans += t.spans.size();
    j.beginObj();
    j.kv("id", iv.id);
    j.kv("shape", "interval");
    j.kv("label", iv.label);
    j.kv("points", spans);
    j.kvOpt("unit", iv.timeUnit);
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

  j.key("flow");
  j.beginArr();
  for (const Flow &f : flows) {
    j.beginObj();
    j.kv("id", f.id);
    j.kv("label", f.label);
    j.kvOpt("description", f.description);
    j.strArray("tags", f.tags);
    j.kv("unit", f.unit);
    j.key("nodes");
    j.beginArr();
    for (const FlowNode &n : f.nodes) {
      j.beginObj();
      j.kv("id", n.id);
      j.kvOpt("label", n.label);
      j.kvOpt("level", n.level);
      j.endObj();
    }
    j.endArr();
    j.key("edges");
    j.beginArr();
    for (const FlowEdge &e : f.edges) {
      j.beginObj();
      j.kv("from", e.from);
      j.kv("to", e.to);
      j.kv("value", e.value);
      j.kvOpt("label", e.label);
      j.endObj();
    }
    j.endArr();
    j.endObj();
  }
  j.endArr();

  j.key("graph");
  j.beginArr();
  for (const Graph &g : graphs) {
    j.beginObj();
    j.kv("id", g.id);
    j.kv("label", g.label);
    j.kvOpt("description", g.description);
    j.strArray("tags", g.tags);
    j.kvBool("directed", g.directed);
    j.key("nodes");
    j.beginArr();
    for (const GraphNode &n : g.nodes) {
      j.beginObj();
      j.kv("id", n.id);
      j.kvOpt("label", n.label);
      j.kvOpt("kind", n.kind);
      j.endObj();
    }
    j.endArr();
    j.key("edges");
    j.beginArr();
    for (const GraphEdge &e : g.edges) {
      j.beginObj();
      j.kv("from", e.from);
      j.kv("to", e.to);
      j.kvOpt("label", e.label);
      j.kvOpt("kind", e.kind);
      j.endObj();
    }
    j.endArr();
    j.endObj();
  }
  j.endArr();

  j.key("series");
  j.beginArr();
  for (const Series &sr : series) {
    j.beginObj();
    j.kv("id", sr.id);
    j.kv("label", sr.label);
    j.kvOpt("description", sr.description);
    j.strArray("tags", sr.tags);
    j.kv("unit", sr.unit);
    j.kv("timeUnit", sr.timeUnit);
    j.key("series");
    j.beginArr();
    for (const SeriesTrack &tr : sr.series) {
      j.beginObj();
      j.kv("entity", tr.entity);
      j.key("points");
      j.beginArr();
      for (const auto &[t2, v] : tr.points) {
        j.beginArr();
        j.u64(t2);
        j.num(v);
        j.endArr();
      }
      j.endArr();
      j.endObj();
    }
    j.endArr();
    j.endObj();
  }
  j.endArr();

  j.key("interval");
  j.beginArr();
  for (const Interval &iv : intervals) {
    j.beginObj();
    j.kv("id", iv.id);
    j.kv("label", iv.label);
    j.kvOpt("description", iv.description);
    j.strArray("tags", iv.tags);
    j.kv("timeUnit", iv.timeUnit);
    j.key("categories");
    j.beginArr();
    for (const IntervalCategory &cat : iv.categories) {
      j.beginObj();
      j.kv("name", cat.name);
      j.kvOpt("color", cat.color);
      j.kvBool("productive", cat.productive);
      j.endObj();
    }
    j.endArr();
    j.key("tracks");
    j.beginArr();
    for (const IntervalTrack &t : iv.tracks) {
      j.beginObj();
      j.kv("entity", t.entity);
      j.key("spans");
      j.beginArr();
      for (const IntervalSpan &s : t.spans) {
        j.beginObj();
        j.kv("start", s.start);
        j.kv("end", s.end);
        j.kv("category", s.category);
        j.endObj();
      }
      j.endArr();
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
///
/// With a region map attached, the leaf becomes a parent whose children are
/// the linker script's own allocations. That is the difference between "this
/// tile used 12 KB" and "the stack is 3328 bytes and the buffer starts at the
/// byte after it".
bool addMemoryLeaf(ContainmentNode &tileNode, const char *name, Memory *mem,
                   bool includeUntouched, const RegionMap *regions,
                   uint32_t bandBase) {
  if (!mem)
    return false;
  uint32_t touched = mem->tracksWrites() ? mem->touchedBytes() : 0;
  if (!touched && !includeUntouched && !regions)
    return false;
  ContainmentNode node;
  node.name = name;
  node.hasCapacity = true;
  node.capacity = mem->size();
  if (!mem->tracksWrites())
    // Otherwise a 0 here reads as "nothing was written" rather than "nobody
    // was counting", which is the same silent-zero mistake the register file
    // exists to prevent.
    node.attrs.push_back({"tracked", "false"});

  bool named = false;
  if (regions) {
    // The address this memory starts at, so an address-ordered view can place
    // its children without inferring a base from their own addresses.
    char base[16];
    std::snprintf(base, sizeof(base), "0x%x", bandBase);
    node.attrs.push_back({"base", base});
    for (const Region &r : regions->regions()) {
      // Only what this band maps: the script also describes the neighbour
      // bands and program memory, which are not this memory's bytes.
      if (r.kind == RegionKind::Program || r.kind == RegionKind::Data)
        continue;
      if (r.begin < bandBase || r.begin >= bandBase + mem->size())
        continue;
      uint32_t off = r.begin - bandBase;
      ContainmentNode leaf;
      leaf.name = r.name;
      leaf.hasValue = true;
      leaf.value = mem->tracksWrites() ? mem->touchedBytesIn(off, r.size) : 0;
      leaf.hasCapacity = true;
      leaf.capacity = r.size;
      leaf.attrs.push_back({"kind", regionKindName(r.kind)});
      char addr[16];
      std::snprintf(addr, sizeof(addr), "0x%x", r.begin);
      leaf.attrs.push_back({"addr", addr});
      node.children.push_back(std::move(leaf));
      named = true;
    }
  }
  if (!named) {
    node.hasValue = true;
    node.value = touched;
  }
  tileNode.children.push_back(std::move(node));
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
      const RegionMap *regions = t->regionMap();
      bool any = addMemoryLeaf(tileNode, "data", t->memory(),
                               config.includeUntouchedTiles, regions,
                               ownMemoryBandBase(dev.generation));
      any |= addMemoryLeaf(tileNode, "program", t->programMemory(),
                           config.includeUntouchedTiles, nullptr, 0);
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

  // --- coverage: instruction semantics, over everything the cores reached.
  //
  // Two sources, merged. The engines accumulate the whole set they executed as
  // they go, which is the real answer; the fault path contributes the
  // instruction a run stopped on, which is all there is when the engine does
  // not report coverage. `seen` means "the engine has semantics for it", so
  // the unseen entries are the gap.
  std::map<std::string, bool> opcodeSeen;
  bool anyEngineReported = false;
  for (uint32_t col = 0; col < dev.numCols; ++col)
    for (uint32_t row = 0; row < dev.numRows; ++row)
      if (Tile *t = array.tile(col, row))
        if (const CoreEngine *eng = t->attachedCoreEngine())
          for (const CoreEngine::OpcodeUse &u : eng->opcodeCoverage()) {
            anyEngineReported = true;
            // Modelled anywhere is modelled: the same opcode cannot be
            // implemented on one core and missing on another.
            auto [it, inserted] = opcodeSeen.emplace(u.name, u.modelled);
            if (!inserted)
              it->second = it->second || u.modelled;
          }
  for (const Array::UnmodelledOpcode &u : array.unmodelledOpcodes())
    opcodeSeen.emplace(u.name, false);

  Coverage semantics;
  semantics.id = "coverage/opcode-semantics";
  semantics.label = "Instruction semantics over the opcodes the cores reached";
  semantics.description =
      "Every distinct instruction a core reached, marked with whether the "
      "engine had semantics for it. The unseen entries are the semantics gap "
      "as a number. This covers what the run REACHED, not what the design "
      "contains: a run that never entered the vector body says nothing about "
      "the vector body.";
  // Which source filled it decides how complete it is, and a consumer cannot
  // tell from the items. With no engine enumerating what it executed, all
  // there is is the instruction the run died on, so the set is a lower bound.
  semantics.description +=
      anyEngineReported
          ? " Enumerated by the core engine, so it is every opcode executed."
          : " No engine reported coverage, so this is only what the fault path "
            "saw: a lower bound, not the full set the run executed.";
  for (const auto &[name, seen] : opcodeSeen) {
    CoverageItem item;
    item.key = name;
    item.seen = seen;
    semantics.items.push_back(std::move(item));
  }
  // universe is the count reached, which is a denominator that exists. The
  // size of the ISA would be an invented one: coverage against it would say
  // more about the ISA than about this run.
  semantics.universe = opcodeSeen.size();
  rec.coverages.push_back(std::move(semantics));

  // --- flow: bytes each memory level exchanged with the stream fabric.
  //
  // Counted where a byte actually crosses -- one DMA channel's word transfer --
  // so this is what the design moved, not what its descriptors said it would.
  // The level is the DMA's own end: shim is DDR, memtile L2, core tile L1.
  const Array::StreamTraffic &tr = array.streamTraffic();
  const uint64_t ddrBytes = tr.ddrRead + tr.ddrWrite;
  Flow moved;
  moved.id = "flow/stream-traffic";
  moved.label = "Bytes moved between each memory level and the stream fabric";
  moved.unit = "bytes";
  moved.description =
      "Which memory level each DMA transfer had at its own end. This does NOT "
      "say which L2 buffer a given DDR read eventually fed -- that is a "
      "stream-switch routing question -- so the edges join each level to the "
      "fabric rather than to each other.";
  moved.nodes = {{"ddr", "DDR", "L3"},
                 {"l2", "MemTile", "L2"},
                 {"l1", "Core tile", "L1"},
                 {"fabric", "Stream fabric", "fabric"}};
  auto edge = [&](const char *from, const char *to, uint64_t bytes,
                  const char *label) {
    if (bytes)
      moved.edges.push_back({from, to, label, double(bytes)});
  };
  edge("ddr", "fabric", tr.ddrRead, "shim MM2S");
  edge("fabric", "ddr", tr.ddrWrite, "shim S2MM");
  edge("l2", "fabric", tr.l2Read, "memtile MM2S");
  edge("fabric", "l2", tr.l2Write, "memtile S2MM");
  edge("l1", "fabric", tr.l1Read, "core MM2S");
  edge("fabric", "l1", tr.l1Write, "core S2MM");
  rec.flows.push_back(std::move(moved));

  // --- graph: the dataflow the configuration code actually programmed.
  //
  // Read out of the stream switches rather than out of the design's source,
  // which is this model's whole point: a routing bug is invisible to a
  // graph-level simulator because that one is handed the intended graph. Here
  // an edge exists because a register says so.
  Graph routes;
  routes.id = "graph/stream-routes";
  routes.label = "Circuit-switched routes the design configured";
  routes.directed = true;
  uint32_t packetMasters = 0;
  std::map<std::string, GraphNode> nodes;
  for (uint32_t col = 0; col < dev.numCols; ++col)
    for (uint32_t row = 0; row < dev.numRows; ++row) {
      Tile *t = array.tile(col, row);
      if (!t || !t->streamSwitch())
        continue;
      packetMasters += t->streamSwitch()->packetModeMasters();
      const std::string where =
          std::to_string(col) + "," + std::to_string(row);
      for (const StreamRoute &r : t->streamSwitch()->configuredRoutes()) {
        auto port = [&](PortBundle b, uint32_t i) {
          std::string name = std::string(portBundleName(b)) + std::to_string(i);
          std::string id = "tile:" + where + "/" + name;
          nodes.emplace(id, GraphNode{id, name + " @ (" + where + ")", "port"});
          return id;
        };
        routes.edges.push_back({port(r.slaveBundle, r.slaveIndex),
                                port(r.masterBundle, r.masterIndex), "", "circuit"});
      }
    }
  for (auto &[id, n] : nodes)
    routes.nodes.push_back(n);
  routes.description =
      "One edge per enabled circuit-mode master, taken from the switch "
      "registers, so it is what the configuration built rather than what the "
      "design meant. Packet-mode masters are ABSENT: a packet master names an "
      "arbiter and an msel mask, not one slave, so its source is decided at "
      "run time. " +
      std::to_string(packetMasters) +
      " packet-mode master(s) are configured and not shown here.";
  rec.graphs.push_back(std::move(routes));

  // --- series: how many components were scheduled, over time.
  //
  // The interval reading says where ONE component's time went; this says how
  // many were running at once, which is the axis a serialised design shows up
  // on. Sampled at change points only -- it is a step function, so that is
  // lossless and an idle stretch costs one entry instead of millions.
  if (array.timelineEnabled() && !array.concurrency().empty()) {
    Series conc;
    conc.id = "series/active-components";
    conc.label = "Components scheduled per cycle";
    conc.unit = "count";
    conc.timeUnit = "cycle";
    conc.description =
        "Change points, not samples: a value holds from its cycle until the "
        "next entry. Counts components the scheduler ran, which includes ones "
        "that stalled -- pair it with interval/stall-attribution to separate "
        "busy from merely scheduled.";
    SeriesTrack track;
    track.entity = "array";
    for (const Array::ConcurrencyPoint &p : array.concurrency())
      track.points.push_back({p.cycle, double(p.active)});
    conc.series.push_back(std::move(track));
    rec.series.push_back(std::move(conc));
  }

  // --- interval: where each component's scheduled cycles went.
  //
  // On a machine with no interlocks the core datapath issues one bundle per
  // cycle unconditionally, so bundle count already IS its cycle count
  // (docs/kb/aie-has-no-interlocks-so-bundles-are-cycles). What can still make
  // a cycle cost more than a bundle is waiting on the fabric -- a lock, a full
  // or empty stream port -- and that is what this attributes.
  Interval stalls;
  stalls.id = "interval/stall-attribution";
  stalls.label = "Where each component's scheduled cycles went";
  stalls.timeUnit = "cycle";
  stalls.categories = {{"running", "", true},
                       {"lock", "", false},
                       {"backpressure", "", false},
                       {"starvation", "", false},
                       {"core-wait", "", false},
                       {"unknown", "", false}};
  uint64_t stalledCycles = 0, runningCycles = 0, unattributedCycles = 0;
  for (const Array::TimelineTrack &t : array.timeline()) {
    IntervalTrack track;
    track.entity = t.entity;
    for (const Array::TimelineSpan &s : t.spans) {
      track.spans.push_back({s.start, s.end, s.category});
      const uint64_t width = s.end - s.start;
      if (std::string_view(s.category) == "running")
        runningCycles += width;
      else {
        stalledCycles += width;
        if (std::string_view(s.category) == "unknown")
          unattributedCycles += width;
      }
    }
    stalls.tracks.push_back(std::move(track));
  }
  stalls.description =
      "Half-open spans over the cycles each component was scheduled. Cycles a "
      "component was NOT scheduled are absent rather than zero-filled: out of "
      "the active set it has nothing to do, which is not the same as waiting. "
      "A DMA's channels share one Steppable, so its track is per module and "
      "the reason is the first stalled channel's.";
  if (array.timelineEnabled())
    rec.intervals.push_back(std::move(stalls));

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
  uint64_t opcodeGap = 0;
  for (const auto &[name, seen] : opcodeSeen)
    if (!seen)
      ++opcodeGap;
  rec.scalars.push_back(
      {"scalar/unmodelled-opcodes", "Instructions with no semantics", "",
       Quantity{double(opcodeGap), "count", "unseen in coverage/opcode-semantics",
                {{"reached", std::to_string(opcodeSeen.size())}}},
       Better::Lower, {}});

  // Denominator is scheduled cycles, not simulated cycles: a component that is
  // not scheduled is idle, and dividing by wall time would report an array
  // that finished early as one that stalled.
  const uint64_t scheduledCycles = runningCycles + stalledCycles;
  if (array.timelineEnabled())
    rec.scalars.push_back(
        {"scalar/stalled-cycles", "Component-cycles spent waiting", "",
         Quantity{double(stalledCycles), "cycles",
                  "component-cycles scheduled but not doing work",
                  {{"scheduled", std::to_string(scheduledCycles)},
                   {"running", std::to_string(runningCycles)},
                   {"unattributed", std::to_string(unattributedCycles)}}},
         Better::Lower, {}});

  rec.scalars.push_back(
      {"scalar/ddr-bytes", "Bytes crossing DDR", "",
       Quantity{double(ddrBytes), "bytes", "shim DMA transfers, both directions",
                {{"read", std::to_string(tr.ddrRead)},
                 {"write", std::to_string(tr.ddrWrite)},
                 {"l2", std::to_string(tr.l2Read + tr.l2Write)},
                 {"l1", std::to_string(tr.l1Read + tr.l1Write)}}},
       Better::Lower, {}});

  rec.headline = {"scalar/cycles", "scalar/memory-touched",
                  "scalar/unclaimed-registers", "scalar/unmodelled-opcodes",
                  "scalar/ddr-bytes"};
  if (array.timelineEnabled())
    rec.headline.push_back("scalar/stalled-cycles");

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

  Verdict semanticsComplete;
  semanticsComplete.id = "all-opcodes-modelled";
  semanticsComplete.label = "Every instruction the design executed is modelled";
  semanticsComplete.evidence = {"coverage/opcode-semantics"};
  if (opcodeSeen.empty()) {
    // No core reached an instruction, so nothing was tested. `unknown` rather
    // than `pass`: a design that never ran must not read as a design whose
    // instructions are all modelled.
    semanticsComplete.outcome = Outcome::Unknown;
    semanticsComplete.why = "No core executed an instruction, so instruction "
                            "semantics were not exercised.";
  } else if (opcodeGap == 0) {
    // Deliberately not a claim about the whole design: a run that never
    // reached the vector body proves nothing about the vector body.
    semanticsComplete.outcome = Outcome::Pass;
    semanticsComplete.why = "All " + std::to_string(opcodeSeen.size()) +
                            " instruction(s) the run reached are modelled.";
  } else {
    semanticsComplete.outcome = Outcome::Fail;
    semanticsComplete.severity = "error";
    std::string first;
    for (const auto &[name, seen] : opcodeSeen)
      if (!seen) { first = name; break; }
    semanticsComplete.why =
        std::to_string(opcodeGap) + " of " + std::to_string(opcodeSeen.size()) +
        " instruction(s) the run reached have no semantics, starting with '" +
        first + "'.";
  }
  rec.verdicts.push_back(std::move(semanticsComplete));

  // Not "was there a stall" -- stalling is how the fabric synchronises, and a
  // design with none would be suspicious. This asks whether the stalls that
  // happened were ATTRIBUTED, because an unexplained wait is the reading
  // silently under-reporting rather than the design behaving well.
  Verdict attributed;
  attributed.id = "stalls-attributed";
  attributed.label = "Every stalled cycle has a reason";
  // Evidence only when the readings it names were actually emitted: citing an
  // observation that is not in the record sends a consumer looking for it.
  if (array.timelineEnabled())
    attributed.evidence = {"interval/stall-attribution",
                           "scalar/stalled-cycles"};
  if (!array.timelineEnabled()) {
    attributed.outcome = Outcome::Unknown;
    attributed.why = "Stall attribution was switched off for this run.";
  } else if (scheduledCycles == 0) {
    attributed.outcome = Outcome::Unknown;
    attributed.why = "No component was ever scheduled, so no time was "
                     "attributed.";
  } else if (unattributedCycles == 0) {
    attributed.outcome = Outcome::Pass;
    attributed.why = std::to_string(stalledCycles) + " stalled cycle(s) of " +
                     std::to_string(scheduledCycles) +
                     " scheduled, all with a named reason.";
  } else {
    attributed.outcome = Outcome::Fail;
    attributed.why = std::to_string(unattributedCycles) + " of " +
                     std::to_string(stalledCycles) +
                     " stalled cycle(s) have no reason, so the breakdown is "
                     "incomplete.";
  }
  rec.verdicts.push_back(std::move(attributed));

  // The "inside a block the stream never leaves L2" invariant, as a check
  // rather than a habit. Deliberately not "DDR is bad": weights and the first
  // upload have to come from somewhere. What it answers is whether THIS run
  // moved anything across DDR, which is the question a resident-block claim
  // stands or falls on.
  Verdict resident;
  resident.id = "stream-stays-on-chip";
  resident.label = "No data crossed DDR during this run";
  resident.evidence = {"flow/stream-traffic", "scalar/ddr-bytes"};
  const uint64_t onChip = tr.l2Read + tr.l2Write + tr.l1Read + tr.l1Write;
  if (ddrBytes == 0 && onChip == 0) {
    // No DMA moved at all, so residency was never put to the test.
    resident.outcome = Outcome::Unknown;
    resident.why = "No DMA transfer happened, so nothing was resident or "
                   "otherwise.";
  } else if (ddrBytes == 0) {
    resident.outcome = Outcome::Pass;
    resident.why = "All " + std::to_string(onChip) +
                   " byte(s) moved stayed between L1 and L2.";
  } else {
    resident.outcome = Outcome::Fail;
    resident.why = std::to_string(ddrBytes) + " byte(s) crossed DDR against " +
                   std::to_string(onChip) + " on-chip.";
  }
  rec.verdicts.push_back(std::move(resident));

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

  // --- stack clearance, and any two regions claiming one byte.
  //
  // Static: it reads the linker script, so it fires before a single cycle is
  // simulated and does not need a core engine. That matters because the bug it
  // describes produces no crash at run time -- the overwrite is silent, and by
  // the time a number looks wrong the cause is many layers away.
  uint32_t worstGap = 0;
  std::string worstNext, worstTile;
  bool haveGap = false;
  uint32_t overlapTotal = 0;
  std::string firstOverlap;
  uint32_t mappedTiles = 0;

  for (uint32_t col = 0; col < dev.numCols; ++col)
    for (uint32_t row = 0; row < dev.numRows; ++row) {
      Tile *t = array.tile(col, row);
      if (!t || !t->regionMap())
        continue;
      ++mappedTiles;
      const RegionMap &rm = *t->regionMap();
      std::string tileName =
          "tile:" + std::to_string(col) + "," + std::to_string(row);

      uint32_t gap = 0;
      std::string next;
      if (rm.stackClearance(gap, next))
        if (!haveGap || gap < worstGap) {
          haveGap = true;
          worstGap = gap;
          worstNext = next;
          worstTile = tileName;
        }
      for (const RegionMap::Overlap &o : rm.overlaps()) {
        ++overlapTotal;
        if (firstOverlap.empty())
          firstOverlap = tileName + " '" + o.a + "' and '" + o.b + "'";
      }
    }

  Verdict clearance;
  clearance.id = "stack-clearance";
  clearance.label = "The stack has room to overrun into";
  clearance.evidence = {"containment/tile-memory"};
  if (!mappedTiles) {
    clearance.outcome = Outcome::Unknown;
    clearance.severity = "info";
    clearance.why = "No linker script was attached, so no region is known.";
  } else if (!haveGap) {
    clearance.outcome = Outcome::Unknown;
    clearance.severity = "info";
    clearance.why = "No stack reservation was found in the attached script(s).";
  } else if (worstGap == 0) {
    clearance.outcome = Outcome::Fail;
    clearance.severity = "error";
    clearance.why = "On " + worstTile +
                    " the first byte past the stack is '" + worstNext +
                    "', so a one-byte frame overrun silently overwrites it.";
  } else {
    clearance.outcome = Outcome::Pass;
    clearance.why = "Smallest gap between a stack and the next region is " +
                    std::to_string(worstGap) + " byte(s), on " + worstTile + ".";
  }
  rec.verdicts.push_back(std::move(clearance));

  Verdict disjoint;
  disjoint.id = "regions-disjoint";
  disjoint.label = "No two regions claim the same byte";
  disjoint.evidence = {"containment/tile-memory"};
  if (!mappedTiles) {
    disjoint.outcome = Outcome::Unknown;
    disjoint.severity = "info";
    disjoint.why = "No linker script was attached.";
  } else if (overlapTotal) {
    disjoint.outcome = Outcome::Fail;
    disjoint.severity = "error";
    disjoint.why = std::to_string(overlapTotal) +
                   " overlapping region pair(s); whichever writes second wins "
                   "silently. First: " + firstOverlap + ".";
  } else {
    disjoint.outcome = Outcome::Pass;
    disjoint.why = "Every allocation in the attached script(s) is disjoint.";
  }
  rec.verdicts.push_back(std::move(disjoint));

  return rec;
}
