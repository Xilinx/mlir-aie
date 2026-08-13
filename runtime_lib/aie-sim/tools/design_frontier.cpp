//===- design_frontier.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// How far a real DESIGN gets, as a number rather than an impression.
//
// elf_frontier answers a narrower question: it drops one core ELF into an
// otherwise empty array, so what it measures is the instruction path alone. It
// reports 136/136 parked, because a core with no DMA and no host configured
// waits at its first acquire forever and nothing was ever going to release it.
//
// This tool supplies exactly what that one was missing: the design's OWN
// configuration, replayed out of the CDO blobs its build emitted -- lock
// initial values, stream-switch routes, buffer descriptors, DMA task-queue
// pushes, the core images, the enables. Same corpus, same tiles, same buckets,
// so the two numbers are directly comparable and the difference between them is
// what configuration buys.
//
// The CDO is used rather than a hand-programmed BD on purpose. A hand-written
// descriptor exercises the descriptor its author thought to write; a design's
// CDO exercises every register its toolchain emits, which is the only way the
// unclaimed-register report below is a measurement instead of a formality.
//
//   design_frontier [--budget N] [--device NAME] [--verbose]
//                   <design.prj | *_aie_cdo_init.bin>...
//
// The unit is a CDO GROUP -- one `*_aie_cdo_init.bin` and its `_elfs` / `_enable`
// siblings -- because that is what the loader brings up as one hardware context.
// A `*.mlir.prj` directory argument expands to every group inside it, which for
// a fused build is one per fused op rather than one per directory.
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/CdoReplay.h"
#include "aiesim/Components.h"
#include "aiesim/CoreEngine.h"
#include "aiesim/Device.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// CORE_CONTROL/CORE_STATUS as the core's own tile sees them, and the two status
// bits that separate a finished core from a halted one. Same offsets as
// elf_frontier, deliberately: the two tools bucket the same way.
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kDone = 1u << 20;
constexpr uint32_t kErrorHalt = 1u << 19;

enum class Bucket { Fault, Unmodelled, Done, Parked, Empty };

const char *bucketName(Bucket b) {
  switch (b) {
  case Bucket::Fault:
    return "fault";
  case Bucket::Unmodelled:
    return "unmodelled";
  case Bucket::Done:
    return "done";
  case Bucket::Parked:
    return "parked";
  case Bucket::Empty:
    return "empty";
  }
  return "?";
}

struct CoreResult {
  uint32_t col = 0, row = 0;
  Bucket bucket = Bucket::Empty;
  uint32_t pc = 0;
  uint64_t engineCycles = 0;
  std::string detail;
};

struct DesignResult {
  std::string name;
  CdoReplayStats cdo;
  Array::StreamTraffic traffic;
  uint64_t arrayCycles = 0;
  bool quiescent = false;
  std::vector<CoreResult> cores;
  std::vector<std::string> diagnostics;
  size_t unclaimedSites = 0;
  /// Tiles whose DMA still has work outstanding when the run ends, and BDs
  /// completed across the array. Together they separate "the configuration
  /// never armed a DMA" from "it armed one that is waiting" -- two very
  /// different reasons for a core to sit at an acquire, indistinguishable from
  /// the core's own pc.
  uint32_t dmaTilesBusy = 0;
  uint64_t dmaBdsCompleted = 0;
};

constexpr char kInitSuffix[] = "_aie_cdo_init.bin";

bool endsWith(const std::string &s, const std::string &suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool isDirectory(const std::string &path) {
  DIR *d = opendir(path.c_str());
  if (!d)
    return false;
  closedir(d);
  return true;
}

bool fileExists(const std::string &path) {
  if (std::FILE *f = std::fopen(path.c_str(), "rb")) {
    std::fclose(f);
    return true;
  }
  return false;
}

/// Every `*_aie_cdo_init.bin` under `dir`, sorted.
///
/// One per DESIGN, not one per directory: a fused build puts every one of its
/// programs in the same `.prj` (`main_`, `op0_LayerNorm_`, `op1_GEMV_` ...),
/// and each is a separate hardware context with its own configuration. Treating
/// the directory as one design would replay them on top of each other.
std::vector<std::string> findInitBlobs(const std::string &dir) {
  DIR *d = opendir(dir.c_str());
  if (!d)
    return {};
  std::vector<std::string> hits;
  while (dirent *e = readdir(d))
    if (endsWith(e->d_name, kInitSuffix))
      hits.push_back(dir + "/" + e->d_name);
  closedir(d);
  std::sort(hits.begin(), hits.end());
  return hits;
}

/// The sibling blob of an init blob: same prefix, `_aie_cdo_<kind>.bin`. Empty
/// if the build emitted none, which is what a design with no cores looks like.
std::string sibling(const std::string &init, const char *kind) {
  std::string path = init.substr(0, init.size() - (sizeof(kInitSuffix) - 1)) +
                     "_aie_cdo_" + kind + ".bin";
  return fileExists(path) ? path : std::string{};
}

/// True if anything was loaded into this tile's program memory. A design
/// enables every core tile in its column, including ones it put no code on, and
/// those are not parked -- they never started. Same test CoreModule.cpp uses to
/// decide an enable is harmless.
bool hasProgram(Tile &tile) {
  Memory *prog = tile.programMemory();
  if (!prog)
    return false;
  const uint8_t *bytes = prog->data();
  return std::any_of(bytes, bytes + prog->size(),
                     [](uint8_t b) { return b != 0; });
}

bool run(const std::string &deviceName, const std::string &init,
         uint64_t budget, DesignResult &out, std::string &error) {
  std::unique_ptr<CoreEngineFactory> factory = loadCoreEngineFactory("", error);
  if (!factory)
    return false;

  DeviceModel dev;
  if (!makeDeviceFromName(deviceName.c_str(), dev, error))
    return false;

  const std::string elfs = sibling(init, "elfs");
  const std::string enable = sibling(init, "enable");

  Array array(dev, std::move(factory));
  // Collect diagnostics instead of aborting on the first one. A design that
  // configures something this model does not have IS the result being
  // measured, so it has to land in the report next to the others rather than
  // taking the whole sweep down.
  array.setDiagnosticHandler(
      [&out](const std::string &m) { out.diagnostics.push_back(m); });

  // The CDO addresses tiles from 0 with no base; this array decodes against
  // the host base its DeviceModel carries.
  for (const std::string &blob : {init, elfs, enable}) {
    if (blob.empty())
      continue; // A design with no cores emits no elfs/enable blob.
    if (!replayCdoFile(array, blob, dev.baseAddr, out.cdo, error))
      return false;
  }

  out.quiescent = array.runUntilQuiescent(budget);
  out.arrayCycles = array.cycle();
  out.traffic = array.streamTraffic();
  out.unclaimedSites = array.unclaimedWrites().size();

  for (uint32_t row = 0; row < dev.numRows; ++row)
    for (uint32_t col = 0; col < dev.numCols; ++col) {
      Tile *tile = array.tile(col, row);
      DmaModule *dma = tile ? tile->dma() : nullptr;
      if (!dma)
        continue;
      if (dma->busy())
        ++out.dmaTilesBusy;
      uint32_t s2mm = 0, mm2s = 0;
      switch (dev.tileTypeAt(row)) {
      case TileType::Core:
        s2mm = dev.numCoreDmaChannelsS2MM;
        mm2s = dev.numCoreDmaChannelsMM2S;
        break;
      case TileType::MemTile:
        s2mm = dev.numMemTileDmaChannelsS2MM;
        mm2s = dev.numMemTileDmaChannelsMM2S;
        break;
      case TileType::Shim:
        s2mm = dev.numShimDmaChannelsS2MM;
        mm2s = dev.numShimDmaChannelsMM2S;
        break;
      case TileType::Invalid:
        break;
      }
      for (uint32_t ch = 0; ch < s2mm; ++ch)
        out.dmaBdsCompleted += dma->completedBds(DmaDirection::S2MM, ch);
      for (uint32_t ch = 0; ch < mm2s; ++ch)
        out.dmaBdsCompleted += dma->completedBds(DmaDirection::MM2S, ch);
    }

  for (uint32_t row = 0; row < dev.numRows; ++row) {
    if (dev.tileTypeAt(row) != TileType::Core)
      continue;
    for (uint32_t col = 0; col < dev.numCols; ++col) {
      Tile *tile = array.tile(col, row);
      if (!tile || !hasProgram(*tile))
        continue;

      CoreResult r;
      r.col = col;
      r.row = row;
      const uint32_t status = tile->regs().read(kStatus);
      const CoreEngine *engine = tile->attachedCoreEngine();
      if (engine) {
        r.pc = engine->getProgramCounter();
        r.engineCycles = engine->cycleCounts().cycles;
      }

      std::string unmodelled;
      if (engine)
        for (const CoreEngine::OpcodeUse &u : engine->opcodeCoverage())
          if (!u.modelled)
            unmodelled += (unmodelled.empty() ? "" : ",") + u.name;

      // Same order as elf_frontier: a fault or a missing opcode says the model
      // stopped it, and only once neither holds does DONE mean it really ran.
      if ((status & kErrorHalt) || (engine && !engine->error().empty())) {
        r.bucket = Bucket::Fault;
        r.detail = !engine || engine->error().empty() ? "ERROR_HALT"
                                                      : engine->error();
      } else if (!unmodelled.empty()) {
        r.bucket = Bucket::Unmodelled;
        r.detail = unmodelled;
      } else if (status & kDone) {
        r.bucket = Bucket::Done;
      } else {
        r.bucket = Bucket::Parked;
      }
      out.cores.push_back(r);
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  uint64_t budget = 200000;
  std::string deviceName = "npu2";
  bool verbose = false;
  std::vector<std::string> args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--budget" && i + 1 < argc)
      budget = std::strtoull(argv[++i], nullptr, 0);
    else if (arg == "--device" && i + 1 < argc)
      deviceName = argv[++i];
    else if (arg == "--verbose")
      verbose = true;
    else
      args.push_back(arg);
  }

  if (args.empty()) {
    std::printf("usage: %s [--budget N] [--device NAME] [--verbose] "
                "<design.prj | *_aie_cdo_init.bin>...\n",
                argv[0]);
    return 2;
  }

  std::vector<std::string> inits;
  for (const std::string &arg : args) {
    if (isDirectory(arg)) {
      std::vector<std::string> found = findInitBlobs(arg);
      if (found.empty())
        std::printf("%-70s NO-CDO (no *%s)\n", arg.c_str(), kInitSuffix);
      inits.insert(inits.end(), found.begin(), found.end());
    } else {
      inits.push_back(arg);
    }
  }

  std::string error;
  if (std::unique_ptr<CoreEngineFactory> probe =
          loadCoreEngineFactory("", error))
    std::printf("engine: %s\n", probe->name().c_str());
  else {
    std::printf("%s\n", error.c_str());
    return 77;
  }

  std::map<std::string, unsigned> buckets;
  std::map<std::string, unsigned> details;
  unsigned failedDesigns = 0, designs = 0;
  Array::StreamTraffic total;

  for (const std::string &init : inits) {
    DesignResult d;
    d.name = init;
    std::string why;
    if (!run(deviceName, init, budget, d, why)) {
      ++failedDesigns;
      std::printf("%-70s REPLAY-FAILED %s\n", init.c_str(), why.c_str());
      continue;
    }
    ++designs;

    std::map<std::string, unsigned> per;
    for (const CoreResult &c : d.cores) {
      ++buckets[bucketName(c.bucket)];
      ++per[bucketName(c.bucket)];
      if (!c.detail.empty())
        ++details[c.detail];
    }
    total.ddrRead += d.traffic.ddrRead;
    total.ddrWrite += d.traffic.ddrWrite;
    total.l2Read += d.traffic.l2Read;
    total.l2Write += d.traffic.l2Write;
    total.l1Read += d.traffic.l1Read;
    total.l1Write += d.traffic.l1Write;

    std::string summary;
    for (const auto &b : per)
      summary += " " + b.first + "=" + std::to_string(b.second);
    std::printf("%-70s cores%s cycles=%llu%s\n", init.c_str(), summary.c_str(),
                (unsigned long long)d.arrayCycles,
                d.quiescent ? "" : " (budget exhausted)");
    if (verbose) {
      std::printf(
          "    cdo: write=%u maskwrite=%u block=%u(%llu words) poll=%u(%u "
          "timed out) nop=%u\n",
          d.cdo.write32, d.cdo.maskWrite32, d.cdo.blockWrite32,
          (unsigned long long)d.cdo.blockWriteWords, d.cdo.maskPoll,
          d.cdo.maskPollTimedOut, d.cdo.noOp);
      std::printf("    dma: %u tile(s) still have work outstanding, %llu BD(s) "
                  "completed\n",
                  d.dmaTilesBusy, (unsigned long long)d.dmaBdsCompleted);
      std::printf("    bytes: l1 %llu/%llu l2 %llu/%llu ddr %llu/%llu "
                  "(read/write), unclaimed sites %zu\n",
                  (unsigned long long)d.traffic.l1Read,
                  (unsigned long long)d.traffic.l1Write,
                  (unsigned long long)d.traffic.l2Read,
                  (unsigned long long)d.traffic.l2Write,
                  (unsigned long long)d.traffic.ddrRead,
                  (unsigned long long)d.traffic.ddrWrite, d.unclaimedSites);
      for (const CoreResult &c : d.cores)
        std::printf("    tile (%u, %u) %-10s pc=0x%X engine=%llu %s\n", c.col,
                    c.row, bucketName(c.bucket), c.pc,
                    (unsigned long long)c.engineCycles, c.detail.c_str());
      for (const std::string &m : d.diagnostics)
        std::printf("    diag: %s\n", m.c_str());
    } else if (!d.diagnostics.empty()) {
      std::printf("    %zu diagnostic(s); first: %s\n", d.diagnostics.size(),
                  d.diagnostics.front().c_str());
    }
  }

  unsigned cores = 0;
  for (const auto &b : buckets)
    cores += b.second;
  std::printf("\n%u design%s (%u core%s with a program), budget %llu cycles, "
              "device %s\n",
              designs, designs == 1 ? "" : "s", cores, cores == 1 ? "" : "s",
              (unsigned long long)budget, deviceName.c_str());
  for (const auto &b : buckets)
    std::printf("  %-10s %u\n", b.first.c_str(), b.second);
  if (failedDesigns)
    std::printf("  %-10s %u\n", "replay-failed", failedDesigns);
  for (const auto &d : details)
    std::printf("  detail: %s (%u)\n", d.first.c_str(), d.second);
  std::printf("  bytes moved: l1 %llu/%llu l2 %llu/%llu ddr %llu/%llu "
              "(read/write)\n",
              (unsigned long long)total.l1Read,
              (unsigned long long)total.l1Write,
              (unsigned long long)total.l2Read,
              (unsigned long long)total.l2Write,
              (unsigned long long)total.ddrRead,
              (unsigned long long)total.ddrWrite);

  // A fault or a missing opcode is a gap in the model; parked is not, for the
  // same reason elf_frontier says so.
  return (buckets["fault"] || buckets["unmodelled"] || failedDesigns) ? 1 : 0;
}
