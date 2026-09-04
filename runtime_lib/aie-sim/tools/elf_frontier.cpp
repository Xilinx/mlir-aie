//===- elf_frontier.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// How far a real kernel gets, as a number rather than an impression.
//
// The corpus is core ELFs a compiler already emitted, so this measures the
// model against what designs actually contain instead of against hand-written
// probes, which only ever reach what their author thought to write.
//
// Each ELF is loaded into a tile and run to a cycle budget, and lands in
// exactly one bucket:
//
//   fault       the model refused something -- `error()` names it
//   unmodelled  the engine reached an opcode it has no semantics for
//   done        the core set DONE, so the whole program ran
//   parked      the core is still busy at a pc, having neither faulted nor
//               finished: it is waiting on state nothing in this run supplies
//
// `parked` is the interesting one and is NOT a defect. A core with no DMA and
// no host configured waits at its first acquire forever, which is the correct
// behaviour, so a corpus that parks says the instruction path is not what is
// missing. Converting parked into done is what configuration work buys, and
// this is the number that shows it moving.
//
// The two cycle counters are reported side by side because they answer
// different questions and their disagreement is the evidence a core is
// blocked rather than stopped: the array clock advances while a waiting core's
// own counter stands still.
//
//   elf_frontier [--budget N] [--device NAME] <core.elf>...
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Components.h"
#include "aiesim/CoreEngine.h"
#include "aiesim/Device.h"
#include "aiesim/ElfLoader.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

// CORE_CONTROL/CORE_STATUS as the core's own tile sees them, and the two status
// bits that separate a finished core from a halted one.
constexpr uint32_t kControl = 0x32000;
constexpr uint32_t kStatus = 0x32004;
constexpr uint32_t kEnable = 1u << 0;
constexpr uint32_t kReset = 1u << 1;
constexpr uint32_t kDone = 1u << 20;
constexpr uint32_t kErrorHalt = 1u << 19;

enum class Bucket { Fault, Unmodelled, Done, Parked };

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
  }
  return "?";
}

struct Result {
  Bucket bucket = Bucket::Parked;
  uint32_t pc = 0;
  uint64_t engineCycles = 0;
  uint64_t arrayCycles = 0;
  std::string detail;
};

/// A fresh array per ELF: lock counters and memory carry over otherwise, and
/// the starting state is part of what is under test.
bool run(const std::string &deviceName, const std::string &path,
         uint64_t budget, Result &out, std::string &error) {
  std::unique_ptr<CoreEngineFactory> factory = loadCoreEngineFactory("", error);
  if (!factory)
    return false;

  DeviceModel dev;
  if (!makeDeviceFromName(deviceName.c_str(), dev, error))
    return false;

  Array array(dev, std::move(factory));
  Tile *tile = array.tile(7, 3);
  if (!tile) {
    error = "device " + deviceName + " has no core tile at (7, 3)";
    return false;
  }

  uint32_t entry = 0;
  if (!loadCoreElfFile(*tile, path.c_str(), entry, error))
    return false;

  tile->regs().write(kControl, kReset);
  tile->regs().write(kControl, kEnable);
  array.runUntilQuiescent(budget);

  const uint32_t status = tile->regs().read(kStatus);
  out.arrayCycles = array.cycle();

  const CoreEngine *engine = tile->attachedCoreEngine();
  if (!engine) {
    error = "no core engine attached after enable";
    return false;
  }
  out.pc = engine->getProgramCounter();
  out.engineCycles = engine->cycleCounts().cycles;

  std::string unmodelled;
  for (const CoreEngine::OpcodeUse &u : engine->opcodeCoverage())
    if (!u.modelled)
      unmodelled += (unmodelled.empty() ? "" : ",") + u.name;

  // Ordered by what explains the run: a fault or a missing opcode says the
  // model stopped it, and only once neither holds does DONE mean the program
  // really ran to the end.
  if ((status & kErrorHalt) || !engine->error().empty()) {
    out.bucket = Bucket::Fault;
    out.detail = engine->error().empty() ? "ERROR_HALT" : engine->error();
  } else if (!unmodelled.empty()) {
    out.bucket = Bucket::Unmodelled;
    out.detail = unmodelled;
  } else if (status & kDone) {
    out.bucket = Bucket::Done;
  } else {
    out.bucket = Bucket::Parked;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  uint64_t budget = 20000;
  std::string deviceName = "npu2";
  std::vector<std::string> paths;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--budget" && i + 1 < argc)
      budget = std::strtoull(argv[++i], nullptr, 0);
    else if (arg == "--device" && i + 1 < argc)
      deviceName = argv[++i];
    else
      paths.push_back(arg);
  }

  if (paths.empty()) {
    std::printf("usage: %s [--budget N] [--device NAME] <core.elf>...\n",
                argv[0]);
    return 2;
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
  unsigned failed = 0;

  for (const std::string &path : paths) {
    Result r;
    std::string why;
    if (!run(deviceName, path, budget, r, why)) {
      ++failed;
      std::printf("%-60s LOAD-FAILED %s\n", path.c_str(), why.c_str());
      continue;
    }
    ++buckets[bucketName(r.bucket)];
    if (!r.detail.empty())
      ++details[r.detail];
    std::printf("%-60s %-10s pc=0x%X engine=%llu array=%llu %s\n", path.c_str(),
                bucketName(r.bucket), r.pc,
                (unsigned long long)r.engineCycles,
                (unsigned long long)r.arrayCycles, r.detail.c_str());
  }

  std::printf("\n%zu ELF%s, budget %llu cycles, device %s\n", paths.size(),
              paths.size() == 1 ? "" : "s", (unsigned long long)budget,
              deviceName.c_str());
  for (const auto &b : buckets)
    std::printf("  %-10s %u\n", b.first.c_str(), b.second);
  if (failed)
    std::printf("  %-10s %u\n", "load-failed", failed);
  for (const auto &d : details)
    std::printf("  detail: %s (%u)\n", d.first.c_str(), d.second);

  // A fault or a missing opcode is a gap in the model; parked is not.
  return (buckets["fault"] || buckets["unmodelled"] || failed) ? 1 : 0;
}
