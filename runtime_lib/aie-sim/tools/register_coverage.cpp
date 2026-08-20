//===- register_coverage.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Reports how much of a generation's register map this model actually claims,
// as a number rather than an impression.
//
// The universe is aie-rt's generated register database: a define is a register
// address when a sibling <NAME>_WIDTH exists and its own value is hex, which
// separates the ~2.1k addresses from the ~31k field LSB/MASK/DEFVAL defines
// around them. Coverage is then RegisterFile::isClaimed() asked once per
// address on a representative tile of each type, so the answer comes from the
// same lookup a running design hits, not from a hand-kept list that can drift.
//
// Unclaimed is not automatically a gap: much of the map is trace, debug,
// performance counters and ECC that this model deliberately does not have. The
// number is a trend line and a review aid, not a target to maximise.
//
//   register_coverage <params-header> [device-name]
//
//===----------------------------------------------------------------------===//

#include "aiesim/Array.h"
#include "aiesim/Device.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <regex>
#include <string>
#include <vector>

using namespace aiesim;

namespace {

struct Group {
  const char *module;
  TileType tile;
};

// CORE_MODULE and MEMORY_MODULE are the two halves of a core tile; PL_MODULE
// and NOC_MODULE are both shim. Names as they appear between the XAIE*GBL_
// prefix and the register name.
const Group kGroups[] = {
    {"CORE_MODULE", TileType::Core},     {"MEMORY_MODULE", TileType::Core},
    {"MEM_TILE_MODULE", TileType::MemTile}, {"PL_MODULE", TileType::Shim},
    {"NOC_MODULE", TileType::Shim},
};

const char *tileTypeName(TileType t) {
  switch (t) {
  case TileType::Core:
    return "core";
  case TileType::MemTile:
    return "memtile";
  case TileType::Shim:
    return "shim";
  default:
    return "invalid";
  }
}

/// Register addresses by module group, parsed out of an aie-rt params header.
std::map<std::string, std::vector<uint32_t>>
parseHeader(const std::string &path, std::string &error) {
  std::map<std::string, std::vector<uint32_t>> out;
  std::ifstream in(path);
  if (!in) {
    error = "cannot open " + path;
    return out;
  }

  static const std::regex kDefine(R"(^#define\s+(\S+)\s+(\S+)\s*$)");
  std::map<std::string, std::string> defines;
  std::string line;
  while (std::getline(in, line)) {
    std::smatch m;
    if (std::regex_match(line, m, kDefine))
      defines.emplace(m[1].str(), m[2].str());
  }

  static const std::regex kAddrValue(R"(^0[xX][0-9a-fA-F]+$)");
  static const std::regex kName(R"(^XAIE[A-Z0-9]*GBL_(.+)$)");
  for (const auto &[name, value] : defines) {
    if (!defines.count(name + "_WIDTH"))
      continue;
    if (!std::regex_match(value, kAddrValue))
      continue;
    std::smatch m;
    if (!std::regex_match(name, m, kName))
      continue;
    const std::string rest = m[1].str();
    for (const Group &g : kGroups) {
      const std::string prefix = std::string(g.module) + "_";
      if (rest.compare(0, prefix.size(), prefix) == 0) {
        out[g.module].push_back(
            static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 16)));
        break;
      }
    }
  }

  if (out.empty())
    error = "no register addresses found in " + path;
  return out;
}

/// First tile of the given type, which is representative: every tile of a type
/// is installed by the same install*() calls.
Tile *representative(Array &array, const DeviceModel &dev, TileType want) {
  for (uint32_t row = 0; row < dev.numRows; ++row)
    if (dev.tileTypeAt(row) == want)
      return array.tile(0, row);
  return nullptr;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: register_coverage <params-header> [device-name]\n");
    return 2;
  }
  const std::string headerPath = argv[1];
  const std::string deviceName = argc > 2 ? argv[2] : "npu2";

  std::string error;
  auto byGroup = parseHeader(headerPath, error);
  if (!error.empty()) {
    std::fprintf(stderr, "register_coverage: %s\n", error.c_str());
    return 1;
  }

  DeviceModel dev;
  if (!makeDeviceFromName(deviceName, dev, error)) {
    std::fprintf(stderr, "register_coverage: %s\n", error.c_str());
    return 1;
  }
  Array array(dev, nullptr);

  std::printf("device %s\n\n", deviceName.c_str());
  std::printf("%-18s %-8s %8s %8s %7s\n", "module", "tile", "claimed", "total",
              "pct");

  size_t grandClaimed = 0, grandTotal = 0;
  std::map<TileType, std::pair<size_t, size_t>> perTile;

  for (const Group &g : kGroups) {
    auto it = byGroup.find(g.module);
    if (it == byGroup.end())
      continue;
    Tile *tile = representative(array, dev, g.tile);
    if (!tile) {
      std::printf("%-18s %-8s %8s %8zu %7s\n", g.module, tileTypeName(g.tile),
                  "-", it->second.size(), "no tile");
      continue;
    }
    size_t claimed = 0;
    for (uint32_t off : it->second)
      if (tile->regs().isClaimed(off))
        ++claimed;
    const size_t total = it->second.size();
    std::printf("%-18s %-8s %8zu %8zu %6.1f%%\n", g.module,
                tileTypeName(g.tile), claimed, total,
                100.0 * static_cast<double>(claimed) /
                    static_cast<double>(total));
    perTile[g.tile].first += claimed;
    perTile[g.tile].second += total;
    grandClaimed += claimed;
    grandTotal += total;
  }

  std::printf("\n");
  for (const auto &[type, counts] : perTile)
    std::printf("%-18s %-8s %8zu %8zu %6.1f%%\n", "(tile total)",
                tileTypeName(type), counts.first, counts.second,
                100.0 * static_cast<double>(counts.first) /
                    static_cast<double>(counts.second));
  std::printf("%-18s %-8s %8zu %8zu %6.1f%%\n", "(all)", "", grandClaimed,
              grandTotal,
              100.0 * static_cast<double>(grandClaimed) /
                  static_cast<double>(grandTotal));
  return 0;
}
