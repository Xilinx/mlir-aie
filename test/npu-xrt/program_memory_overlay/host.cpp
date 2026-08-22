//===- host.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Host side of the program-memory overlay tests.
//
// Each dummy overlay computes out[i] = in[i] + tag for its own tag, so what ran
// in a phase is readable straight off the output and there is no reference run
// to build or keep in step. --tags says which overlay is expected in which
// phase, so a permuted or replayed order is expressed by permuting the list.
//
// A phase that ran the wrong overlay reports the tag it actually saw. That is
// the difference between "phase 2 is wrong" and "phase 2 ran overlay 1", and
// with a poisoned slot it also distinguishes "no overlay was loaded" from "the
// wrong one was".

#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Written into every slot during setup, so a phase whose payload never lands
// produces this rather than whatever the previous xclbin left behind. Must
// match POISON_TAG in pmlib/design.py.
constexpr int32_t POISON_TAG = 0x7BAD;

// "11,22,33" -> those three; "11*123" -> 123 copies of 11. The repeat form
// mirrors --phases, and exists so a run with a hundred-odd phases stays legible
// in a RUN line.
static std::vector<int32_t> parse_tags(const std::string &s) {
  std::vector<int32_t> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty())
      continue;
    const size_t star = item.find('*');
    if (star == std::string::npos) {
      out.push_back(std::stoi(item));
      continue;
    }
    const int32_t tag = std::stoi(item.substr(0, star));
    const int count = std::stoi(item.substr(star + 1));
    out.insert(out.end(), count, tag);
  }
  return out;
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("program_memory_overlay");
  test_utils::add_default_options(options);
  options.add_options()(
      "tags", "expected overlay tag per phase, comma separated",
      cxxopts::value<std::string>())("n-elems", "elements per tile",
                                     cxxopts::value<int>()->default_value("256"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  const std::vector<int32_t> tags = parse_tags(vm["tags"].as<std::string>());
  const int n = vm["n-elems"].as<int>();
  const int phases = static_cast<int>(tags.size());
  if (phases == 0) {
    std::cout << "--tags is required\n";
    return 1;
  }

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());

  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, vm["kernel"].as<std::string>());

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in = xrt::bo(device, n * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(3));
  auto bo_out = xrt::bo(device, phases * n * sizeof(int32_t),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  memcpy(bo_instr.map<void *>(), instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  int32_t *bufIn = bo_in.map<int32_t *>();
  for (int i = 0; i < n; i++)
    bufIn[i] = i * 7 + 1; // distinct per element, so a shifted read shows up
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Poison the output too. A phase that never writes then shows as poison
  // rather than as a plausible zero row.
  int32_t *bufOut = bo_out.map<int32_t *>();
  for (int i = 0; i < phases * n; i++)
    bufOut[i] = INT32_MIN;
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = kernel(3, bo_instr, instr_v.size(), bo_in, bo_out);
  if (run.wait() != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete\n";
    return 1;
  }
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  int bad_phases = 0;
  for (int p = 0; p < phases; p++) {
    // Every element carries the tag, so recovering it from element 0 and then
    // checking the rest separates "ran the wrong overlay" from "ran the right
    // one incorrectly".
    const int32_t want = tags[p];
    const int32_t saw = bufOut[p * n] - bufIn[0];

    int mismatches = 0, first = -1;
    for (int i = 0; i < n; i++) {
      if (bufOut[p * n + i] == bufIn[i] + want)
        continue;
      if (first < 0)
        first = i;
      mismatches++;
    }
    if (mismatches == 0) {
      std::cout << "phase " << p << ": overlay " << want << " ok\n";
      continue;
    }
    bad_phases++;
    if (bufOut[p * n] == INT32_MIN) {
      std::cout << "phase " << p << ": nothing was written -- the core did not "
                << "run this phase\n";
    } else if (saw == POISON_TAG) {
      std::cout << "phase " << p << ": ran the poison fill, so no overlay was "
                << "written into the slot\n";
    } else if (mismatches == n) {
      std::cout << "phase " << p << ": expected overlay " << want << ", ran "
                << saw << "\n";
    } else {
      std::cout << "phase " << p << ": overlay " << want << " ran but "
                << mismatches << " of " << n << " elements are wrong, first at "
                << first << "\n";
    }
  }

  if (bad_phases) {
    std::cout << "\n" << bad_phases << " phase(s) wrong\nfailed.\n";
    return 1;
  }
  std::cout << "\nPASS!\n";
  return 0;
}
