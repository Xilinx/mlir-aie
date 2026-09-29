// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace {
constexpr int kLength = 256;
constexpr int kDefaultIterations = 1000;
constexpr unsigned int kDispatchTimeoutMs = 2000;

std::vector<uint32_t> loadInstructions(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("cannot open instruction file: " + path);
  const std::streamsize bytes = stream.tellg();
  if (bytes <= 0 || bytes % sizeof(uint32_t) != 0)
    throw std::runtime_error("invalid instruction file size");
  stream.seekg(0);
  std::vector<uint32_t> instructions(bytes / sizeof(uint32_t));
  if (!stream.read(reinterpret_cast<char *>(instructions.data()), bytes))
    throw std::runtime_error("cannot read instruction file");
  return instructions;
}
} // namespace

int main(int argc, char **argv) {
  const int iterations = argc > 1 ? std::stoi(argv[1]) : kDefaultIterations;
  if (iterations <= 0) {
    std::cerr << "iteration count must be positive\n";
    return 1;
  }

  const auto instructions = loadInstructions("insts.bin");
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(std::string("final.xclbin"));
  auto kernels = xclbin.get_kernels();
  const auto kernelIt =
      std::find_if(kernels.begin(), kernels.end(), [](auto &candidate) {
        return candidate.get_name().rfind("MLIR_AIE", 0) == 0;
      });
  if (kernelIt == kernels.end()) {
    std::cerr << "MLIR_AIE kernel not found\n";
    return 1;
  }

  device.register_xclbin(xclbin);
  auto context = xrt::hw_context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelIt->get_name());
  auto instructionBo = xrt::bo(device, instructions.size() * sizeof(uint32_t),
                               XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto inputBo = xrt::bo(device, kLength * sizeof(int32_t),
                         XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto outputBo = xrt::bo(device, kLength * sizeof(int32_t),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  std::memcpy(instructionBo.map<void *>(), instructions.data(),
              instructions.size() * sizeof(uint32_t));
  instructionBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto *input = inputBo.map<int32_t *>();
  auto *output = outputBo.map<int32_t *>();

  // Every dispatch runs on the same hardware context, with no reload between
  // them: what the re-arm has to carry is the resident fifo's lock and
  // channel-queue state.
  for (int dispatch = 0; dispatch < iterations; ++dispatch) {
    // The input varies per dispatch, so a fifo that stopped delivering shows up
    // as the previous dispatch's output. Do not clear the output BO: syncing it
    // to device races the output DMA.
    for (int i = 0; i < kLength; ++i)
      input[i] = dispatch * 1024 + i;
    inputBo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    auto run = kernel(3, instructionBo, instructions.size(), inputBo, outputBo);
    ert_cmd_state state;
    try {
      state = run.wait(kDispatchTimeoutMs);
    } catch (const std::exception &error) {
      std::cerr << "WAIT ERROR at dispatch " << dispatch << ": " << error.what()
                << "\n";
      return 1;
    }
    if (state == ERT_CMD_STATE_TIMEOUT) {
      run.abort();
      std::cerr << "TIMEOUT at dispatch " << dispatch << "\n";
      return 2;
    }
    if (state != ERT_CMD_STATE_COMPLETED) {
      std::cerr << "command state " << state << " at dispatch " << dispatch
                << "\n";
      return 1;
    }

    outputBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < kLength; ++i) {
      const int32_t expected = input[i] + i + 1;
      if (output[i] != expected) {
        std::cerr << "MISMATCH at dispatch " << dispatch << ", element " << i
                  << ": expected " << expected << ", got " << output[i] << "\n";
        return 1;
      }
    }
  }

  std::cout << "PASS: " << iterations
            << " exact dispatches on one hardware context\n";
  return 0;
}
