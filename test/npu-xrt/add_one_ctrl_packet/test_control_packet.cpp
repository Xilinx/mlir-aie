// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

int main(int argc, char **argv) {
  const bool useXclbin = argc > 1;
  constexpr int count = 8;
  auto device = xrt::device(0);
  xrt::hw_context context;
  xrt::kernel kernel;
  xrt::bo instructions, packets;
  if (useXclbin) {
    xrt::xclbin image{std::string(argv[1])};
    device.register_xclbin(image);
    context = xrt::hw_context(device, image.get_uuid());
    kernel = xrt::kernel(context, "MLIR_AIE");
    auto loadBuffer = [&](const char *path, xrt::bo::flags flags, int group) {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file)
        throw std::runtime_error(std::string("Cannot open ") + path);
      auto size = static_cast<size_t>(file.tellg());
      xrt::bo buffer(device, size, flags, kernel.group_id(group));
      file.seekg(0);
      if (!file.read(buffer.map<char *>(), size))
        throw std::runtime_error(std::string("Cannot read ") + path);
      buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      return buffer;
    };
    instructions = loadBuffer("insts.bin", xrt::bo::flags::cacheable, 1);
    packets = loadBuffer("full_elf_main_sequence.ctrlpkt.bin",
                         xrt::bo::flags::host_only, 5);
  } else {
    xrt::elf elf{"aie.elf"};
    context = xrt::hw_context(device, elf);
    kernel = xrt::ext::kernel(context, "main:sequence");
  }
  xrt::bo readback = xrt::ext::bo{device, count * sizeof(int32_t)};
  xrt::bo output = xrt::ext::bo{device, count * sizeof(int32_t)};
  auto *controlData = readback.map<int32_t *>();
  auto *dmaData = output.map<int32_t *>();
  int errors = 0;
  for (int pass = 0; pass < 3; ++pass) {
    std::fill_n(controlData, count, -1);
    std::fill_n(dmaData, count, -1);
    readback.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrt::run run(kernel);
    if (useXclbin) {
      run.set_arg(0, 3);
      run.set_arg(1, instructions);
      run.set_arg(2, instructions.size() / sizeof(uint32_t));
      run.set_arg(5, packets);
    }
    run.set_arg(useXclbin ? 3 : 0, readback);
    run.set_arg(useXclbin ? 4 : 1, output);
    run.start();
    run.wait2();
    readback.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    for (int i = 0; i < count; ++i) {
      if (dmaData[i] != 7 + i || controlData[i] != 8 + i) {
        std::cerr << "pass " << pass << ", element " << i << ": DMA "
                  << dmaData[i] << " (expected " << 7 + i << "), control "
                  << controlData[i] << " (expected " << 8 + i << ")\n";
        ++errors;
      }
    }
  }
  std::cout << (errors ? "FAIL!" : "PASS!") << std::endl;
  return errors ? 1 : 0;
}
