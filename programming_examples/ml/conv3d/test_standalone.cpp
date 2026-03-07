//===- test_standalone.cpp --------------------------------------*- C++ -*-===//
//
// Standalone test for conv3d using XRT directly (no dependencies)
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int DEPTH = 8;
constexpr int HEIGHT = 8;
constexpr int WIDTH = 8;
constexpr int IN_CHANNELS = 8;
constexpr int OUT_CHANNELS = 8;

std::vector<uint32_t> load_instr(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file) {
    std::cerr << "Unable to open " << filename << std::endl;
    exit(1);
  }

  auto size = file.tellg();
  file.seekg(0, std::ios::beg);

  auto num_instr = size / sizeof(int);
  std::vector<uint32_t> instr(num_instr);
  file.read(reinterpret_cast<char *>(instr.data()), size);

  return instr;
}

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0]
              << " <instr.bin> <xclbin> <kernel_name>" << std::endl;
    return 1;
  }

  std::cout << "=== Conv3D Test (XRT Direct) ===\n";
  std::cout << "Volume: " << DEPTH << "x" << HEIGHT << "x" << WIDTH << "\n";
  std::cout << "Channels: " << IN_CHANNELS << " -> " << OUT_CHANNELS << "\n\n";

  // Load instruction sequence
  auto instr_v = load_instr(argv[1]);
  std::cout << "Loaded " << instr_v.size() << " instructions\n";

  // Start XRT
  auto device = xrt::device(0);
  auto xclbin_uuid = device.load_xclbin(argv[2]);
  auto kernel = xrt::kernel(device, xclbin_uuid, argv[3]);

  std::cout << "XRT initialized, kernel loaded\n";

  // Calculate buffer sizes
  auto input_size = DEPTH * HEIGHT * WIDTH * IN_CHANNELS;
  auto weights_size = IN_CHANNELS * OUT_CHANNELS;
  auto output_size = DEPTH * HEIGHT * WIDTH * OUT_CHANNELS;

  std::cout << "Buffer sizes: input=" << input_size << " weights="
            << weights_size << " output=" << output_size << "\n";

  // Create buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_inA = xrt::bo(device, input_size, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_weights = xrt::bo(device, weights_size, XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(4));
  auto bo_out = xrt::bo(device, output_size, XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  // Write instruction stream
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize input
  uint8_t *bufInA = bo_inA.map<uint8_t *>();
  for (int i = 0; i < input_size; i++) {
    bufInA[i] = i % 256;
  }

  // Initialize weights (all ones)
  int8_t *bufWeights = bo_weights.map<int8_t *>();
  for (int i = 0; i < weights_size; i++) {
    bufWeights[i] = 1;
  }

  // Initialize output to zero
  uint8_t *bufOut = bo_out.map<uint8_t *>();
  memset(bufOut, 0, output_size);

  std::cout << "Data initialized\n";

  // Sync to device
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  std::cout << "Buffers synced to device\n";

  // Execute kernel
  std::cout << "Running kernel...\n";
  unsigned int opcode = 3;
  auto start = std::chrono::high_resolution_clock::now();
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_weights, bo_out);
  run.wait();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  // Sync from device
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << "\n✅ SUCCESS! Kernel completed in " << duration.count()
            << " μs\n";

  // Print some output
  std::cout << "First 16 output values: ";
  for (int i = 0; i < 16; i++) {
    std::cout << (int)bufOut[i] << " ";
  }
  std::cout << "\n";

  std::cout << "\nPASS!\n";
  return 0;
}
