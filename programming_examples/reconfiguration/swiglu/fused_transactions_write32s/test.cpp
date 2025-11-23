#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

constexpr int EMBEDDING_DIM = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int NUM_ITERATIONS = 1000;

using dtype = uint16_t;

struct BufferInfo {
  std::string name;
  size_t size;
};

std::vector<BufferInfo> buffers_config = {
    {"input", EMBEDDING_DIM},
    {"weights_1", EMBEDDING_DIM * HIDDEN_DIM},
    {"weights_2", EMBEDDING_DIM * HIDDEN_DIM},
    {"weights_3", HIDDEN_DIM * EMBEDDING_DIM},
    {"left", HIDDEN_DIM},
    {"left_swished", HIDDEN_DIM},
    {"right", HIDDEN_DIM},
    {"intermediate", HIDDEN_DIM},
    {"output", EMBEDDING_DIM},
};

std::vector<uint16_t> load_reference(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return {};
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint16_t> data(size / sizeof(uint16_t));
  file.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

int main(int argc, const char *argv[]) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  std::string kernelName = "main:sequence";
  xrt::elf ctx_elf{"aie.elf"};
  xrt::hw_context context = xrt::hw_context(device, ctx_elf);
  auto kernel = xrt::ext::kernel(context, kernelName);

  size_t total_size = 0;
  for (const auto &binfo : buffers_config) {
    total_size += binfo.size;
  }

  xrt::bo bo_inout = xrt::ext::bo{device, total_size * sizeof(dtype)};
  dtype *buf = bo_inout.map<dtype *>();

  std::map<std::string, dtype *> buffer_ptrs;
  size_t offset = 0;
  for (const auto &binfo : buffers_config) {
    buffer_ptrs[binfo.name] = buf + offset;
    offset += binfo.size;
  }

  for (size_t i = 0; i < buffers_config[0].size; i++) { // input
    buffer_ptrs["input"][i] = 0x3F80;
  }
  for (size_t i = 0; i < buffers_config[1].size; i++) { // weights_1
    buffer_ptrs["weights_1"][i] = 0x3F00;
  }
  for (size_t i = 0; i < buffers_config[2].size; i++) { // weights_2
    buffer_ptrs["weights_2"][i] = 0x3F00;
  }
  for (size_t i = 0; i < buffers_config[3].size; i++) { // weights_3
    buffer_ptrs["weights_3"][i] = 0x3F00;
  }

  bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_inout);

  std::cerr << "Running verification..." << std::endl;
  run.start();
  run.wait2();

  bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  dtype *output_verify = buffer_ptrs["output"];

  auto reference = load_reference("../../reference_output.bin");
  if (!reference.empty()) {
    int mismatches = 0;
    for (size_t i = 0; i < EMBEDDING_DIM && i < reference.size(); i++) {
      if (output_verify[i] != reference[i]) {
        mismatches++;
        if (mismatches <= 10) {
          std::cerr << "Mismatch at " << i << ": got 0x" << std::hex
                    << output_verify[i] << ", expected 0x" << reference[i]
                    << std::dec << std::endl;
        }
      }
    }
    if (mismatches == 0) {
      std::cerr << "Verification PASSED: all outputs match reference" << std::endl;
    } else {
      std::cerr << "Verification FAILED: " << mismatches << " mismatches"
                << std::endl;
    }
  } else {
    std::cerr << "Warning: reference output not found, skipping validation"
              << std::endl;
  }

  std::cerr << "Running timed iterations..." << std::endl;
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    run.start();
    run.wait2();
    auto iter_end = std::chrono::high_resolution_clock::now();
    double iter_time = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();
    iteration_times.push_back(iter_time);
  }

  bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Output CSV format: variant,iteration,time_us
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    std::cout << "fused_transactions_write32s," << i << "," << iteration_times[i] << std::endl;
  }

  return 0;
}
