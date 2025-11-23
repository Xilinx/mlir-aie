#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdfloat>
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
constexpr int NUM_TEST_CASES = 2;

using dtype = std::bfloat16_t;

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

std::vector<dtype> load_bfloat16(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return {};
  }
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint16_t> raw_data(size / sizeof(uint16_t));
  file.read(reinterpret_cast<char *>(raw_data.data()), size);
  
  std::vector<dtype> data(raw_data.size());
  std::memcpy(data.data(), raw_data.data(), size);
  return data;
}

bool almost_equal(dtype a, dtype b, float rtol = 1e-1, float atol = 1e-2) {
  float fa = static_cast<float>(a);
  float fb = static_cast<float>(b);
  return std::abs(fa - fb) <= (atol + rtol * std::abs(fb));
}

struct Mismatch {
  size_t index;
  dtype got;
  dtype expected;
  float error;
};

bool verify_output(const dtype *output, const std::vector<dtype> &reference, 
                   size_t size, const std::string &test_name) {
  std::vector<Mismatch> mismatches;
  
  for (size_t i = 0; i < size && i < reference.size(); i++) {
    if (!almost_equal(output[i], reference[i])) {
      float error = std::abs(static_cast<float>(output[i]) - static_cast<float>(reference[i]));
      mismatches.push_back({i, output[i], reference[i], error});
    }
  }
  
  if (mismatches.empty()) {
    std::cerr << test_name << " PASSED: all outputs match reference" << std::endl;
    return true;
  } else {
    std::cerr << test_name << " FAILED: " << mismatches.size() << " mismatches" << std::endl;
    
    // Sort by error to find top 5 diverging values
    std::sort(mismatches.begin(), mismatches.end(), 
              [](const Mismatch &a, const Mismatch &b) { return a.error > b.error; });
    
    std::cerr << "Top 5 diverging values:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), mismatches.size()); i++) {
      const auto &m = mismatches[i];
      std::cerr << "  [" << m.index << "] got " << static_cast<float>(m.got)
                << ", expected " << static_cast<float>(m.expected)
                << ", error " << m.error << std::endl;
    }
    return false;
  }
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

  auto run = xrt::run(kernel);
  run.set_arg(0, bo_inout);

  // Run verification on multiple test cases
  bool all_tests_passed = true;
  
  for (int test_case = 0; test_case < NUM_TEST_CASES; test_case++) {
    std::cerr << "\n=== Test Case " << test_case << " ===" << std::endl;
    
    // Load inputs for this test case
    auto input_data = load_bfloat16("../../input_" + std::to_string(test_case) + ".bin");
    auto weights_1_data = load_bfloat16("../../weights_1_" + std::to_string(test_case) + ".bin");
    auto weights_2_data = load_bfloat16("../../weights_2_" + std::to_string(test_case) + ".bin");
    auto weights_3_data = load_bfloat16("../../weights_3_" + std::to_string(test_case) + ".bin");
    auto reference = load_bfloat16("../../reference_output_" + std::to_string(test_case) + ".bin");
    
    if (input_data.empty() || weights_1_data.empty() || weights_2_data.empty() || 
        weights_3_data.empty() || reference.empty()) {
      std::cerr << "Warning: test case " << test_case << " data not found, skipping" << std::endl;
      continue;
    }
    
    // Copy input data to buffers
    std::memcpy(buffer_ptrs["input"], input_data.data(), 
                std::min(input_data.size(), buffers_config[0].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_1"], weights_1_data.data(), 
                std::min(weights_1_data.size(), buffers_config[1].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_2"], weights_2_data.data(), 
                std::min(weights_2_data.size(), buffers_config[2].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_3"], weights_3_data.data(), 
                std::min(weights_3_data.size(), buffers_config[3].size) * sizeof(dtype));
    
    bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Run kernel
    run.start();
    run.wait2();
    
    bo_inout.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    // Load reference data for intermediates
    auto reference_left = load_bfloat16("../../reference_left_" + std::to_string(test_case) + ".bin");
    auto reference_right = load_bfloat16("../../reference_right_" + std::to_string(test_case) + ".bin");
    auto reference_left_swished = load_bfloat16("../../reference_left_swished_" + std::to_string(test_case) + ".bin");
    auto reference_intermediate = load_bfloat16("../../reference_intermediate_" + std::to_string(test_case) + ".bin");
    
    // Verify all intermediate buffers
    bool passed = true;
    
    if (!reference_left.empty()) {
      dtype *left_ptr = buffer_ptrs["left"];
      passed = verify_output(left_ptr, reference_left, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - left") && passed;
    }
    
    if (!reference_right.empty()) {
      dtype *right_ptr = buffer_ptrs["right"];
      passed = verify_output(right_ptr, reference_right, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - right") && passed;
    }
    
    if (!reference_left_swished.empty()) {
      dtype *left_swished_ptr = buffer_ptrs["left_swished"];
      passed = verify_output(left_swished_ptr, reference_left_swished, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - left_swished") && passed;
    }
    
    if (!reference_intermediate.empty()) {
      dtype *intermediate_ptr = buffer_ptrs["intermediate"];
      passed = verify_output(intermediate_ptr, reference_intermediate, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - intermediate") && passed;
    }
    
    if (!reference.empty()) {
      dtype *output_verify = buffer_ptrs["output"];
      passed = verify_output(output_verify, reference, EMBEDDING_DIM, 
                             "Test case " + std::to_string(test_case) + " - output") && passed;
    }
    
    all_tests_passed = all_tests_passed && passed;
  }
  
  if (!all_tests_passed) {
    std::cerr << "\nSome verification tests failed!" << std::endl;
  } else {
    std::cerr << "\nAll verification tests passed!" << std::endl;
  }

  // For benchmarking, use the first test case
  auto input_data = load_bfloat16("../../input_0.bin");
  auto weights_1_data = load_bfloat16("../../weights_1_0.bin");
  auto weights_2_data = load_bfloat16("../../weights_2_0.bin");
  auto weights_3_data = load_bfloat16("../../weights_3_0.bin");
  
  if (!input_data.empty() && !weights_1_data.empty() && 
      !weights_2_data.empty() && !weights_3_data.empty()) {
    std::memcpy(buffer_ptrs["input"], input_data.data(), 
                std::min(input_data.size(), buffers_config[0].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_1"], weights_1_data.data(), 
                std::min(weights_1_data.size(), buffers_config[1].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_2"], weights_2_data.data(), 
                std::min(weights_2_data.size(), buffers_config[2].size) * sizeof(dtype));
    std::memcpy(buffer_ptrs["weights_3"], weights_3_data.data(), 
                std::min(weights_3_data.size(), buffers_config[3].size) * sizeof(dtype));
    bo_inout.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
