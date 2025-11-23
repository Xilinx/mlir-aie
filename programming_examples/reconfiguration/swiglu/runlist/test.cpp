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

#include "xrt/experimental/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int EMBEDDING_DIM = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int NUM_ITERATIONS = 1000;
constexpr int NUM_TEST_CASES = 2;

using dtype = std::bfloat16_t;

struct KernelInfo {
  std::string name;
  std::string instr_file;
};

struct BufferInfo {
  std::string name;
  size_t size;
};

struct Operation {
  int kernel_idx;
  std::vector<std::string> buffer_names;
};

std::string combined_xclbin = "swiglu_combined.xclbin";

std::vector<KernelInfo> kernels_config = {
    {"swiglu_gemv_1", "swiglu_gemv_1.bin"},
    {"swiglu_silu", "swiglu_silu.bin"},
    {"swiglu_eltwise_mul", "swiglu_eltwise_mul.bin"},
    {"swiglu_gemv_2", "swiglu_gemv_2.bin"},
};

std::vector<BufferInfo> buffers_config = {
    {"input", EMBEDDING_DIM * sizeof(dtype)},
    {"weights_1", EMBEDDING_DIM * HIDDEN_DIM * sizeof(dtype)},
    {"weights_2", EMBEDDING_DIM * HIDDEN_DIM * sizeof(dtype)},
    {"weights_3", HIDDEN_DIM * EMBEDDING_DIM * sizeof(dtype)},
    {"left", HIDDEN_DIM * sizeof(dtype)},
    {"left_swished", HIDDEN_DIM * sizeof(dtype)},
    {"right", HIDDEN_DIM * sizeof(dtype)},
    {"intermediate", HIDDEN_DIM * sizeof(dtype)},
    {"output", EMBEDDING_DIM * sizeof(dtype)},
};

std::vector<Operation> runlist_config = {
    {0, {"weights_1", "input", "left"}},
    {0, {"weights_2", "input", "right"}},
    {1, {"left", "left_swished"}},
    {2, {"left_swished", "right", "intermediate"}},
    {3, {"weights_3", "intermediate", "output"}},
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

bool almost_equal(dtype a, dtype b, float rtol = 1e-2, float atol = 1e-3) {
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

std::vector<uint32_t> load_instr_binary(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint32_t> data(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

int main(int argc, const char *argv[]) {
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  auto xclbin = xrt::xclbin(combined_xclbin);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  std::vector<xrt::kernel> kernels;
  for (const auto &kinfo : kernels_config) {
    kernels.push_back(xrt::kernel(context, kinfo.name));
  }

  std::map<std::string, xrt::bo> buffers;
  for (const auto &binfo : buffers_config) {
    int group_id = (binfo.name == "weights_1" || binfo.name == "weights_2" || binfo.name == "input") ? 3 : 4;
    int kernel_idx = (binfo.name == "weights_3") ? 3 : 
                     (binfo.name == "left_swished") ? 1 :
                     (binfo.name == "intermediate") ? 2 : 0;
    buffers[binfo.name] = xrt::bo(device, binfo.size, XRT_BO_FLAGS_HOST_ONLY,
                                  kernels[kernel_idx].group_id(group_id));
  }

  std::vector<xrt::bo> bo_instrs;
  std::vector<std::vector<uint32_t>> instr_data;
  for (size_t i = 0; i < kernels_config.size(); i++) {
    instr_data.push_back(load_instr_binary(kernels_config[i].instr_file));
    bo_instrs.push_back(xrt::bo(device, instr_data[i].size() * sizeof(uint32_t),
                                XCL_BO_FLAGS_CACHEABLE, kernels[i].group_id(1)));
    void *buf = bo_instrs[i].map<void *>();
    memcpy(buf, instr_data[i].data(), instr_data[i].size() * sizeof(uint32_t));
    bo_instrs[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  unsigned int opcode = 3;

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
    dtype *input_ptr = buffers["input"].map<dtype *>();
    dtype *weights_1_ptr = buffers["weights_1"].map<dtype *>();
    dtype *weights_2_ptr = buffers["weights_2"].map<dtype *>();
    dtype *weights_3_ptr = buffers["weights_3"].map<dtype *>();
    
    std::memcpy(input_ptr, input_data.data(), 
                std::min(input_data.size(), size_t(EMBEDDING_DIM)) * sizeof(dtype));
    std::memcpy(weights_1_ptr, weights_1_data.data(), 
                std::min(weights_1_data.size(), size_t(EMBEDDING_DIM * HIDDEN_DIM)) * sizeof(dtype));
    std::memcpy(weights_2_ptr, weights_2_data.data(), 
                std::min(weights_2_data.size(), size_t(EMBEDDING_DIM * HIDDEN_DIM)) * sizeof(dtype));
    std::memcpy(weights_3_ptr, weights_3_data.data(), 
                std::min(weights_3_data.size(), size_t(HIDDEN_DIM * EMBEDDING_DIM)) * sizeof(dtype));
    
    // Zero-initialize output buffer
    dtype *output_ptr = buffers["output"].map<dtype *>();
    std::memset(output_ptr, 0, EMBEDDING_DIM * sizeof(dtype));
    
    buffers["input"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_1"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_2"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_3"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["left"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["left_swished"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["right"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["intermediate"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["output"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    // Run kernel
    xrt::runlist runlist_verify = xrt::runlist(context);
    for (const auto &op : runlist_config) {
      int kidx = op.kernel_idx;
      if (op.buffer_names.size() == 3) {
        auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                 buffers[op.buffer_names[0]],
                                 buffers[op.buffer_names[1]],
                                 buffers[op.buffer_names[2]]);
        runlist_verify.add(run);
      } else if (op.buffer_names.size() == 2) {
        auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                 buffers[op.buffer_names[0]],
                                 buffers[op.buffer_names[1]]);
        runlist_verify.add(run);
      }
    }
    runlist_verify.execute();
    runlist_verify.wait();
    
    // Sync all buffers from device to verify intermediates
    buffers["left"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffers["right"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffers["left_swished"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffers["intermediate"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffers["output"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    // Load reference data for intermediates
    auto reference_left = load_bfloat16("../../reference_left_" + std::to_string(test_case) + ".bin");
    auto reference_right = load_bfloat16("../../reference_right_" + std::to_string(test_case) + ".bin");
    auto reference_left_swished = load_bfloat16("../../reference_left_swished_" + std::to_string(test_case) + ".bin");
    auto reference_intermediate = load_bfloat16("../../reference_intermediate_" + std::to_string(test_case) + ".bin");
    
    // Verify all intermediate buffers
    bool passed = true;
    
    if (!reference_left.empty()) {
      dtype *left_ptr = buffers["left"].map<dtype *>();
      passed = verify_output(left_ptr, reference_left, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - left") && passed;
    }
    
    if (!reference_right.empty()) {
      dtype *right_ptr = buffers["right"].map<dtype *>();
      passed = verify_output(right_ptr, reference_right, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - right") && passed;
    }
    
    if (!reference_left_swished.empty()) {
      dtype *left_swished_ptr = buffers["left_swished"].map<dtype *>();
      passed = verify_output(left_swished_ptr, reference_left_swished, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - left_swished") && passed;
    }
    
    if (!reference_intermediate.empty()) {
      dtype *intermediate_ptr = buffers["intermediate"].map<dtype *>();
      passed = verify_output(intermediate_ptr, reference_intermediate, HIDDEN_DIM, 
                             "Test case " + std::to_string(test_case) + " - intermediate") && passed;
    }
    
    if (!reference.empty()) {
      dtype *output_verify = buffers["output"].map<dtype *>();
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
    dtype *input_ptr = buffers["input"].map<dtype *>();
    dtype *weights_1_ptr = buffers["weights_1"].map<dtype *>();
    dtype *weights_2_ptr = buffers["weights_2"].map<dtype *>();
    dtype *weights_3_ptr = buffers["weights_3"].map<dtype *>();
    
    std::memcpy(input_ptr, input_data.data(), 
                std::min(input_data.size(), size_t(EMBEDDING_DIM)) * sizeof(dtype));
    std::memcpy(weights_1_ptr, weights_1_data.data(), 
                std::min(weights_1_data.size(), size_t(EMBEDDING_DIM * HIDDEN_DIM)) * sizeof(dtype));
    std::memcpy(weights_2_ptr, weights_2_data.data(), 
                std::min(weights_2_data.size(), size_t(EMBEDDING_DIM * HIDDEN_DIM)) * sizeof(dtype));
    std::memcpy(weights_3_ptr, weights_3_data.data(), 
                std::min(weights_3_data.size(), size_t(HIDDEN_DIM * EMBEDDING_DIM)) * sizeof(dtype));
    
    // Zero-initialize output buffer
    dtype *output_ptr = buffers["output"].map<dtype *>();
    std::memset(output_ptr, 0, EMBEDDING_DIM * sizeof(dtype));
    
    buffers["input"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_1"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_2"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["weights_3"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["left"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["left_swished"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["right"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["intermediate"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffers["output"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  xrt::runlist runlist = xrt::runlist(context);
  for (const auto &op : runlist_config) {
    int kidx = op.kernel_idx;
    if (op.buffer_names.size() == 3) {
      auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                buffers[op.buffer_names[0]],
                                buffers[op.buffer_names[1]],
                                buffers[op.buffer_names[2]]);
      runlist.add(run);
    } else if (op.buffer_names.size() == 2) {
      auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                buffers[op.buffer_names[0]],
                                buffers[op.buffer_names[1]]);
      runlist.add(run);
    }
  }

  std::cerr << "Running timed iterations..." << std::endl;
  std::vector<double> iteration_times;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    auto iter_start = std::chrono::high_resolution_clock::now();
    runlist.execute();
    runlist.wait();
    auto iter_end = std::chrono::high_resolution_clock::now();
    double iter_time = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();
    iteration_times.push_back(iter_time);
  }

  buffers["output"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Output CSV format: variant,iteration,time_us
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    std::cout << "runlist," << i << "," << iteration_times[i] << std::endl;
  }

  return 0;
}
