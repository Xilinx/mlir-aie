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

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int EMBEDDING_DIM = 2048;
constexpr int HIDDEN_DIM = 8192;
constexpr int NUM_ITERATIONS = 1000;

using dtype = uint16_t;

struct KernelInfo {
  std::string name;
  std::string xclbin_file;
  std::string instr_file;
  uint32_t kernel_id;
};

struct BufferInfo {
  std::string name;
  size_t size;
};

struct Operation {
  int kernel_idx;
  std::vector<std::string> buffer_names;
};

std::vector<KernelInfo> kernels_config = {
    {"swiglu_gemv_1", "swiglu_gemv_1.xclbin", "swiglu_gemv_1.bin", 0x901},
    {"swiglu_silu", "swiglu_silu.xclbin", "swiglu_silu.bin", 0x902},
    {"swiglu_eltwise_mul", "swiglu_eltwise_mul.xclbin", "swiglu_eltwise_mul.bin", 0x903},
    {"swiglu_gemv_2", "swiglu_gemv_2.xclbin", "swiglu_gemv_2.bin", 0x904},
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

  std::vector<xrt::xclbin> xclbins;
  std::vector<xrt::hw_context> contexts;
  std::vector<xrt::kernel> kernels;

  for (const auto &kernel_info : kernels_config) {
    xclbins.push_back(xrt::xclbin(kernel_info.xclbin_file));
    device.register_xclbin(xclbins.back());
    contexts.push_back(xrt::hw_context(device, xclbins.back().get_uuid()));
    kernels.push_back(xrt::kernel(contexts.back(), kernel_info.name));
  }

  std::map<std::string, xrt::bo> buffers;
  for (const auto &buf_info : buffers_config) {
    int kernel_idx = 0;
    int arg_idx = 3;
    for (const auto &op : runlist_config) {
      for (size_t i = 0; i < op.buffer_names.size(); i++) {
        if (op.buffer_names[i] == buf_info.name) {
          kernel_idx = op.kernel_idx;
          arg_idx = 3 + i;
          goto found;
        }
      }
    }
    found:
    buffers[buf_info.name] = xrt::bo(device, buf_info.size,
                                     XRT_BO_FLAGS_HOST_ONLY,
                                     kernels[kernel_idx].group_id(arg_idx));
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

  dtype *input_ptr = buffers["input"].map<dtype *>();
  dtype *weights_1_ptr = buffers["weights_1"].map<dtype *>();
  dtype *weights_2_ptr = buffers["weights_2"].map<dtype *>();
  dtype *weights_3_ptr = buffers["weights_3"].map<dtype *>();

  for (int i = 0; i < EMBEDDING_DIM; i++) {
    input_ptr[i] = 0x3F80;
  }
  for (int i = 0; i < EMBEDDING_DIM * HIDDEN_DIM; i++) {
    weights_1_ptr[i] = 0x3F00;
    weights_2_ptr[i] = 0x3F00;
  }
  for (int i = 0; i < HIDDEN_DIM * EMBEDDING_DIM; i++) {
    weights_3_ptr[i] = 0x3F00;
  }

  buffers["input"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  buffers["weights_1"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  buffers["weights_2"].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  buffers["weights_3"].sync(XCL_BO_SYNC_BO_TO_DEVICE);

  unsigned int opcode = 3;

  std::cerr << "Running verification..." << std::endl;
  for (const auto &op : runlist_config) {
    int kidx = op.kernel_idx;
    std::vector<xrt::bo*> op_buffers;
    for (const auto &bname : op.buffer_names) {
      op_buffers.push_back(&buffers[bname]);
    }
    
    if (op_buffers.size() == 3) {
      auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                               *op_buffers[0], *op_buffers[1], *op_buffers[2]);
      run.wait2();
    } else if (op_buffers.size() == 2) {
      auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                               *op_buffers[0], *op_buffers[1]);
      run.wait2();
    }
  }

  buffers["output"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  dtype *output_verify = buffers["output"].map<dtype *>();

  auto reference = load_reference("../../reference_output.bin");
  if (!reference.empty()) {
    int mismatches = 0;
    for (int i = 0; i < EMBEDDING_DIM && i < reference.size(); i++) {
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
    
    for (const auto &op : runlist_config) {
      int kidx = op.kernel_idx;
      std::vector<xrt::bo*> op_buffers;
      for (const auto &bname : op.buffer_names) {
        op_buffers.push_back(&buffers[bname]);
      }
      
      if (op_buffers.size() == 3) {
        auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                 *op_buffers[0], *op_buffers[1], *op_buffers[2]);
        run.wait2();
      } else if (op_buffers.size() == 2) {
        auto run = kernels[kidx](opcode, bo_instrs[kidx], instr_data[kidx].size(),
                                 *op_buffers[0], *op_buffers[1]);
        run.wait2();
      }
    }
    
    auto iter_end = std::chrono::high_resolution_clock::now();
    double iter_time = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start).count();
    iteration_times.push_back(iter_time);
  }

  buffers["output"].sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Output CSV format: variant,iteration,time_us
  for (int i = 0; i < NUM_ITERATIONS; i++) {
    std::cout << "separate_xclbins," << i << "," << iteration_times[i] << std::endl;
  }

  return 0;
}
