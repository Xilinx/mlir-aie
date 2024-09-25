
#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "test_utils.h"

namespace po = boost::program_options;
using TY_2 = std::uint32_t;
const int scaleFactor = 3;

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  po::options_description desc("Allowed options");
  po::variables_map vm;
  desc.add_options()("help,h", "produce help message")(
    "xclbin", po::value<std::string>()->required(),
    "the input xclbin path2")(
    "kernel,k", po::value<std::string>()->required(),
    "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
    "verbosity,v", po::value<int>()->default_value(0),
    "the verbosity of the output")(
    "instr", po::value<std::string>()->required(),
    "path of file containing userspace instructions sent to the NPU")(
    "instr_2", po::value<std::string>()->required(),
    "path of file containing userspace instructions sent to the NPU")(
    "verify", po::value<bool>()->default_value(true),
    "whether to verify the AIE computed output")(
    "iters", po::value<int>()->default_value(1))(
    "warmup", po::value<int>()->default_value(0))(
    "trace_sz,t", po::value<int>()->default_value(0))(
    "trace_file", po::value<std::string>()->default_value("trace.txt"),
    "where to store trace output");

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();

  constexpr int IN_SIZE_1 = DATASIZE;
  constexpr int IN_VOLUME_2 = DATASIZE;

  constexpr int OUT_SIZE_1 = DATASIZE;
  constexpr int OUT_VOLUME_2 = DATASIZE;
  int IN_SIZE_2 = IN_VOLUME_2 * sizeof(TY_2);
  int OUT_SIZE_2 = OUT_VOLUME_2 * sizeof(TY_2) + trace_size;

  unsigned int device_index = 0;
  auto device = xrt::device(device_index);
  std::vector<uint32_t> instr_v1 = test_utils::load_instr_sequence(vm["instr"].as<std::string>());
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>()); // Load the xclbin
  device.register_xclbin(xclbin); // Register xclbin
  xrt::hw_context context_1(device, xclbin.get_uuid()); // Get a hardware context
  auto kernel_1 = xrt::kernel(context_1, "VVM");

  std::vector<uint32_t> instr_v2 = test_utils::load_instr_sequence(vm["instr_2"].as<std::string>());
  auto kernel_2 = xrt::kernel(context_1, "VSM"); // Get a kernel_2 handle: MLIR_AIE


  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr_1 = xrt::bo(device, instr_v1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_1.group_id(1));
  void *bufInstr_1 = bo_instr_1.map<void *>();
  memcpy(bufInstr_1, instr_v1.data(), instr_v1.size() * sizeof(int));

  auto bo_inA_1   = xrt::bo(device, IN_SIZE_1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(3));
  uint32_t *bufInA_1 = bo_inA_1.map<uint32_t *>();
  std::vector<uint32_t> srcVecA_1;
  for (int i = 0; i < IN_SIZE_1; i++)
    srcVecA_1.push_back(i + 1);
  memcpy(bufInA_1, srcVecA_1.data(), (srcVecA_1.size() * sizeof(uint32_t)));

  auto bo_inB_1   = xrt::bo(device, IN_SIZE_1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(4));
  uint32_t *bufInB_1 = bo_inB_1.map<uint32_t *>();
  std::vector<uint32_t> srcVecB_1;
  for (int i = 0; i < IN_SIZE_1; i++)
    srcVecB_1.push_back(i);
  memcpy(bufInB_1, srcVecB_1.data(), (srcVecB_1.size() * sizeof(uint32_t)));

  auto bo_out_1   = xrt::bo(device, OUT_SIZE_1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(5));


  auto bo_instr_2  = xrt::bo(device, instr_v2.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_2.group_id(1));
  void *bufInstr_2 = bo_instr_2.map<void *>();
  memcpy(bufInstr_2, instr_v2.data(), instr_v2.size() * sizeof(int));

  auto bo_inA_2  = xrt::bo(device, IN_SIZE_2, XRT_BO_FLAGS_HOST_ONLY, kernel_2.group_id(3));
  TY_2 *bufInA_2 = bo_inA_2.map<TY_2 *>();
  for (int i = 0; i < IN_VOLUME_2; i++)
    bufInA_2[i] = i + 1;

  auto bo_inFactor = xrt::bo(device, 1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_2.group_id(4));
  int32_t *bufInFactor = bo_inFactor.map<int32_t *>();
  *bufInFactor = (TY_2)scaleFactor;

  auto bo_outC_2 = xrt::bo(device, OUT_SIZE_2, XRT_BO_FLAGS_HOST_ONLY, kernel_2.group_id(5));


  bo_instr_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  bo_instr_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inFactor.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // RUN KERNEL

  #define SIZE 4096

  unsigned int opcode = 3;

  std::ofstream f_time;
  std::string file_name = "time_" + std::to_string(DATASIZE) + ".txt";
  f_time.open(file_name);
  for (int i=1; i<=1000; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    auto run1 = kernel_1(opcode, bo_instr_1, instr_v1.size(), bo_inA_1, bo_inB_1, bo_out_1);
    run1.wait();
    auto run2 = kernel_2(opcode, bo_instr_2, instr_v2.size(), bo_inA_2, bo_inFactor, bo_outC_2);
    run2.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    if (i<11)
      std::cout << i << " " << IN_SIZE_1 << " NPU time: " << npu_time << "us." << std::endl;
    f_time << npu_time << "\n";
  }
  f_time.close();

  bo_out_1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  bo_outC_2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint32_t *bufOut_1 = bo_out_1.map<uint32_t *>();
  TY_2 *bufOut_2 = bo_outC_2.map<TY_2 *>();


  // COMPARE
  int errors = 0;

  for (uint32_t i = 0; i < SIZE; i++) {
    if (*(bufOut_1 + i) != *(bufInA_1 + i) * *(bufInB_1 + i)) {
      std::cout << "Error in output " << *(bufOut_1 + i)
                << " != " << *(bufInA_1 + i) << " * " << *(bufInB_1 + i)
                << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << *(bufOut_1 + i)
                  << " == " << *(bufInA_1 + i) * *(bufInB_1 + i) << std::endl;
    }
  }

  for (uint32_t i = 0; i < SIZE; i++) {
    int32_t ref = bufInA_2[i] * scaleFactor;
    int32_t test = bufOut_2[i];
    if (test != ref) {
      if (verbosity >= 1)
        std::cout << "Error in output " << test << " != " << ref << std::endl;
      errors++;
    } else {
      if (verbosity >= 1)
        std::cout << "Correct output " << test << " == " << ref << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed with errors:" << errors << ".\n\n";
    return 1;
  }
  return 0;
}
