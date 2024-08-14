
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

int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  po::options_description desc("Allowed options");
  po::variables_map vm;
  desc.add_options()("help,h", "produce help message")(
    "xclbin_1", po::value<std::string>()->required(),
    "the input xclbin path")(
    "xclbin_2", po::value<std::string>()->required(),
    "the input xclbin path")(
    "kernel,k", po::value<std::string>()->required(),
    "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
    "verbosity,v", po::value<int>()->default_value(0),
    "the verbosity of the output")(
    "instr_1", po::value<std::string>()->required(),
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

  constexpr int IN_SIZE = 256*60;
  constexpr int OUT_SIZE = IN_SIZE;

  std::vector<uint32_t> instr_v1 = test_utils::load_instr_sequence(vm["instr_1"].as<std::string>());

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  // Get a device handle
  unsigned int device_index = 0;
  auto kernelName = "MLIR_AIE";

  auto device_1 = xrt::device(device_index);
  auto xclbin_1 = xrt::xclbin(vm["xclbin_1"].as<std::string>()); // Load the xclbin
  device_1.register_xclbin(xclbin_1); // Register xclbin
  xrt::hw_context context_1(device_1, xclbin_1.get_uuid()); // Get a hardware context
  auto kernel_1 = xrt::kernel(context_1, kernelName); // Get a kernel handle: MLIR_AIE


  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------

  auto bo_instr_1 = xrt::bo(device_1, instr_v1.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel_1.group_id(1));
  auto bo_inA   = xrt::bo(device_1, IN_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(3));
  auto bo_inB   = xrt::bo(device_1, IN_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(4));
  auto bo_out   = xrt::bo(device_1, OUT_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel_1.group_id(5));

  uint32_t *bufInA = bo_inA.map<uint32_t *>();
  std::vector<uint32_t> srcVecA;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecA.push_back(i + 1);
  memcpy(bufInA, srcVecA.data(), (srcVecA.size() * sizeof(uint32_t)));

  uint32_t *bufInB = bo_inB.map<uint32_t *>();
  std::vector<uint32_t> srcVecB;
  for (int i = 0; i < IN_SIZE; i++)
    srcVecB.push_back(i);
  memcpy(bufInB, srcVecB.data(), (srcVecB.size() * sizeof(uint32_t)));

  void *bufInstr_1 = bo_instr_1.map<void *>();
  memcpy(bufInstr_1, instr_v1.data(), instr_v1.size() * sizeof(int));

  bo_instr_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // RUN KERNEL

  unsigned int opcode = 3;

  std::ofstream f_time;
  f_time.open("time.txt");
  for (int i=1; i<=1000; i++) {
    auto start = std::chrono::high_resolution_clock::now();

    auto run = kernel_1(opcode, bo_instr_1, instr_v1.size(), bo_inA, bo_inB, bo_out);
    run.wait();

    auto stop = std::chrono::high_resolution_clock::now();
    float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    if (i<11)
      std::cout << i << " " << IN_SIZE << " NPU time: " << npu_time << "us." << std::endl;
    f_time << npu_time << "\n";
  }
  f_time.close();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint32_t *bufOut = bo_out.map<uint32_t *>();

  // COMPARE

  int errors = 0;

  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    if (*(bufOut + i) != *(bufInA + i) * *(bufInB + i)) {
      std::cout << "Error in output " << *(bufOut + i)
                << " != " << *(bufInA + i) << " * " << *(bufInB + i)
                << std::endl;
      errors++;
    } else {
      if (verbosity > 1)
        std::cout << "Correct output " << *(bufOut + i)
                  << " == " << *(bufInA + i) * *(bufInB + i) << std::endl;
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
