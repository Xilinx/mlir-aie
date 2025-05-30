#include "cxxopts.hpp"
#include <bits/stdc++.h>
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

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED

using INPUT_DATATYPE = std::bfloat16_t;
using OUTPUT_DATATYPE = std::bfloat16_t;
#endif

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(int argc, const char *argv[]) {

  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("SAXPY Test");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  test_utils::parse_options(argc, argv, options, vm);
  int verbosity = vm["verbosity"].as<int>();
  int do_verify = vm["verify"].as<bool>();
  int n_iterations = vm["iters"].as<int>();
  int n_warmup_iterations = vm["warmup"].as<int>();

  int INPUT_VOLUME = 4096;
  int OUTPUT_VOLUME = 4096;

  size_t INPUT_SIZE = INPUT_VOLUME * sizeof(INPUT_DATATYPE);
  size_t OUTPUT_SIZE = OUTPUT_VOLUME * sizeof(OUTPUT_DATATYPE);

  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // ------------------------------------------------------
  // Get device, load the xclbin & kernel and register them
  // ------------------------------------------------------
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // ------------------------------------------------------
  // Initialize input/ output buffer sizes and sync them
  // ------------------------------------------------------
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_input_x =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_input_y =
      xrt::bo(device, INPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_output =
      xrt::bo(device, OUTPUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
  auto bo_dummy0 =
      xrt::bo(device, 4, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
  auto bo_trace =
      xrt::bo(device, 16384, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Initialize instruction buffer
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize input buffers
  INPUT_DATATYPE *bufInputX = bo_input_x.map<INPUT_DATATYPE *>();
  INPUT_DATATYPE *bufInputY = bo_input_y.map<INPUT_DATATYPE *>();
  for (int i = 0; i < INPUT_VOLUME; i++) {
    bufInputX[i] = test_utils::random_bfloat16_t(1.0f, 0.0f);
    bufInputY[i] = test_utils::random_bfloat16_t(1.0f, 0.0f);
  }

  // Initialize trace buffer
  void *bufTrace = bo_trace.map<void *>();
  memset(bufTrace, 0, 16384);

  // Sync buffers to update input buffer values
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_input_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_dummy0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = n_iterations + n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (verbosity >= 1) {
      std::cout << "Running Kernel.\n";
    }

    // Run kernel
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_input_x, bo_input_y, bo_output, bo_dummy0, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    OUTPUT_DATATYPE *bufOutput = bo_output.map<OUTPUT_DATATYPE *>();

    if (iter < n_warmup_iterations) {
      /* Warmup iterations do not count towards average runtime. */
      continue;
    }

    // Copy output results and verify they are correct
    if (do_verify) {
      if (verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();
      for (int i = 0; i < OUTPUT_VOLUME; i++) {
        float expected = 3.141f * static_cast<float>(bufInputX[i]) + static_cast<float>(bufInputY[i]);
        if (!test_utils::nearly_equal(static_cast<float>(bufOutput[i]), expected, 0.008f)) {
          errors++;
          std::cout << "Mismatch at index " << i << ": expected " << expected << ", got " << bufOutput[i] << std::endl;
        }
      }
      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (verbosity >= 1) {
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
      }
    } else {
      if (verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0
    test_utils::write_out_trace(((char *)bufTrace), 16384, "trace.txt");

    // Accumulate run times
    float npu_time =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
            .count();

    npu_time_total += npu_time;
    npu_time_min = (npu_time < npu_time_min) ? npu_time : npu_time_min;
    npu_time_max = (npu_time > npu_time_max) ? npu_time : npu_time_max;
  }

  // ------------------------------------------------------
  // Print verification and timing results
  // ------------------------------------------------------

  // TODO - Mac count to guide gflops
  float macs = 0;

  std::cout << std::endl
            << "Avg NPU time: " << npu_time_total / n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / n_iterations) << std::endl;

  std::cout << std::endl
            << "Min NPU time: " << npu_time_min << "us." << std::endl;
  if (macs > 0)
    std::cout << "Max NPU gflops: " << macs / (1000 * npu_time_min)
              << std::endl;

  std::cout << std::endl
            << "Max NPU time: " << npu_time_max << "us." << std::endl;
  if (macs > 0)
    std::cout << "Min NPU gflops: " << macs / (1000 * npu_time_max)
              << std::endl;

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  }
}
