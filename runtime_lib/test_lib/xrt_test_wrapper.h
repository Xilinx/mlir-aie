#include "cxxopts.hpp"
#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <fstream>
#include <iostream>
#include <sstream>

struct args {
  int verbosity;
  int do_verify;
  int n_iterations;
  int n_warmup_iterations;
  int trace_size;
  std::string instr;
  std::string xclbin;
  std::string kernel;
  std::string trace_file;
};

struct args parse_args(int argc, const char *argv[]) {
  // ------------------------------------------------------
  // Parse program arguments
  // ------------------------------------------------------
  cxxopts::Options options("XRT Test Wrapper");
  cxxopts::ParseResult vm;
  test_utils::add_default_options(options);

  struct args myargs;

  test_utils::parse_options(argc, argv, options, vm);
  myargs.verbosity = vm["verbosity"].as<int>();
  myargs.do_verify = vm["verify"].as<bool>();
  myargs.n_iterations = vm["iters"].as<int>();
  myargs.n_warmup_iterations = vm["warmup"].as<int>();
  myargs.trace_size = vm["trace_sz"].as<int>();
  myargs.instr = vm["instr"].as<std::string>();
  myargs.xclbin = vm["xclbin"].as<std::string>();
  myargs.kernel = vm["kernel"].as<std::string>();
  myargs.trace_file = vm["trace_file"].as<std::string>();

  return myargs;
}

uint32_t getParity(uint32_t n) {
  int count = 0;
  while (n > 0) {
    if (n & 1) { // Check if the least significant bit is 1
      count++;
    }
    n >>= 1; // Right shift to check the next bit
  }
  return (count % 2 == 0) ? 0 : 1; // 0 for even parity, 1 for odd parity
}

uint32_t create_ctrl_pkt(int operation, int beats, int addr,
                         int ctrl_pkt_read_id = 28) {
  uint32_t ctrl_pkt = ((ctrl_pkt_read_id & 0xFF) << 24) |
                      ((operation & 0x3) << 22) | ((beats & 0x3) << 20) |
                      (addr & 0x7FFFF);
  ctrl_pkt |= (0x1 ^ getParity(ctrl_pkt)) << 31;
  return ctrl_pkt;
}

/*
 ******************************************************************************
 * XRT based test wrapper for 2 inputs and 1 output
 ******************************************************************************
 */
template <typename T1, typename T2, typename T3, void (*init_bufIn1)(T1 *, int),
          void (*init_bufIn2)(T2 *, int), void (*init_bufOut)(T3 *, int),
          int (*verify_results)(T1 *, T2 *, T3 *, int, int)>
int setup_and_run_aie(int IN1_VOLUME, int IN2_VOLUME, int OUT_VOLUME,
                      struct args myargs, bool enable_ctrl_pkts = false) {

  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                                   myargs.xclbin, myargs.kernel);

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(T1), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_in2 = xrt::bo(device, IN2_VOLUME * sizeof(T2), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(T3), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));

  // If we enable control packets, then this is the input xrt buffer for that.
  // Otherwise, this is a dummy placedholder buffer.
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  // Second workaround for driver issue. Allocate large trace buffer *4
  // This includes the 8 bytes needed for control packet response.
  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer objects
  T1 *bufIn1 = bo_in1.map<T1 *>();
  T2 *bufIn2 = bo_in2.map<T2 *>();
  T3 *bufOut = bo_out.map<T3 *>();
  char *bufTrace = bo_trace.map<char *>();
  uint32_t *bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();

  init_bufIn1(bufIn1, IN1_VOLUME);
  init_bufIn2(bufIn2, IN2_VOLUME);
  init_bufOut(bufOut, OUT_VOLUME); // <<< what size do I pass it?

  // char *bufTmp1 = bo_tmp1.map<char *>();
  // memset(bufTmp1, 0, 4);

  if (myargs.trace_size > 0)
    memset(bufTrace, 0, myargs.trace_size);

  // Set control packet values
  if (myargs.trace_size > 0 && enable_ctrl_pkts) {
    bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
    bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
    if (myargs.verbosity >= 1) {
      std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
      std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
    }
  }

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // bo_tmp1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (myargs.trace_size > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (enable_ctrl_pkts)
      bo_ctrlpkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    // Run kernel
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_in2, bo_out,
                      bo_ctrlpkts, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      /* Warmup iterations do not count towards average runtime. */
      continue;

    // Copy output results and verify they are correct
    if (myargs.do_verify) {
      if (myargs.verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();

      errors +=
          verify_results(bufIn1, bufIn2, bufOut, IN1_VOLUME, myargs.verbosity);

      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (myargs.verbosity >= 1)
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
    } else {
      if (myargs.verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0 and first iteration
    if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations) {
      test_utils::write_out_trace(((char *)bufTrace), myargs.trace_size,
                                  myargs.trace_file);
    }

    // Write out control packet outputs
    if (enable_ctrl_pkts) {
      uint32_t *ctrl_pkt_out =
          (uint32_t *)(((char *)bufTrace) + myargs.trace_size);
      if (myargs.verbosity >= 1) {
        std::cout << "ctrl_pkt_out[0]:" << std::hex << ctrl_pkt_out[0]
                  << std::endl;
        std::cout << "ctrl_pkt_out[1]:" << std::hex << ctrl_pkt_out[1]
                  << std::endl;
      }
      int col = (ctrl_pkt_out[0] >> 21) & 0x7F;
      int row = (ctrl_pkt_out[0] >> 16) & 0x1F;
      if ((ctrl_pkt_out[1] >> 8) == 3)
        std::cout << "WARNING: Trace overflow detected in tile(" << row << ","
                  << col << ". Trace results may be invalid." << std::endl;
    }

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
            << "Avg NPU time: " << npu_time_total / myargs.n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / myargs.n_iterations)
              << std::endl;

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

/*
 ******************************************************************************
 * XRT based test wrapper for 1 input and 1 output
 ******************************************************************************
 */
template <typename T1, typename T3, void (*init_bufIn1)(T1 *, int),
          void (*init_bufOut)(T3 *, int),
          int (*verify_results)(T1 *, T3 *, int, int)>
int setup_and_run_aie(int IN1_VOLUME, int OUT_VOLUME, struct args myargs,
                      bool enable_ctrl_pkts = false) {
  srand(time(NULL));

  // Load instruction sequence
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                                   myargs.xclbin, myargs.kernel);

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_in1 = xrt::bo(device, IN1_VOLUME * sizeof(T1), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(3));
  auto bo_out = xrt::bo(device, OUT_VOLUME * sizeof(T3), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(4));

  // Placeholder dummy buffer objects because 0 causes seg faults
  auto bo_tmp1 = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  // If we enable control packets, then this is the input xrt buffer for that.
  // Otherwise, this is a dummy placedholder buffer.
  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));

  // Workaround so we declare a really small trace buffer when one is not used
  // Second workaround for driver issue. Allocate large trace buffer *4
  int tmp_trace_size = (myargs.trace_size > 0) ? myargs.trace_size * 4 : 1;
  auto bo_trace = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(7));

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize buffer objects
  T1 *bufIn1 = bo_in1.map<T1 *>();
  T3 *bufOut = bo_out.map<T3 *>();

  char *bufTrace = bo_trace.map<char *>();
  uint32_t *bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();

  init_bufIn1(bufIn1, IN1_VOLUME);
  init_bufOut(bufOut,
              OUT_VOLUME); // <<< what size do I pass it? reset with trace?
  if (myargs.trace_size > 0)
    memset(bufTrace, 0, myargs.trace_size);

  // Set control packet values
  if (myargs.trace_size > 0 && enable_ctrl_pkts) {
    bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
    bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
    if (myargs.verbosity >= 1) {
      std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
      std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
    }
  }

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  if (myargs.trace_size > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (enable_ctrl_pkts)
      bo_ctrlpkts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // ------------------------------------------------------
  // Initialize run configs
  // ------------------------------------------------------
  unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
  float npu_time_total = 0;
  float npu_time_min = 9999999;
  float npu_time_max = 0;

  int errors = 0;

  // ------------------------------------------------------
  // Main run loop
  // ------------------------------------------------------
  for (unsigned iter = 0; iter < num_iter; iter++) {

    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";

    // Run kernel
    if (myargs.verbosity >= 1)
      std::cout << "Running Kernel.\n";
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;
    // auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_out);
    // auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_out, 0, 0,
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_in1, bo_out, bo_tmp1,
                      bo_ctrlpkts, bo_trace);
    run.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    if (myargs.trace_size > 0)
      bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    if (iter < myargs.n_warmup_iterations)
      /* Warmup iterations do not count towards average runtime. */
      continue;

    // Copy output results and verify they are correct
    if (myargs.do_verify) {
      if (myargs.verbosity >= 1) {
        std::cout << "Verifying results ..." << std::endl;
      }
      auto vstart = std::chrono::system_clock::now();

      errors += verify_results(bufIn1, bufOut, IN1_VOLUME, myargs.verbosity);

      auto vstop = std::chrono::system_clock::now();
      float vtime =
          std::chrono::duration_cast<std::chrono::seconds>(vstop - vstart)
              .count();
      if (myargs.verbosity >= 1)
        std::cout << "Verify time: " << vtime << "secs." << std::endl;
    } else {
      if (myargs.verbosity >= 1)
        std::cout << "WARNING: results not verified." << std::endl;
    }

    // Write trace values if trace_size > 0 and first iteration
    if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations) {
      // std::cout << "Writing to offset " << OUT_VOLUME * sizeof(T3) <<
      // std::endl;
      std::cout << "Writing trace of size " << myargs.trace_size << std::endl;
      // test_utils::write_out_trace(((char *)bufOut) + OUT_VOLUME * sizeof(T3),
      test_utils::write_out_trace((char *)bufTrace, myargs.trace_size,
                                  myargs.trace_file);
    }

    // Write out control packet outputs
    if (enable_ctrl_pkts) {
      uint32_t *ctrl_pkt_out =
          (uint32_t *)(((char *)bufTrace) + myargs.trace_size);
      if (myargs.verbosity >= 1) {
        std::cout << "ctrl_pkt_out[0]:" << std::hex << ctrl_pkt_out[0]
                  << std::endl;
        std::cout << "ctrl_pkt_out[1]:" << std::hex << ctrl_pkt_out[1]
                  << std::endl;
      }
      int col = (ctrl_pkt_out[0] >> 21) & 0x7F;
      int row = (ctrl_pkt_out[0] >> 16) & 0x1F;
      if ((ctrl_pkt_out[1] >> 8) == 3)
        std::cout << "WARNING: Trace overflow detected in tile(" << row << ","
                  << col << ". Trace results may be invalid." << std::endl;
    }

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
            << "Avg NPU time: " << npu_time_total / myargs.n_iterations << "us."
            << std::endl;
  if (macs > 0)
    std::cout << "Avg NPU gflops: "
              << macs / (1000 * npu_time_total / myargs.n_iterations)
              << std::endl;

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
