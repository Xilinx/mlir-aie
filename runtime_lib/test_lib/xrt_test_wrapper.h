/* Copyright (C) 2025 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */

#include "cxxopts.hpp"
#include "test_utils.h"

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

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
 * Typed buffer descriptors for the variadic XRT test wrapper
 ******************************************************************************
 *
 * The host buffer ABI is the runtime_sequence operand list: the kernel takes
 * its data buffers in order, then -- only when enabled -- a control-packet
 * buffer, then a trace buffer, each appended at the tail. The wrapper builds
 * the kernel argument list to match exactly, so it works for any number of
 * inputs and never passes a buffer the kernel did not declare.
 */

// A typed input buffer: element type, element count, and initializer.
template <typename T>
struct InBuf {
  using elem_type = T;
  int volume;
  void (*init)(T *, int);
};

template <typename T>
InBuf<T> make_in(int volume, void (*init)(T *, int)) {
  return InBuf<T>{volume, init};
}

// A typed output buffer.
template <typename T>
struct OutBuf {
  using elem_type = T;
  int volume;
  void (*init)(T *, int);
};

template <typename T>
OutBuf<T> make_out(int volume, void (*init)(T *, int)) {
  return OutBuf<T>{volume, init};
}

/*
 ******************************************************************************
 * XRT based test wrapper for N inputs and 1 output
 ******************************************************************************
 *
 * Inputs are passed as InBuf<T> descriptors (any number, any types). The
 * verifier is called as verify(in0*, in1*, ..., out*, out_volume, verbosity)
 * and returns an error count.
 *
 * Host buffer indices follow the runtime_sequence operand list:
 *   group_id(3 + i) for data buffer i, then (if enabled) control packets, then
 *   (if trace_size > 0) the trace buffer -- each appended at the tail.
 */
template <typename Verify, typename TOut, typename... TIns>
int setup_and_run_aie(Verify verify_results, std::tuple<InBuf<TIns>...> inputs,
                      OutBuf<TOut> output, struct args myargs,
                      bool enable_ctrl_pkts = false,
                      int num_trailing_params = 0) {

  srand(time(NULL));

  constexpr int num_inputs = sizeof...(TIns);
  // Some designs place scalar/param buffers AFTER the output in the host ABI.
  // The IRON `transform` runtime sequence, for example, is ordered
  // [inputs, output, params].  `num_trailing_params` counts how many of the
  // trailing entries in `inputs` are actually such params: they are bound to
  // kernel arg slots *after* the output instead of before it.  The verifier
  // still receives all buffers in declared order (inputs..., params..., out).
  const int num_real_inputs = num_inputs - num_trailing_params;

  // Load instruction sequence
  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(myargs.instr);
  if (myargs.verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, myargs.verbosity,
                                   myargs.xclbin, myargs.kernel);

  // The fixed scalar args occupy kernel arg slots 0..2 (opcode, instr, ninstr).
  // Data buffers start at slot 3, i.e. group_id(3 + data_index).
  constexpr int kFirstDataArg = 3;
  // Output slot follows the *real* inputs; trailing params come after it.
  const int out_arg = kFirstDataArg + num_real_inputs;
  // Kernel arg slot for the idx-th declared `inputs` entry (real input or
  // trailing param).
  auto data_arg_slot = [&](int idx) {
    return (idx < num_real_inputs)
               ? (kFirstDataArg + idx)
               : (out_arg + 1 + (idx - num_real_inputs));
  };

  // Instruction buffer (kernel arg slot 1).
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (myargs.verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  // Allocate + initialize the input buffer objects, in order. Each input lands
  // at the next data arg slot.
  std::vector<xrt::bo> in_bos;
  in_bos.reserve(num_inputs);
  std::tuple<TIns *...> in_ptrs;

  // The verifier's SIZE argument is the first input's element count, matching
  // the historical wrapper contract (verify iterates the primary input).
  int first_in_volume = 0;

  auto alloc_inputs = [&](auto &...inb) {
    int idx = 0;
    auto one = [&](auto &desc, auto &ptr_slot) {
      using T = typename std::remove_reference_t<decltype(desc)>::elem_type;
      auto bo = xrt::bo(device, desc.volume * sizeof(T), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(data_arg_slot(idx)));
      T *mapped = bo.template map<T *>();
      desc.init(mapped, desc.volume);
      bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      ptr_slot = mapped;
      in_bos.push_back(std::move(bo));
      if (idx == 0)
        first_in_volume = desc.volume;
      idx++;
    };
    std::apply([&](auto &...slots) { (one(inb, slots), ...); }, in_ptrs);
  };
  std::apply(alloc_inputs, inputs);

  // Output buffer follows the real inputs (any params come after it).
  auto bo_out = xrt::bo(device, output.volume * sizeof(TOut),
                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(out_arg));
  TOut *bufOut = bo_out.map<TOut *>();
  output.init(bufOut, output.volume);
  bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Control-packet and trace buffers are appended (in that order) only when
  // enabled, matching what trace lowering appends to the runtime_sequence.
  // Control packets are a trace-side feature, so they are only part of the ABI
  // when tracing is also enabled.
  const bool use_ctrl_pkts = enable_ctrl_pkts && myargs.trace_size > 0;
  int next_arg = out_arg + 1 + num_trailing_params;

  xrt::bo bo_ctrlpkts;
  uint32_t *bufCtrlPkts = nullptr;
  int ctrlpkt_arg = -1;
  if (use_ctrl_pkts) {
    ctrlpkt_arg = next_arg++;
    bo_ctrlpkts = xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(ctrlpkt_arg));
    bufCtrlPkts = bo_ctrlpkts.map<uint32_t *>();
  }

  xrt::bo bo_trace;
  char *bufTrace = nullptr;
  int trace_arg = -1;
  if (myargs.trace_size > 0) {
    trace_arg = next_arg++;
    // Allocate a larger trace buffer (driver workaround) that also holds the
    // 8-byte control packet response when control packets are enabled.
    int alloc_trace_size = myargs.trace_size * 4;
    bo_trace = xrt::bo(device, alloc_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                       kernel.group_id(trace_arg));
    bufTrace = bo_trace.map<char *>();
    memset(bufTrace, 0, myargs.trace_size);
  }

  // Set control packet values.
  if (use_ctrl_pkts) {
    bufCtrlPkts[0] = create_ctrl_pkt(1, 0, 0x32004); // core status
    bufCtrlPkts[1] = create_ctrl_pkt(1, 0, 0x320D8); // trace status
    if (myargs.verbosity >= 1) {
      std::cout << "bufCtrlPkts[0]:" << std::hex << bufCtrlPkts[0] << std::endl;
      std::cout << "bufCtrlPkts[1]:" << std::hex << bufCtrlPkts[1] << std::endl;
    }
  }

  // Sync trace/ctrlpkt buffers to device.
  if (myargs.trace_size > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    if (use_ctrl_pkts)
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

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int opcode = 3;

    // Build the kernel argument list to match the declared host ABI exactly:
    // opcode, instr, ninstr, data buffers..., [ctrlpkt], [trace].
    xrt::run run(kernel);
    run.set_arg(0, opcode);
    run.set_arg(1, bo_instr);
    run.set_arg(2, instr_v.size());
    for (int i = 0; i < num_inputs; i++)
      run.set_arg(data_arg_slot(i), in_bos[i]);
    run.set_arg(out_arg, bo_out);
    if (ctrlpkt_arg >= 0)
      run.set_arg(ctrlpkt_arg, bo_ctrlpkts);
    if (trace_arg >= 0)
      run.set_arg(trace_arg, bo_trace);
    run.start();
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
      if (myargs.verbosity >= 1)
        std::cout << "Verifying results ..." << std::endl;
      auto vstart = std::chrono::system_clock::now();

      // verify(in0*, in1*, ..., out*, first_input_volume, verbosity)
      errors += std::apply(
          [&](auto *...in_ptr) {
            return verify_results(in_ptr..., bufOut, first_in_volume,
                                  myargs.verbosity);
          },
          in_ptrs);

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

    // Write trace values if trace_size > 0 and first non-warmup iteration
    if (myargs.trace_size > 0 && iter == myargs.n_warmup_iterations) {
      test_utils::write_out_trace(((char *)bufTrace), myargs.trace_size,
                                  myargs.trace_file);
    }

    // Write out control packet outputs
    if (use_ctrl_pkts) {
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
