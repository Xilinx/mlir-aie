//===- hrx_test_wrapper.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// HRX host backend for the example `test.cpp` files.
//
// It exposes the *same* `args`, `parse_args()`, and templated
// `setup_and_run_aie<...>()` surface the example `test.cpp` files already use,
// and dispatches through HRX (libhrx + the amdxdna HAL). A test.cpp keeps
// `#include "xrt_test_wrapper.h"`; defining `TEST_UTILS_USE_HRX` makes that
// header pull this one in instead, so no example source changes are needed.
//
// It consumes the `aiecc` artifacts (`final.xclbin` + `insts.bin`): the raw
// insts.bin TXN stream is handed straight to libhrx via
// `hrx_amdxdna_executable_create`, which builds the amdxdna XADX package and
// derives the (offset, arg_idx, addend) patch table from the transaction's
// BLOCKWRITE/DDR_PATCH ops internally -- there is no separate XADX helper or
// host-side patch-table extraction. An ELF input (aiecc --aie-generate-elf) is
// reduced to its .ctrltext (the TXN verbatim) so libhrx still sees those ops.
//
// The bindings list is [in1, in2, out] (2-input form) or [in1, out] (1-input
// form), so each buffer lands in the binding slot the transaction's DDR-patch
// ops expect.
//
//===----------------------------------------------------------------------===//

#ifndef HRX_TEST_WRAPPER_H
#define HRX_TEST_WRAPPER_H

#include "cxxopts.hpp"
#include "test_utils.h"

#include "hrx_amdxdna.h"
#include "hrx_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace hrx_test {

// ---------------------------------------------------------------------------
// Status check
// ---------------------------------------------------------------------------
inline void hrx_check(hrx_status_t status, const char *what) {
  if (hrx_status_is_ok(status))
    return;
  char *msg = nullptr;
  size_t len = 0;
  hrx_status_t s2 = hrx_status_to_string(status, &msg, &len);
  int code = hrx_status_code(status);
  std::string text = msg ? std::string(msg, len) : std::string("?");
  if (msg)
    hrx_status_free_message(msg);
  hrx_status_ignore(s2);
  hrx_status_ignore(status);
  throw std::runtime_error(std::string(what) + " failed (hrx status code " +
                           std::to_string(code) + "): " + text);
}

// ---------------------------------------------------------------------------
// Process-wide HRX context (device + dispatch stream), created once.
// ---------------------------------------------------------------------------
class Context {
public:
  static Context &get() {
    static Context instance;
    return instance;
  }

  hrx_device_t device = nullptr;
  hrx_stream_t stream = nullptr;

private:
  Context() {
    hrx_check(hrx_gpu_initialize(0), "hrx_gpu_initialize");
    hrx_check(hrx_gpu_device_get(0, &device), "hrx_gpu_device_get");
    hrx_check(hrx_stream_create(device, 0, &stream), "hrx_stream_create");
  }
};

// ---------------------------------------------------------------------------
// A persistent, host-mapped, device-visible HRX buffer.
// Coherence is maintained explicitly via flush_range / invalidate_range.
// ---------------------------------------------------------------------------
class Buffer {
public:
  Buffer(hrx_stream_t stream, size_t nbytes) : nbytes_(nbytes) {
    size_t alloc = nbytes ? nbytes : 1; // HRX rejects 0-size allocations
    hrx_check(hrx_buffer_allocate(stream, alloc,
                                  HRX_MEMORY_TYPE_HOST_LOCAL |
                                      HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                                  HRX_BUFFER_USAGE_DEFAULT |
                                      HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                                  &handle_),
              "hrx_buffer_allocate");
    // PERSISTENT keeps host_ptr_ valid across dispatches (paired with
    // HRX_BUFFER_USAGE_MAPPING_PERSISTENT above).
    hrx_check(hrx_buffer_map_with_mode(handle_, HRX_MAPPING_MODE_PERSISTENT,
                                       HRX_MAP_READ | HRX_MAP_WRITE, 0, alloc,
                                       &host_ptr_),
              "hrx_buffer_map_with_mode");
  }

  ~Buffer() {
    if (handle_)
      hrx_buffer_release(handle_);
  }

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  void flush() {
    if (nbytes_)
      hrx_check(hrx_buffer_flush_range(handle_, 0, nbytes_),
                "hrx_buffer_flush_range");
  }
  void invalidate() {
    if (nbytes_)
      hrx_check(hrx_buffer_invalidate_range(handle_, 0, nbytes_),
                "hrx_buffer_invalidate_range");
  }

  hrx_buffer_t handle() const { return handle_; }
  void *host_ptr() const { return host_ptr_; }
  size_t nbytes() const { return nbytes_; }

private:
  hrx_buffer_t handle_ = nullptr;
  void *host_ptr_ = nullptr;
  size_t nbytes_ = 0;
};

// ---------------------------------------------------------------------------
// Extract the XAie transaction words (.ctrltext) from a control ELF (ELF32,
// little-endian). Used only for an ELF input; .ctrltext is the TXN verbatim.
// ---------------------------------------------------------------------------
inline std::vector<uint8_t>
control_code_from_elf(const std::vector<uint8_t> &d) {
  auto rd32 = [&](size_t off) -> uint32_t {
    uint32_t v;
    std::memcpy(&v, d.data() + off, 4);
    return v;
  };
  auto rd16 = [&](size_t off) -> uint16_t {
    uint16_t v;
    std::memcpy(&v, d.data() + off, 2);
    return v;
  };

  if (d.size() < 52 || d[0] != 0x7f || d[1] != 'E' || d[2] != 'L' ||
      d[3] != 'F')
    throw std::runtime_error("control ELF is not a valid ELF32 file");

  uint32_t e_shoff = rd32(0x20);
  uint16_t e_shentsize = rd16(0x2E);
  uint16_t e_shnum = rd16(0x30);
  uint16_t e_shstrndx = rd16(0x32);

  auto sh = [&](uint32_t i, uint32_t f) -> size_t {
    return e_shoff + static_cast<size_t>(i) * e_shentsize + f;
  };
  uint32_t shstr_off = rd32(sh(e_shstrndx, 0x10));

  auto sname = [&](uint32_t i) -> std::string {
    uint32_t nm = rd32(sh(i, 0));
    size_t off = shstr_off + nm, end = off;
    while (end < d.size() && d[end] != 0)
      ++end;
    return std::string(reinterpret_cast<const char *>(d.data() + off),
                       end - off);
  };

  for (uint32_t i = 0; i < e_shnum; ++i) {
    if (sname(i) == ".ctrltext") {
      uint32_t coff = rd32(sh(i, 0x10));
      uint32_t csize = rd32(sh(i, 0x14));
      return std::vector<uint8_t>(d.begin() + coff, d.begin() + coff + csize);
    }
  }
  throw std::runtime_error("control ELF has no .ctrltext section");
}

// ---------------------------------------------------------------------------
// Read the XAie transaction libhrx patches from. For a raw insts.bin the
// transaction is the file verbatim; for an ELF input (aiecc --aie-generate-elf)
// it is the .ctrltext section (also the TXN verbatim). libhrx derives the patch
// table from the transaction, so no host-side patch-table extraction is needed.
// ---------------------------------------------------------------------------
inline std::vector<uint8_t> read_transaction(const std::string &insts_path) {
  std::ifstream f(insts_path, std::ios::binary);
  if (!f)
    throw std::runtime_error("could not read insts: " + insts_path);
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());

  if (data.size() >= 4 && data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' &&
      data[3] == 'F')
    return control_code_from_elf(data);
  return data;
}

// ---------------------------------------------------------------------------
// Build + load an HRX executable from the artifacts, returning (exe, ordinal).
// ---------------------------------------------------------------------------
struct LoadedKernel {
  hrx_executable_t exe = nullptr;
  uint32_t ordinal = 0;
};

inline LoadedKernel load_kernel(const std::string &xclbin_path,
                                const std::string &insts_path,
                                const std::string &kernel_name) {
  // Read xclbin bytes.
  std::ifstream xf(xclbin_path, std::ios::binary);
  if (!xf)
    throw std::runtime_error("could not read xclbin: " + xclbin_path);
  std::vector<uint8_t> xclbin((std::istreambuf_iterator<char>(xf)),
                              std::istreambuf_iterator<char>());

  // Resolve the XAie transaction. For a raw insts.bin this is the TXN verbatim;
  // for an ELF input it is the .ctrltext (also the TXN verbatim). libhrx
  // derives the patch table from the transaction internally.
  std::vector<uint8_t> transaction = read_transaction(insts_path);

  // Describe the executable to libhrx: one xclbin + one CREATE entry point with
  // a single run carrying the transaction. libhrx builds the amdxdna XADX
  // package and derives the patch table internally.
  hrx_amdxdna_executable_run_t run{};
  run.record_length = sizeof(run);
  run.abi_version = HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0;
  run.transaction.data = transaction.data();
  run.transaction.data_length = transaction.size();
  run.data_payload.data = nullptr;
  run.data_payload.data_length = 0;

  hrx_amdxdna_executable_entry_point_t entry{};
  entry.record_length = sizeof(entry);
  entry.abi_version = HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0;
  entry.name.data = kernel_name.c_str();
  entry.name.size = kernel_name.size();
  entry.context_mode = HRX_AMDXDNA_CONTEXT_MODE_CREATE;
  entry.xclbin_ordinal = 0;
  entry.pdi_ordinal = 0;
  entry.source_line = 0;
  entry.source_file.data = nullptr;
  entry.source_file.size = 0;
  entry.runs = &run;
  entry.run_count = 1;

  hrx_const_byte_span_t xclbin_span{};
  xclbin_span.data = xclbin.data();
  xclbin_span.data_length = xclbin.size();

  hrx_amdxdna_executable_create_params_t params{};
  params.record_length = sizeof(params);
  params.abi_version = HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0;
  params.flags = 0;
  params.reserved = 0;
  params.xclbins = &xclbin_span;
  params.xclbin_count = 1;
  params.entry_points = &entry;
  params.entry_point_count = 1;

  Context &ctx = Context::get();
  LoadedKernel lk;
  hrx_check(hrx_amdxdna_executable_create(ctx.device, &params, &lk.exe),
            "hrx_amdxdna_executable_create");
  hrx_check(hrx_executable_lookup_export_by_name(lk.exe, kernel_name.c_str(),
                                                 &lk.ordinal),
            "hrx_executable_lookup_export_by_name");
  return lk;
}

// ---------------------------------------------------------------------------
// One dispatch + synchronize of `lk` with `bindings`. Returns elapsed us.
// ---------------------------------------------------------------------------
inline double dispatch_once(const LoadedKernel &lk,
                            const std::vector<Buffer *> &bindings) {
  Context &ctx = Context::get();
  std::vector<hrx_buffer_ref_t> refs(bindings.size());
  for (size_t i = 0; i < bindings.size(); ++i) {
    refs[i].buffer = bindings[i]->handle();
    refs[i].offset = 0;
    refs[i].length = bindings[i]->nbytes();
  }
  hrx_dispatch_config_t cfg{};
  cfg.workgroup_count[0] = cfg.workgroup_count[1] = cfg.workgroup_count[2] = 1;
  cfg.workgroup_size[0] = cfg.workgroup_size[1] = cfg.workgroup_size[2] = 1;
  cfg.subgroup_size = 0;

  auto start = std::chrono::high_resolution_clock::now();
  hrx_check(hrx_stream_dispatch(ctx.stream, lk.exe, lk.ordinal, &cfg, nullptr,
                                0, refs.data(), refs.size(),
                                HRX_DISPATCH_FLAG_NONE),
            "hrx_stream_dispatch");
  hrx_check(hrx_stream_synchronize(ctx.stream), "hrx_stream_synchronize");
  auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
      .count();
}

// ---------------------------------------------------------------------------
// Multi-dispatch chain. Each entry is one kernel dispatch with its own
// bindings; all are recorded, in order, into a single HRX command buffer.
// HRX inserts an execution + memory barrier after every dispatch, so a later
// run observes an earlier run's device writes (producer -> consumer chains
// work, e.g. one run's output buffer is the next run's input). A single
// `synchronize` then submits the whole batch as one execution -- the amdxdna
// HAL lowers a multi-dispatch command buffer into one `ERT_CMD_CHAIN`. Entries
// may share one LoadedKernel (re-dispatch the same kernel) or use different
// ones (a true multi-kernel pipeline). Returns elapsed us for the whole chain.
// ---------------------------------------------------------------------------
struct ChainRun {
  const LoadedKernel *lk;
  std::vector<Buffer *> bindings;
};

inline double dispatch_chain(const std::vector<ChainRun> &runs) {
  Context &ctx = Context::get();
  hrx_dispatch_config_t cfg{};
  cfg.workgroup_count[0] = cfg.workgroup_count[1] = cfg.workgroup_count[2] = 1;
  cfg.workgroup_size[0] = cfg.workgroup_size[1] = cfg.workgroup_size[2] = 1;
  cfg.subgroup_size = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (const auto &run : runs) {
    // hrx_stream_dispatch copies the binding refs into the recorded command, so
    // a per-iteration local is safe; the batch stays pending until synchronize.
    std::vector<hrx_buffer_ref_t> refs(run.bindings.size());
    for (size_t i = 0; i < run.bindings.size(); ++i) {
      refs[i].buffer = run.bindings[i]->handle();
      refs[i].offset = 0;
      refs[i].length = run.bindings[i]->nbytes();
    }
    hrx_check(hrx_stream_dispatch(ctx.stream, run.lk->exe, run.lk->ordinal,
                                  &cfg, nullptr, 0, refs.data(), refs.size(),
                                  HRX_DISPATCH_FLAG_NONE),
              "hrx_stream_dispatch");
  }
  hrx_check(hrx_stream_synchronize(ctx.stream), "hrx_stream_synchronize");
  auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
      .count();
}

// ---------------------------------------------------------------------------
// Reject features the HRX backend does not implement (trace capture, control
// packets). Throwing here -- rather than warning and continuing -- means a
// caller that asked for trace/ctrl-pkts gets a hard error instead of a
// misleading PASS. Trace/control-packet designs are not supported on HRX.
// ---------------------------------------------------------------------------
inline void reject_unsupported_features(int trace_size, bool enable_ctrl_pkts) {
  if (trace_size > 0)
    throw std::runtime_error(
        "trace capture is not supported on the HRX backend "
        "(--trace_sz/trace_size > 0). Re-run without trace, or use the XRT "
        "backend for trace-enabled designs.");
  if (enable_ctrl_pkts)
    throw std::runtime_error(
        "control packets are not supported on the HRX backend "
        "(enable_ctrl_pkts=true). Use the XRT backend for control-packet "
        "designs.");
}

inline void report_timing(double total, double mn, double mx, int n_iters) {
  std::cout << std::endl
            << "Avg NPU time: " << (n_iters ? total / n_iters : 0.0) << "us."
            << std::endl;
  std::cout << std::endl << "Min NPU time: " << mn << "us." << std::endl;
  std::cout << std::endl << "Max NPU time: " << mx << "us." << std::endl;
}

} // namespace hrx_test

// ---------------------------------------------------------------------------
// Public surface: identical to xrt_test_wrapper.h
// ---------------------------------------------------------------------------
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

inline struct args parse_args(int argc, const char *argv[]) {
  cxxopts::Options options("HRX Test Wrapper");
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

/*
 ******************************************************************************
 * Typed buffer descriptors for the variadic HRX test wrapper
 ******************************************************************************
 *
 * Typed buffer descriptors so example test.cpp files that use the variadic form
 *   setup_and_run_aie(verify, std::make_tuple(make_in<T>(...), ...),
 *                     make_out<T>(...), myargs)
 * build unchanged against the HRX backend. Buffer binding order: data buffers
 * in order, then the output buffer.
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
 * HRX based test wrapper for N inputs and 1 output (variadic)
 ******************************************************************************
 *
 * Inputs are passed as InBuf<T> descriptors (any number, any types). The
 * verifier is called as verify(in0*, in1*, ..., out*, first_input_volume,
 * verbosity) and returns an error count -- matching the XRT wrapper contract.
 */
template <typename Verify, typename TOut, typename... TIns>
int setup_and_run_aie(Verify verify_results, std::tuple<InBuf<TIns>...> inputs,
                      OutBuf<TOut> output, struct args myargs,
                      bool enable_ctrl_pkts = false) {
  using namespace hrx_test;
  srand(time(NULL));

  try {
    // The HRX backend does not (yet) implement trace capture or control-packet
    // configuration. Fail loudly rather than silently ignoring the request:
    // running without them would emit a misleading "PASS" with no trace / an
    // unconfigured control path. Trace/control-packet designs must use XRT.
    reject_unsupported_features(myargs.trace_size, enable_ctrl_pkts);

    Context &ctx = Context::get();
    LoadedKernel lk = load_kernel(myargs.xclbin, myargs.instr, myargs.kernel);

    constexpr int num_inputs = sizeof...(TIns);

    // Allocate + initialize the input buffers, in order. Buffer is non-movable
    // (owns an HRX allocation), so hold them via unique_ptr.
    std::vector<std::unique_ptr<Buffer>> in_bufs;
    in_bufs.reserve(num_inputs);
    std::tuple<TIns *...> in_ptrs;
    // The verifier's SIZE argument is the first input's element count, matching
    // the historical wrapper contract (verify iterates the primary input).
    int first_in_volume = 0;

    auto alloc_inputs = [&](auto &...inb) {
      int idx = 0;
      auto one = [&](auto &desc, auto &ptr_slot) {
        using T = typename std::remove_reference_t<decltype(desc)>::elem_type;
        auto buf = std::make_unique<Buffer>(ctx.stream,
                                            (size_t)desc.volume * sizeof(T));
        T *mapped = reinterpret_cast<T *>(buf->host_ptr());
        desc.init(mapped, desc.volume);
        buf->flush();
        ptr_slot = mapped;
        if (idx == 0)
          first_in_volume = desc.volume;
        in_bufs.push_back(std::move(buf));
        idx++;
      };
      std::apply([&](auto &...slots) { (one(inb, slots), ...); }, in_ptrs);
    };
    std::apply(alloc_inputs, inputs);

    // Output buffer follows the inputs.
    Buffer bo_out(ctx.stream, (size_t)output.volume * sizeof(TOut));
    TOut *bufOut = reinterpret_cast<TOut *>(bo_out.host_ptr());
    output.init(bufOut, output.volume);
    bo_out.flush();

    // Binding order: data buffers in order, then out.
    std::vector<Buffer *> bindings;
    bindings.reserve(num_inputs + 1);
    for (auto &b : in_bufs)
      bindings.push_back(b.get());
    bindings.push_back(&bo_out);

    unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
    double npu_time_total = 0, npu_time_min = 1e30, npu_time_max = 0;
    int errors = 0;

    for (unsigned iter = 0; iter < num_iter; iter++) {
      double us = dispatch_once(lk, bindings);
      bo_out.invalidate();

      if (iter < (unsigned)myargs.n_warmup_iterations)
        continue;

      if (myargs.do_verify) {
        // verify(in0*, in1*, ..., out*, first_input_volume, verbosity)
        errors += std::apply(
            [&](auto *...in_ptr) {
              return verify_results(in_ptr..., bufOut, first_in_volume,
                                    myargs.verbosity);
            },
            in_ptrs);
      }
      npu_time_total += us;
      npu_time_min = us < npu_time_min ? us : npu_time_min;
      npu_time_max = us > npu_time_max ? us : npu_time_max;
    }

    report_timing(npu_time_total, npu_time_min, npu_time_max,
                  myargs.n_iterations);

    if (!errors) {
      std::cout << "\nPASS!\n\n";
      return 0;
    }
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "\nHRX error: " << e.what() << "\n\nFailed.\n\n";
    return 1;
  }
}

/*
 ******************************************************************************
 * HRX based test wrapper for 2 inputs and 1 output (legacy fixed-arity)
 ******************************************************************************
 */
template <typename T1, typename T2, typename T3, void (*init_bufIn1)(T1 *, int),
          void (*init_bufIn2)(T2 *, int), void (*init_bufOut)(T3 *, int),
          int (*verify_results)(T1 *, T2 *, T3 *, int, int)>
int setup_and_run_aie(int IN1_VOLUME, int IN2_VOLUME, int OUT_VOLUME,
                      struct args myargs, bool enable_ctrl_pkts = false) {
  using namespace hrx_test;
  srand(time(NULL));

  try {
    // The HRX backend does not (yet) implement trace capture or control-packet
    // configuration. Fail loudly rather than silently ignoring the request:
    // running without them would emit a misleading "PASS" with no trace / an
    // unconfigured control path. Trace/control-packet designs must use XRT.
    reject_unsupported_features(myargs.trace_size, enable_ctrl_pkts);

    Context &ctx = Context::get();
    LoadedKernel lk = load_kernel(myargs.xclbin, myargs.instr, myargs.kernel);

    Buffer bo_in1(ctx.stream, (size_t)IN1_VOLUME * sizeof(T1));
    Buffer bo_in2(ctx.stream, (size_t)IN2_VOLUME * sizeof(T2));
    Buffer bo_out(ctx.stream, (size_t)OUT_VOLUME * sizeof(T3));

    init_bufIn1(reinterpret_cast<T1 *>(bo_in1.host_ptr()), IN1_VOLUME);
    init_bufIn2(reinterpret_cast<T2 *>(bo_in2.host_ptr()), IN2_VOLUME);
    init_bufOut(reinterpret_cast<T3 *>(bo_out.host_ptr()), OUT_VOLUME);
    bo_in1.flush();
    bo_in2.flush();
    bo_out.flush();

    // Binding order: [in1, in2, out].
    std::vector<Buffer *> bindings = {&bo_in1, &bo_in2, &bo_out};

    unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
    double npu_time_total = 0, npu_time_min = 1e30, npu_time_max = 0;
    int errors = 0;

    for (unsigned iter = 0; iter < num_iter; iter++) {
      double us = dispatch_once(lk, bindings);
      bo_out.invalidate();

      if (iter < (unsigned)myargs.n_warmup_iterations)
        continue;

      if (myargs.do_verify) {
        errors += verify_results(reinterpret_cast<T1 *>(bo_in1.host_ptr()),
                                 reinterpret_cast<T2 *>(bo_in2.host_ptr()),
                                 reinterpret_cast<T3 *>(bo_out.host_ptr()),
                                 IN1_VOLUME, myargs.verbosity);
      }
      npu_time_total += us;
      npu_time_min = us < npu_time_min ? us : npu_time_min;
      npu_time_max = us > npu_time_max ? us : npu_time_max;
    }

    report_timing(npu_time_total, npu_time_min, npu_time_max,
                  myargs.n_iterations);

    if (!errors) {
      std::cout << "\nPASS!\n\n";
      return 0;
    }
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "\nHRX error: " << e.what() << "\n\nFailed.\n\n";
    return 1;
  }
}

/*
 ******************************************************************************
 * HRX based test wrapper for 1 input and 1 output
 ******************************************************************************
 */
template <typename T1, typename T3, void (*init_bufIn1)(T1 *, int),
          void (*init_bufOut)(T3 *, int),
          int (*verify_results)(T1 *, T3 *, int, int)>
int setup_and_run_aie(int IN1_VOLUME, int OUT_VOLUME, struct args myargs,
                      bool enable_ctrl_pkts = false) {
  using namespace hrx_test;
  srand(time(NULL));

  try {
    // The HRX backend does not (yet) implement trace capture or control-packet
    // configuration. Fail loudly rather than silently ignoring the request:
    // running without them would emit a misleading "PASS" with no trace / an
    // unconfigured control path. Trace/control-packet designs must use XRT.
    reject_unsupported_features(myargs.trace_size, enable_ctrl_pkts);

    Context &ctx = Context::get();
    LoadedKernel lk = load_kernel(myargs.xclbin, myargs.instr, myargs.kernel);

    Buffer bo_in1(ctx.stream, (size_t)IN1_VOLUME * sizeof(T1));
    Buffer bo_out(ctx.stream, (size_t)OUT_VOLUME * sizeof(T3));

    init_bufIn1(reinterpret_cast<T1 *>(bo_in1.host_ptr()), IN1_VOLUME);
    init_bufOut(reinterpret_cast<T3 *>(bo_out.host_ptr()), OUT_VOLUME);
    bo_in1.flush();
    bo_out.flush();

    // Binding order: [in1, out].
    std::vector<Buffer *> bindings = {&bo_in1, &bo_out};

    unsigned num_iter = myargs.n_iterations + myargs.n_warmup_iterations;
    double npu_time_total = 0, npu_time_min = 1e30, npu_time_max = 0;
    int errors = 0;

    for (unsigned iter = 0; iter < num_iter; iter++) {
      double us = dispatch_once(lk, bindings);
      bo_out.invalidate();

      if (iter < (unsigned)myargs.n_warmup_iterations)
        continue;

      if (myargs.do_verify) {
        errors += verify_results(reinterpret_cast<T1 *>(bo_in1.host_ptr()),
                                 reinterpret_cast<T3 *>(bo_out.host_ptr()),
                                 IN1_VOLUME, myargs.verbosity);
      }
      npu_time_total += us;
      npu_time_min = us < npu_time_min ? us : npu_time_min;
      npu_time_max = us > npu_time_max ? us : npu_time_max;
    }

    report_timing(npu_time_total, npu_time_min, npu_time_max,
                  myargs.n_iterations);

    if (!errors) {
      std::cout << "\nPASS!\n\n";
      return 0;
    }
    std::cout << "\nError count: " << errors << "\n\n";
    std::cout << "\nFailed.\n\n";
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "\nHRX error: " << e.what() << "\n\nFailed.\n\n";
    return 1;
  }
}

#endif // HRX_TEST_WRAPPER_H
