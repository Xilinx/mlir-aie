//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr int IN_SIZE = 64;
constexpr int OUT_SIZE = 64;

struct Buffers {
  std::unique_ptr<xrt::bo> instr;
  std::unique_ptr<xrt::bo> in_a;
  std::unique_ptr<xrt::bo> in_b;
  std::unique_ptr<xrt::bo> out;
};

static std::string find_kernel_name(xrt::xclbin &xclbin,
                                    const std::string &kernel_prefix) {
  auto xkernels = xclbin.get_kernels();
  auto it = std::find_if(xkernels.begin(), xkernels.end(),
                         [&](xrt::xclbin::kernel &kernel) {
                           auto name = kernel.get_name();
                           std::cout << "Name: " << name << "\n";
                           return name.rfind(kernel_prefix, 0) == 0;
                         });
  if (it == xkernels.end())
    throw std::runtime_error("kernel not found: " + kernel_prefix);
  return it->get_name();
}

static Buffers make_buffers(xrt::device &instr_device, xrt::device &io_device,
                            xrt::kernel &kernel,
                            const std::vector<uint32_t> &instr_v) {
  Buffers buffers;
  buffers.instr = std::make_unique<xrt::bo>(
      instr_device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE,
      kernel.group_id(1));
  buffers.in_a = std::make_unique<xrt::bo>(
      io_device, IN_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
      kernel.group_id(3));
  buffers.in_b = std::make_unique<xrt::bo>(
      io_device, IN_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
      kernel.group_id(4));
  buffers.out = std::make_unique<xrt::bo>(
      io_device, OUT_SIZE * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
      kernel.group_id(5));
  return buffers;
}

static void initialize_buffers(Buffers &buffers,
                               const std::vector<uint32_t> &instr_v) {
  auto *buf_in_a = buffers.in_a->map<uint32_t *>();
  std::vector<uint32_t> src_vec_a;
  for (int i = 0; i < IN_SIZE; i++)
    src_vec_a.push_back(i + 1);
  std::memcpy(buf_in_a, src_vec_a.data(), src_vec_a.size() * sizeof(uint32_t));

  void *buf_instr = buffers.instr->map<void *>();
  std::memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(uint32_t));

  buffers.instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
  buffers.in_a->sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static int run_and_check(xrt::kernel &kernel, Buffers &buffers,
                         std::size_t instr_word_count) {
  auto run = kernel(3, *buffers.instr, instr_word_count, *buffers.in_a,
                    *buffers.in_b, *buffers.out);
  ert_cmd_state state = run.wait();
  if (state != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << state << "\n";
    return 1;
  }

  buffers.out->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  auto *buf_out = buffers.out->map<uint32_t *>();

  int errors = 0;
  for (uint32_t i = 0; i < OUT_SIZE; i++) {
    uint32_t ref = i + 2;
    if (*(buf_out + i) != ref) {
      std::cout << "Error in output " << *(buf_out + i) << " != " << ref
                << "\n";
      errors++;
    }
  }

  if (errors) {
    std::cout << "failed.\n";
    return 1;
  }

  std::cout << "PASS!\n";
  return 0;
}

static void destroy_io_bos(Buffers &buffers) {
  buffers.in_a.reset();
  buffers.in_b.reset();
  buffers.out.reset();
}

static int run_mode(const cxxopts::ParseResult &vm,
                    const std::vector<uint32_t> &instr_v,
                    const std::string &mode) {
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  std::string kernel_name = find_kernel_name(xclbin, vm["kernel"].as<std::string>());

  device.register_xclbin(xclbin);
  auto context = std::make_unique<xrt::hw_context>(device, xclbin.get_uuid());
  auto kernel = std::make_unique<xrt::kernel>(*context, kernel_name);

  if (mode == "ordered") {
    auto buffers = make_buffers(device, device, *kernel, instr_v);
    initialize_buffers(buffers, instr_v);
    return run_and_check(*kernel, buffers, instr_v.size());
  }

  if (mode == "stale-instr-bo") {
    auto instr_device = xrt::device(0);
    auto buffers = make_buffers(instr_device, device, *kernel, instr_v);
    initialize_buffers(buffers, instr_v);

    int result = run_and_check(*kernel, buffers, instr_v.size());
    if (result != 0)
      return result;

    destroy_io_bos(buffers);
    kernel.reset();
    context.reset();
    std::cout << "Destroying instruction BO after kernel/context.\n";
    buffers.instr.reset();
    return 0;
  }

  if (mode == "stale-io-bos") {
    auto io_device = xrt::device(0);
    auto buffers = make_buffers(device, io_device, *kernel, instr_v);
    initialize_buffers(buffers, instr_v);

    int result = run_and_check(*kernel, buffers, instr_v.size());
    if (result != 0)
      return result;

    buffers.instr.reset();
    kernel.reset();
    context.reset();
    std::cout << "Destroying input/output BOs after kernel/context.\n";
    destroy_io_bos(buffers);
    return 0;
  }

  throw std::runtime_error("unknown mode: " + mode);
}

int main(int argc, const char *argv[]) {
  cxxopts::Options options("xrt_handle_lifetime");
  test_utils::add_default_options(options);
  options.add_options()("mode", "Test mode",
                        cxxopts::value<std::string>()->default_value("ordered"));

  cxxopts::ParseResult vm;
  test_utils::parse_options(argc, argv, options, vm);

  std::string mode = vm["mode"].as<std::string>();
  auto instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());
  std::cout << "mode=" << mode << "\n";
  std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  return run_mode(vm, instr_v, mode);
}
