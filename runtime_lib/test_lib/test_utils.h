//===- test_utils.h ----------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// This file contains common helper functions for the generic host code

#ifndef _TEST_UTILS_H_
#define _TEST_UTILS_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>

#include <boost/program_options.hpp>
#include <cmath>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

namespace po = boost::program_options;

namespace test_utils {

void check_arg_file_exists(po::variables_map &vm_in, std::string name);

void add_default_options(po::options_description &desc);

void parse_options(int argc, const char *argv[], po::options_description &desc,
                   po::variables_map &vm);

std::vector<uint32_t> load_instr_sequence(std::string instr_path);

void init_xrt_load_kernel(xrt::device &device, xrt::kernel &kernel, int verbosity,
                          std::string xclbinFileName,
                          std::string kernelNameInXclbin);

static inline std::int16_t random_int16_t();

void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);

} // namespace test_utils

#endif // _TEST_UTILS_H_