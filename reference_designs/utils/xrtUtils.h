//===- xrtUtils.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _XRTUTILS_H_
#define _XRTUTILS_H_

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

std::vector<uint32_t> load_instr_sequence(std::string instr_path);
void check_arg_file_exists(boost::program_options::variables_map &vm_in, std::string name) ;

void initXrtLoadKernel(xrt::device &device, xrt::kernel &kernel, int verbosity, std::string xclbinFileName, std::string kernelNameInXclbin);

#endif //_XRTUTILS_H_