//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <boost/program_options.hpp>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "OpenCVUtils.h"
#include "xrtUtils.h"

// #define IMAGE_WIDTH_IN 256
// #define IMAGE_HEIGHT_IN 256

// #define IMAGE_WIDTH_IN 128
// #define IMAGE_HEIGHT_IN 64
constexpr int testImageWidth = COLORTHRESHOLD_WIDTH;
constexpr int testImageHeight = COLORTHRESHOLD_HEIGHT;

// #define IMAGE_WIDTH_OUT IMAGE_WIDTH_IN
// #define IMAGE_HEIGHT_OUT IMAGE_HEIGHT_IN

// #define IMAGE_AREA_IN (IMAGE_HEIGHT_IN * IMAGE_WIDTH_IN)
// #define IMAGE_AREA_OUT (IMAGE_HEIGHT_OUT * IMAGE_WIDTH_OUT)

constexpr int imageAreaIn = testImageWidth * testImageHeight;
constexpr int imageAreaOut = testImageWidth * testImageHeight;

// constexpr int IN_SIZE = (IMAGE_AREA_IN * sizeof(uint8_t));
// constexpr int OUT_SIZE = (IMAGE_AREA_OUT * sizeof(uint8_t));
constexpr int IN_SIZE = (imageAreaIn * sizeof(uint8_t));
constexpr int OUT_SIZE = (imageAreaOut * sizeof(uint8_t));

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  /*
   ****************************************************************************
   * Program arguments parsing
   ****************************************************************************
   */
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

  /*
   ****************************************************************************
   * Load instruction sequence
   ****************************************************************************
   */
  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  /*
   ****************************************************************************
   * Start the XRT context and load the kernel
   ****************************************************************************
   */
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  /*
   ****************************************************************************
   * Set up the buffer objects
   ****************************************************************************
   */

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_in =
      xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto debug =
      xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out =
      xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t *bufIn = bo_in.map<uint8_t *>();
  // Copy cv::Mat input image to xrt buffer object
  std::vector<uint8_t> srcVec;
  for (int i = 0; i < imageAreaIn; i++)
    srcVec.push_back(rand() % UINT8_MAX);
  memcpy(bufIn, srcVec.data(), (srcVec.size() * sizeof(uint8_t)));

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(bo_instr, instr_v.size(), bo_in, debug, bo_out);
  run.wait();

  // Sync device to host memories
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Store result in cv::Mat
  uint8_t *bufOut = bo_out.map<uint8_t *>();

  int errors = 0;
  int max_errors = 64000;

  /*
   ****************************************************************************
   * Compare to expected values
   ****************************************************************************
   */
  std::cout << std::dec;
  for (uint32_t i = 0; i < imageAreaOut; i++) {
    if (srcVec[i] <= 50) { // Obviously change this back to 100
      if (*(bufOut + i) != 0) {
        if (errors < max_errors)
          std::cout << "Error: " << (uint32_t)(uint8_t) * (bufOut + i) << " at "
                    << i << " should be zero "
                    << " : input " << std::dec << (uint32_t)(uint8_t)srcVec[i]
                    << std::endl;
        errors++;
      } else {
        //        std::cout << "Below threshold:   " << (uint32_t)(uint8_t) *
        //        (bufOut + i)
        //                  << " at " << i << " is correct "
        //                  << " : input " << std::dec <<
        //                  (uint32_t)(uint8_t)srcVec[i]
        //                 << std::endl;
      }
    } else {
      if (*(bufOut + i) != UINT8_MAX) {
        if (errors < max_errors)
          std::cout << "Error: " << (uint32_t)(uint8_t) * (bufOut + i) << " at "
                    << i << " should be UINT8_MAX "
                    << " : input " << std::dec << (uint32_t)(uint8_t)srcVec[i]
                    << std::endl;
        errors++;
      } else {
        //        std::cout << "Above threshold:  " << (uint32_t)(uint8_t) *
        //        (bufOut + i)
        //                  << " at " << i << " is correct "
        //                  << " : input " << std::dec <<
        //                  (uint32_t)(uint8_t)srcVec[i]
        //                  << std::endl;
      }
    }
  }

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
