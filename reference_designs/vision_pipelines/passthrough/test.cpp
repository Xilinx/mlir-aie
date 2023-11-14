//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

#include "xrt/xrt_bo.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "OpenCVUtils.h"
#include "xrtUtils.h"

constexpr int testImageWidth = 1920*4;//64*8;
constexpr int testImageHeight = 1080;//9;
constexpr int testImageSize = testImageWidth*testImageHeight;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("xclbin,x", po::value<std::string>()->required(), "the input xclbin path")
    ("image,p", po::value<std::string>(), "the input image")
    ("outfile,o", po::value<std::string>()->default_value("passThroughOut_test.jpg"), "the output image")
    ("kernel,k", po::value<std::string>()->required(), "the kernel name in the XCLBIN (for instance PP_PRE_FD)")
    ("verbosity,v", po::value<int>()->default_value(0), "the verbosity of the output")
    ("instr,i", po::value<std::string>()->required(), "path of file containing userspace instructions to be sent to the LX6");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  try {
    check_arg_file_exists(vm, "xclbin");
    check_arg_file_exists(vm, "instr");
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
  }

  // Read the input image or generate random one if no input file argument provided
  cv::Mat inImageGray;
  cv::String fileIn;
  if(vm.count("image")) {
    fileIn = vm["image"].as<std::string>(); //"/group/xrlabs/imagesAndVideos/images/minion128x128.jpg";
    initializeSingleGrayImageTest(fileIn, inImageGray);
  }
  else
  {
    fileIn = "RANDOM";
    inImageGray = cv::Mat(testImageHeight, testImageWidth, CV_8UC1);
    cv::randu(inImageGray, cv::Scalar(0), cv::Scalar(255));
  }
  
  cv::String fileOut = vm["outfile"].as<std::string>();//"passThroughOut_test.jpg";
  printf("Load input image %s and run passThrough\n", fileIn.c_str());
  
  cv::resize(inImageGray,inImageGray,cv::Size(testImageWidth,testImageHeight));

  // Calculate OpenCV refence for passThrough   
  cv::Mat outImageReference = inImageGray.clone();
  cv::Mat outImageTest(testImageHeight, testImageWidth, CV_8UC1);

  // Load instruction sequence
  std::vector<uint32_t> instr_v = load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  initXrtLoadKernel(device,kernel,verbosity, vm["xclbin"].as<std::string>(), vm["kernel"].as<std::string>());  

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inA = xrt::bo(device, inImageGray.total() * inImageGray.elemSize(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inB = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, (outImageTest.total() * outImageTest.elemSize()), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t *bufInA = bo_inA.map<uint8_t *>();
  
  // Copyt cv::Mat input image to xrt buffer object
  memcpy(bufInA, inImageGray.data, (inImageGray.total() * inImageGray.elemSize()));

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_inB, bo_out);
  run.wait();

  // Sync device to host memories
  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Store result in cv::Mat
  uint8_t *bufOut = bo_out.map<uint8_t *>();
  memcpy(outImageTest.data, bufOut, (outImageTest.total() * outImageTest.elemSize()));

  // Compare to OpenCV reference
  int numberOfDifferences = 0;
	double errorPerPixel = 0;
	imageCompare(outImageTest, outImageReference, numberOfDifferences, errorPerPixel, true, false);
  printf("Number of differences: %d, average L1 error: %f\n", numberOfDifferences, errorPerPixel);

  cv::imwrite(fileOut, outImageTest);
  
  // Print Pass/Fail result of our test
  int res = 0;
  if (numberOfDifferences == 0) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("Testing passThrough done!\n");
  return res;
}