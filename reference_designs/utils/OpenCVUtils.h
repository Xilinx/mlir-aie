//===- OpenCVUtils.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef _OPENCVUTILS_H_
#define _OPENCVUTILS_H_

#include <opencv2/core/core.hpp>

bool imageCompare(cv::Mat &test, cv::Mat &golden, int &numberOfDifferences, double &error, bool listPositionFirstDifference = false, bool displayResult = false, double epsilon = 0.01);
void readImage(const std::string &fileName, cv::Mat &image, int flags = 1);
void initializeSingleGrayImageTest(std::string fileName, cv::Mat &src);
void initializeSingleImageTest(std::string fileName, cv::Mat &src);

void addSaltPepperNoise(cv::Mat &src, float percentWhite, float percentBlack);
void medianBlur1D(cv::Mat src, cv::Mat &dst, int ksizeHor);

#endif // _OPENCVUTILS_H_