//===- OpenCVUtils.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <random>

#include "OpenCVUtils.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;

void readImage(const std::string &fileName, Mat &image, int flags)
{
	// Load an image
	image = imread(fileName, flags);
	if (!image.data) {
		std::stringstream errorMessage;
		errorMessage << "Failed to load image " << fileName;
		CV_Error(cv::Error::StsBadArg, errorMessage.str());
	}
}

void initializeSingleGrayImageTest(std::string fileName, Mat &src)
{
	// check program argument and try to open input files
	try
	{
		// Load an image
		readImage(fileName, src, IMREAD_GRAYSCALE);
	}
	catch (std::exception &e)
	{
		const char* errorMessage = e.what();
		std::cerr << "Exception caught: " << errorMessage << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}
}

void initializeSingleImageTest(std::string fileName, Mat &src)
{
	// check program argument and try to open input files
	try
	{
		// Load an image
		readImage(fileName, src, IMREAD_COLOR);
	}
	catch (std::exception &e)
	{
		const char* errorMessage = e.what();
		std::cerr << "Exception caught: " << errorMessage << std::endl;
		std::cout << "\nhit enter to quit...";
		std::cin.get();
		exit(-1);
	}
}

template<typename T>
void listFirstDifferenceTwoMatrices(Mat &test, Mat &golden, double epsilon)
{
	bool foundFirstDifference = false;

	for (int i = 0; i < golden.rows; i++) {
		for (int j = 0; j < golden.cols; j++) {
			T *pGolden = golden.ptr<T>(i, j);
			T *pTest = test.ptr<T>(i, j);
			for (int k = 0; k < golden.channels(); k++) {
				double goldenValue = (double)pGolden[k];
				double testValue = (double)pTest[k];
				if (abs(goldenValue - testValue) > epsilon) {
					std::cout << "Mismatach at (" << i << "," << j << ") channel: " << k << " golden: " << goldenValue << " test: " << testValue << std::endl;
					foundFirstDifference = true;
				}
			}
			if (foundFirstDifference)
				break;
		}
		if (foundFirstDifference)
			break;
	}
}

bool imageCompare(Mat &test, Mat &golden, int &numberOfDifferences, double &error, bool listPositionFirstDifference, bool displayResult, double epsilon)
{
	bool identical = true;
	numberOfDifferences = -1;
	error = -1;

	if(test.rows != golden.rows || test.cols != golden.cols || test.channels() != golden.channels() || test.depth() != golden.depth()) {
		identical = false;
		std::cerr << "Error: image sizes do not match, golden: " << golden.cols << "x" << golden.rows << " test: " << test.cols << "x" << test.rows <<" golden channels " << golden.channels() << " test channels " << test.channels() << " golden depth " << golden.depth() << " test depth " << test.depth() <<  std::endl;
	}
	else {
		Mat difference = Mat(golden.size(), golden.type());

		error = norm(test, golden, NORM_L1);
		error /= (double)(test.rows*test.cols);
		absdiff(test, golden, difference);

		numberOfDifferences = 0;
		for (int k = 0; k < golden.channels(); k++) {
			Mat differenceChannel;
			extractChannel(difference, differenceChannel, k);
			numberOfDifferences += countNonZero(differenceChannel);
		}

		if (numberOfDifferences != 0)
			identical = false;

		if (displayResult)
		{
			/*const char* differenceWindowName = "difference";
			namedWindow(differenceWindowName, WINDOW_AUTOSIZE);
			imshow(differenceWindowName, difference);*/
		}

		if (listPositionFirstDifference && !identical) {
			switch (golden.depth()) {
				case CV_8U:
					listFirstDifferenceTwoMatrices<uchar>(test, golden, epsilon);
					break;
				case CV_8S:
					listFirstDifferenceTwoMatrices<char>(test, golden, epsilon);
				case CV_16U:
					listFirstDifferenceTwoMatrices<ushort>(test, golden, epsilon);
					break;
				case CV_16S:
					listFirstDifferenceTwoMatrices<short>(test, golden, epsilon);
					break;
				case CV_32S:
					listFirstDifferenceTwoMatrices<int>(test, golden, epsilon);
					break;
				case CV_32F:
					listFirstDifferenceTwoMatrices<float>(test, golden, epsilon);
					break;
				case CV_64F:
					listFirstDifferenceTwoMatrices<double>(test, golden, epsilon);
					break;
				default:
					std::cerr << "unexpected CV type" << std::endl;
			}
		}
	}

	return identical;
}

void addSaltPepperNoise(cv::Mat &src, float percentWhite, float percentBlack)
{       
    int amountWhite = (int)src.total() * percentWhite;
    int amountBlack = (int)src.total() * percentBlack;
    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distribRows(0, src.rows-1);
    std::uniform_int_distribution<> distribCols(0, src.cols-1);

	switch (src.channels()){
		case 1:
			for(int i=0; i<amountWhite; ++i){
        		int rRow = distribRows(gen);
        		int rCol = distribCols(gen);
            	src.at<uint8_t>(rRow,rCol) = 255;
    		}
    
    		for(int i=0; i<amountBlack; ++i){
        		int rRow = distribRows(gen);
        		int rCol = distribCols(gen);
            	src.at<uint8_t>(rRow,rCol) = 0;
			}
			break;
		case 3:
		    for(int i=0; i<amountWhite; ++i){
        		int rRow = distribRows(gen);
        		int rCol = distribCols(gen);
        		src.at<cv::Vec3b>(rRow,rCol)[0] = 255;
				src.at<cv::Vec3b>(rRow,rCol)[1] = 255;
				src.at<cv::Vec3b>(rRow,rCol)[2] = 255;
    		}
    
    		for(int i=0; i<amountBlack; ++i){
        		int rRow = distribRows(gen);
        		int rCol = distribCols(gen);
        		src.at<cv::Vec3b>(rRow,rCol)[0] = 0;
				src.at<cv::Vec3b>(rRow,rCol)[1] = 0;
				src.at<cv::Vec3b>(rRow,rCol)[2] = 0;

			}
			break;
		default:
			break;
	} 

}

// FROM: https://www.softwaretestinghelp.com/insertion-sort/
template <typename T>  
static void insertion_sort(T *a, int32_t size) {
  for(int k=1; k<size; k++)   
    {  
        T temp = a[k];  
        int j= k-1;  
        while(j>=0 && temp <= a[j])  
        {  
            a[j+1] = a[j];   
            j = j-1;  
        }  
        a[j+1] = temp;  
    }  
}

void median1DLine(uint8_t *inputLine, uint8_t *outputLine, int lineWidth, int ksizeHor) 
{
  uint8_t *in = new uint8_t[ksizeHor];
  
  // left
  in[0] = inputLine[0];
  
  for (int ki = 1; ki < ksizeHor; ki++) {
    in[ki] = inputLine[ki - 1];
  
  }

  insertion_sort<uint8_t>(in,ksizeHor);
  outputLine[0] = in[ksizeHor/2]; 
  
  // middle
  for (int i = 1; i < lineWidth-1; i++) {
    for (int ki = 0; ki < ksizeHor; ki++) {
      in[ki] = inputLine[i + ki - (ksizeHor/2)];
    }
    insertion_sort<uint8_t>(in,ksizeHor);
    outputLine[i] = in[ksizeHor/2];
  }
  
  // right
  for (int ki = 0; ki < (ksizeHor-1); ki++) {
    in[ki] = inputLine[lineWidth + ki - (ksizeHor-1)];
  }
  in[(ksizeHor-1)] = inputLine[lineWidth - 1];
  insertion_sort<uint8_t>(in,ksizeHor);
  outputLine[lineWidth-1] = in[ksizeHor/2];

  delete[] in;

}

void medianBlur1D(cv::Mat src, cv::Mat &dst, int ksizeHor)
{
    if(dst.empty())
        dst = cv::Mat(src.rows,src.cols,src.depth());
    
    for (int i = 0; i < src.rows; i++) {
        median1DLine(src.data+i*src.cols,dst.data+i*dst.cols,src.cols,ksizeHor);
		
    }
}
