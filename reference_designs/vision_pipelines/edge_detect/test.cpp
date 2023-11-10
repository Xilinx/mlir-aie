
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

double epsilon = 0.1;

constexpr int testImageWidth = 64;
constexpr int testImageHeight = 36;
constexpr int testImageSize = testImageWidth*testImageHeight;
constexpr int kernelSize = 3;

namespace po = boost::program_options;

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("xclbin,x", po::value<std::string>()->required(), "the input xclbin path")
    ("image,p", po::value<std::string>(), "the input image")
    ("outfile,o", po::value<std::string>()->default_value("edgeDetectOut_test.jpg"), "the output image")
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

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");

  // Read the input image or generate random one if no input file argument provided
  cv::Mat inImage, inImageRGBA;
  cv::String fileIn;
  if(vm.count("image")) {
    fileIn = vm["image"].as<std::string>(); //"/group/xrlabs/imagesAndVideos/images/minion128x128.jpg";
    initializeSingleImageTest(fileIn, inImage);
  }
  else
  {
    fileIn = "RANDOM";
    inImage = cv::Mat(testImageHeight, testImageWidth, CV_8UC3);
    cv::randu(inImage, cv::Scalar(0,0,0), cv::Scalar(255,255,255));
  }
  
  cv::String fileOut = vm["outfile"].as<std::string>();//"edgeDetectOut_test.jpg";
  printf("Load input image %s and run edgeDetect\n", fileIn.c_str());
  
  cv::resize(inImage,inImage,cv::Size(testImageWidth,testImageHeight));
  cv::cvtColor(inImage,inImageRGBA,cv::COLOR_BGR2RGBA);

  // Calculate OpenCV refence for edgeDetect   
  cv::Mat inImageGray, imageFiltered, imageThreshold, imageThresholdBRG, outImageReference, outImageTestBGR;
  cv::cvtColor(inImage,inImageGray,cv::COLOR_BGR2GRAY);
  cv::Mat filterKernel;
  cv::Point anchor( -1, -1 );
	double delta = 0.0; //no delta added to filtered pixels
	int ddepth = -1; //dst type equals src type
  //filterKernel = cv::Mat::ones( kernelSize, kernelSize, CV_32F )/ (float)(kernelSize*kernelSize);
  filterKernel = (cv::Mat_<float>(kernelSize,kernelSize) << 0, 1, 0, 1, -4, 1, 0, 1, 0); // Laplacian, high pass
  cv::filter2D(inImageGray, imageFiltered, ddepth , filterKernel, anchor, delta, cv::BORDER_REPLICATE );
  cv::threshold(imageFiltered, imageThreshold, 10,255,cv::THRESH_BINARY);
  cv::cvtColor(imageThreshold,imageThresholdBRG,cv::COLOR_GRAY2BGR);
  double alpha = 1.0;
  double beta = 1.0;
  double gamma = 0.0;
  cv::addWeighted(imageThresholdBRG,alpha,inImage,beta,gamma,outImageReference);
  cv::cvtColor(outImageReference,outImageReference,cv::COLOR_BGR2RGBA);

  cv::Mat outImageTest(testImageHeight, testImageWidth, CV_8UC4);

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
  auto bo_inA = xrt::bo(device, inImageRGBA.total() * inImageRGBA.elemSize(), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_inB = xrt::bo(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_out = xrt::bo(device, (outImageTest.total() * outImageTest.elemSize()), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";

  uint8_t *bufInA = bo_inA.map<uint8_t *>();
  
  // Copyt cv::Mat input image to xrt buffer object
  memcpy(bufInA, inImageRGBA.data, (inImageRGBA.total() * inImageRGBA.elemSize()));

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

  cv::cvtColor(outImageTest,outImageTestBGR,cv::COLOR_RGBA2BGR);
  cv::imwrite(fileOut, outImageTestBGR);
  
  // Print Pass/Fail result of our test
  int res = 0;
  if (numberOfDifferences == 0) {
    printf("PASS!\n");
    res = 0;
  } else {
    printf("Fail!\n");
    res = -1;
  }

  printf("Testing edgeDetect done!\n");
  return res;
}