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

// #define IMAGE_WIDTH_IN 256
// #define IMAGE_HEIGHT_IN 256

#define IMAGE_WIDTH_IN 128
#define IMAGE_HEIGHT_IN 64

#define IMAGE_WIDTH_OUT IMAGE_WIDTH_IN
#define IMAGE_HEIGHT_OUT IMAGE_HEIGHT_IN

#define IMAGE_AREA_IN (IMAGE_HEIGHT_IN * IMAGE_WIDTH_IN)
#define IMAGE_AREA_OUT (IMAGE_HEIGHT_OUT * IMAGE_WIDTH_OUT)

constexpr int IN_SIZE = (IMAGE_AREA_IN * sizeof(uint8_t));
constexpr int OUT_SIZE = (IMAGE_AREA_OUT * sizeof(uint8_t));

namespace po = boost::program_options;

void check_arg_file_exists(po::variables_map &vm_in, std::string name) {
  if (!vm_in.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  } else {
    std::ifstream test(vm_in[name].as<std::string>());
    if (!test) {
      throw std::runtime_error("The " + name + " file " +
                               vm_in[name].as<std::string>() +
                               " does not exist.\n");
    }
  }
}

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
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

  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

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
  std::vector<uint8_t> srcVec;
  for (int i = 0; i < IMAGE_AREA_IN; i++)
    srcVec.push_back(rand() % UINT8_MAX);
  memcpy(bufIn, srcVec.data(), (srcVec.size() * sizeof(uint8_t)));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  auto run = kernel(bo_instr, instr_v.size(), bo_in, debug, bo_out);
  run.wait();

  bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  uint8_t *bufOut = bo_out.map<uint8_t *>();

  int errors = 0;
  int max_errors = 64000;

  std::cout << std::dec;
  for (uint32_t i = 0; i < IMAGE_AREA_OUT; i++) {
    if (srcVec[i] <= 50) { // Obviously change this back to 100
      if (*(bufOut + i) != 0) {
        if (errors < max_errors)
          std::cout << "Error: " << (uint8_t) * (bufOut + i) << " at " << i
                    << " should be zero "
                    << " : input " << (uint8_t)srcVec[i] << std::endl;
        errors++;
      } else {
        std::cout << "Wow:   " << (uint8_t) * (bufOut + i) << " at " << i
                  << " is correct "
                  << " : input " << (uint8_t)srcVec[i] << std::endl;
      }
    } else {
      if (*(bufOut + i) != UINT8_MAX) {
        if (errors < max_errors)
          std::cout << "Error: " << (uint8_t) * (bufOut + i) << " at " << i
                    << " should be UINT8_MAX "
                    << " : input " << (uint8_t)srcVec[i] << std::endl;
        errors++;
      } else {
        std::cout << "WowT:  " << (uint8_t) * (bufOut + i) << " at " << i
                  << " is correct "
                  << " : input " << (uint8_t)srcVec[i] << std::endl;
      }
    }
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
