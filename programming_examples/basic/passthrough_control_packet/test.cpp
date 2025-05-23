
#include <cassert>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "cxxopts.hpp"
#include "test_utils.h"

using int32 = std::int32_t;
typedef uint32_t TYPE_IN;
typedef uint32_t TYPE_OUT;

int main(int argc, const char *argv[]) {

  cxxopts::Options options("Passthrough control packet",
                           "Demonstrate control packet feature");
  options.add_options()("help,h", "produce help message")(
      "xclbin,x", "the input xclbin path", cxxopts::value<std::string>())(
      "kernel,k", "the kernel name in the XCLBIN (for instance PP_PRE_FD)",
      cxxopts::value<std::string>())("verbosity,v",
                                     "the verbosity of the output",
                                     cxxopts::value<int>()->default_value("0"))(
      "instr,i",
      "path of file containing userspace instructions to be sent to the LX6",
      cxxopts::value<std::string>());

  auto vm = options.parse(argc, argv);

  // Check required options
  if (!vm.count("xclbin") || !vm.count("kernel") || !vm.count("instr")) {
    std::cerr << "Error: Required options missing\n\n";
    std::cerr << "Usage:\n" << options.help() << std::endl;
    return 1;
  }

  std::vector<uint32_t> instr_v =
      test_utils::load_instr_binary(vm["instr"].as<std::string>());
  assert(instr_v.size() > 0);

  // Get a device handle
  unsigned int device_index = 0;
  xrt::device device = xrt::device(device_index);

  // Load the xclbin
  xrt::xclbin xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  std::string Node = vm["kernel"].as<std::string>();
  // Get the kernel from the xclbin
  std::vector<xrt::xclbin::kernel> xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });

  std::string kernel_name = xkernel.get_name();
  assert(strcmp(kernel_name.c_str(), Node.c_str()) == 0);

  device.register_xclbin(xclbin);

  // get a hardware context
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  auto kernel = xrt::kernel(context, kernel_name);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  int total_size = 4096;
  int TRACE_SIZE = 8192;

  auto in_0 = xrt::bo(device, total_size * sizeof(TYPE_IN),
                      XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

  auto out_0 = xrt::bo(device, total_size * sizeof(TYPE_OUT),
                       XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  auto zero_0 = xrt::bo(device, 1 * sizeof(TYPE_IN), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(5));
  auto zero_1 = xrt::bo(device, 1 * sizeof(TYPE_IN), XRT_BO_FLAGS_HOST_ONLY,
                        kernel.group_id(6));

  int tmp_trace_size = (TRACE_SIZE * 4 > 0) ? TRACE_SIZE : 1;
  auto trace_res = xrt::bo(device, tmp_trace_size, XRT_BO_FLAGS_HOST_ONLY,
                           kernel.group_id(7));

  TYPE_IN *in_0_buf = in_0.map<TYPE_IN *>();

 TYPE_OUT *out_0_buf = out_0.map<TYPE_OUT *>();
  memset(out_0_buf, 0, total_size* sizeof(TYPE_OUT) );


  for (uint32_t i = 0; i < total_size; i++) {
    *(in_0_buf + i) = i + 1;
  }



  // Instruction buffer for DMA configuration
  void *buf_instr = bo_instr.map<void *>();
  memcpy(buf_instr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  in_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  char *bufTrace = trace_res.map<char *>();
  if (TRACE_SIZE > 0) {
    memset(bufTrace, 0, TRACE_SIZE);
    trace_res.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), in_0, out_0, zero_0,
                    zero_1, trace_res);
  ert_cmd_state r = run.wait();
  if (r != ERT_CMD_STATE_COMPLETED) {
    std::cout << "Kernel did not complete. Returned status: " << r << "\n";
    return 1;
  }
  out_0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  if (TRACE_SIZE > 0) {
    trace_res.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    test_utils::write_out_trace(((char *)bufTrace), TRACE_SIZE, "trace.txt");
  }

  bool pass = true;
 
  for (auto k = 0; k < 10; k++) {

    if (*(out_0_buf + k) != *(in_0_buf + k)) {
      pass = false;
      std::cout << "&&&&&&&&&&&&&&" << std::endl;
      std::cout << " " << std::hex << k << "  index for  " << std::hex
                << *(out_0_buf + k) << " with expect" <<  *(in_0_buf + k) << std::endl;

    } else {
    }
  }

  if (pass == false) {
    std::cout << "Fail stage 1" << std::endl;
  } else {
    printf("passed first stage input\n");
  }

  return 0;
}
