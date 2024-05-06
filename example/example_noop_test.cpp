#include "transactionFWDump.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

constexpr uint32_t SIZE_4K_HERE = 4 * 1024;
constexpr uint32_t OFFSET_3K_HERE = 3 * 1024;

static inline uint64_t getTileAddrHere(uint8_t c, uint8_t r) {
  return (((uint64_t)r & 0xFFU) << XAIE_ROW_SHIFT) |
         (((uint64_t)c & 0xFFU) << XAIE_COL_SHIFT);
}

const std::vector<uint32_t> prolog{
    0x00000011, 0x01000405, 0x01000100, 0x0B590100, 0x000055FF, 0x00000001,
    0x00000010, 0x314E5A5F, 0x635F5F31, 0x676E696C, 0x39354E5F, 0x6E693131,
    0x5F727473, 0x64726F77, 0x00004573, 0x07BD9630, 0x000055FF,
};

void testAdd256UsingDmaOpNoDoubleBuffering() {
  unsigned int deviceIndex = 0;
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/final.xclbin");

  auto device = xrt::device(deviceIndex);
  auto xclbin = xrt::xclbin(xclbinFile);

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, "MLIR_AIE");

  std::vector<uint32_t> ipuInsts(prolog);
  std::vector<uint32_t> shimInsts{
      100663552,  0,          128,      0,        0,        0,
      2147483648, 0,          0,        33554432, 33554432, 119316,
      0,          100663585,  0,        128,      0,        0,
      0,          2147483648, 0,        0,        33554432, 33554432,
      119300,     2147483649, 50331648, 65792,
  };
  ipuInsts.reserve(ipuInsts.size() + shimInsts.size());
  ipuInsts.insert(ipuInsts.end(), shimInsts.begin(), shimInsts.end());
  xrt::bo npuInstructions(device, ipuInsts.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  npuInstructions.write(ipuInsts.data());
  npuInstructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // group_id matches kernels.json
  const int LEN = 128; /* in bytes */
  auto in = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(2));
  auto tmp = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(3));
  auto out = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(4));

  auto *inPtr = in.map<uint32_t *>();
  auto *tmpPtr = tmp.map<uint32_t *>();
  for (int i = 0; i < LEN; ++i) {
    inPtr[i] = 1;
    tmpPtr[i] = 0;
  }

  in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  tmp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::run run_(kernel);
  run_.set_arg(0, npuInstructions);
  run_.set_arg(1, npuInstructions.size());
  run_.set_arg(2, in);
  run_.set_arg(3, tmp);
  run_.set_arg(4, out);
  run_.start();
  run_.wait2();

  in.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  tmp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  auto *outPtr = out.map<uint32_t *>();
  for (int i = 0; i < LEN; ++i)
    std::cout << outPtr[i] << ", ";
  std::cout << "\n";
}

constexpr std::uint32_t DUMP_REGISTERS_OPCODE = 18;

void dumpRegistersDPU() {
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/final.xclbin");
  auto xclbin = xrt::xclbin(xclbinFile);
  auto xkernel = xclbin.get_kernel("MLIR_AIE");
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  std::vector<uint64_t> regOffsets{
      // corestatus
      0x00032004,
      // Stream_Switch_Master_Config_AIE_Core0
      0x0003F000,
      // module clock control
      0x00060000,
      // Event_Group_Stream_Switch_Enable
      0x00034518,
      // core le
      0x00031150,
      // Core_CR
      0x00031170,
      0x00030C00,
      0x00030C10,

  };

  std::vector<uint32_t> instructionSequence{
      (DUMP_REGISTERS_OPCODE << 24) | static_cast<uint32_t>(regOffsets.size())};
  //  std::vector<uint32_t> instructionSequence;
  int col = 0, row = 2;
  for (const auto &reg : regOffsets) {
    uint64_t absAddr = reg + getTileAddrHere(col, row);
    instructionSequence.push_back(absAddr & 0xFFFFFFFF);
    instructionSequence.push_back((absAddr >> 32) & 0xFFFFFFFF);
  }
  instructionSequence.insert(instructionSequence.begin(), prolog.begin(),
                             prolog.end());

  xrt::bo npuInstructions(device, instructionSequence.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  npuInstructions.write(instructionSequence.data());
  npuInstructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // group_id matches kernels.json
  const int LEN = 128; /* in bytes */
  auto in = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(2));
  auto tmp = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(3));
  auto out = xrt::bo(device, LEN * sizeof(uint32_t), XRT_BO_FLAGS_HOST_ONLY,
                     kernel.group_id(4));

  in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  tmp.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  out.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::run run_(kernel);
  run_.set_arg(0, npuInstructions);
  run_.set_arg(1, npuInstructions.size());
  run_.set_arg(2, in);
  run_.set_arg(3, tmp);
  run_.set_arg(4, out);
  run_.start();
  run_.wait2();

  xrt::bo resultBo = xrt::bo(context.get_device(), SIZE_4K_HERE,
                             XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  if (!resultBo)
    throw std::runtime_error("couldn't get resultBo");

  resultBo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  uint8_t *resultBoMap = resultBo.map<uint8_t *>();
  uint32_t *output = reinterpret_cast<uint32_t *>(resultBoMap + OFFSET_3K_HERE);

  for (int i = 0; i < 1024; ++i)
    std::cout << output[i] << ", ";
  std::cout << "\n";
}

int main() {
  validate();
  dumpRegistersTransaction();

  testAdd256UsingDmaOpNoDoubleBuffering();
  dumpRegistersDPU();
}

// vim: ts=2 sw=2
