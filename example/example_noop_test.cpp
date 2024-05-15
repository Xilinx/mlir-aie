#include "hwctx.h"
#include "transactionFWDump.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

static inline uint64_t getTileAddrHere(uint8_t c, uint8_t r) {
  return (((uint64_t)r & 0xFFU) << XAIE_ROW_SHIFT) |
         (((uint64_t)c & 0xFFU) << XAIE_COL_SHIFT);
}

void testRepeatCount() {
  unsigned int deviceIndex = 0;
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/final.xclbin");

  auto device = xrt::device(deviceIndex);
  auto xclbin = xrt::xclbin(xclbinFile);

  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  xrt::kernel kernel(context, "MLIR_AIE");

  std::vector<uint32_t> ipuInsts{
      17,         16778245,   16777472,   190382336,  22015,     1,
      16,         827218527,  1667194673, 1735289196, 959794783, 1852387633,
      1601336435, 1685221239, 17779,      129865264,  22015,     100663552,
      0,          1,          0,          0,          0,         2147483648,
      0,          0,          33554432,   33554432,   119300,    2147483648,
      50331648,   65792,      100729104,  0,          1,         0,
      0,          0,          2147483648, 0,          0,         33554432,
      33619968,   119300,     2147483648, 50397184,   65792,     100794656,
      0,          1,          0,          0,          0,         2147483648,
      0,          0,          33554432,   33685504,   119300,    2147483648,
      50462720,   65792,      100860208,  0,          1,         0,
      0,          0,          2147483648, 0,          0,         33554432,
      33751040,   119300,     2147483648, 50528256,   65792};
  xrt::bo npuInstructions(device, ipuInsts.size() * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  npuInstructions.write(ipuInsts.data());
  npuInstructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // group_id matches kernels.json
  auto c0 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(2));
  auto c1 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(3));
  auto c2 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(4));
  auto c3 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                    kernel.group_id(5));

  c0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  c1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  c2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  c3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  xrt::run run_(kernel);
  run_.set_arg(0, npuInstructions);
  run_.set_arg(1, npuInstructions.size());
  run_.set_arg(2, c0);
  run_.set_arg(3, c1);
  run_.set_arg(4, c2);
  run_.set_arg(5, c3);
  run_.start();
  run_.wait2();

  c0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  c3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  std::cout << c0.map<float *>()[0] << ", ";
  std::cout << c1.map<float *>()[0] << ", ";
  std::cout << c2.map<float *>()[0] << ", ";
  std::cout << c3.map<float *>()[0] << ", ";
  std::cout << "\n";
}

constexpr std::uint32_t DUMP_REGISTERS_OPCODE = 18;

void dumpRegistersDPU() {
  unsigned int deviceIndex = 0;
  auto device = xrt::device(deviceIndex);
  std::string xclbinFile(
      "/home/mlevental/dev_projects/mlir-aie/example/final.xclbin");
  {
    auto xclbin = xrt::xclbin(xclbinFile);
    auto xkernel = xclbin.get_kernel("MLIR_AIE");
    auto kernelName = xkernel.get_name();
    device.register_xclbin(xclbin);
    xrt::hw_context context(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(context, kernelName);

    std::vector<uint32_t> ipuInsts{
        17,         16778245,   16777472,   190382336,  22015,     1,
        16,         827218527,  1667194673, 1735289196, 959794783, 1852387633,
        1601336435, 1685221239, 17779,      129865264,  22015,     100663552,
        0,          1,          0,          0,          0,         2147483648,
        0,          0,          33554432,   33554432,   119300,    2147483648,
        50331648,   65792,      100729104,  0,          1,         0,
        0,          0,          2147483648, 0,          0,         33554432,
        33619968,   119300,     2147483648, 50397184,   65792,     100794656,
        0,          1,          0,          0,          0,         2147483648,
        0,          0,          33554432,   33685504,   119300,    2147483648,
        50462720,   65792,      100860208,  0,          1,         0,
        0,          0,          2147483648, 0,          0,         33554432,
        33751040,   119300,     2147483648, 50528256,   65792};

    std::vector<uint64_t> regOffsets{// corestatus
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
                                     0x00031170, 0x00030C00, 0x00030C10,
                                     // lower timer
                                     0x340F8};

    std::vector<uint32_t> dumpRegInsts;
    int count = 0;
    for (int col = 0; col < 4; ++col) {
      int row = 2;
      for (const auto &reg : regOffsets) {
        uint64_t absAddr = reg + getTileAddrHere(col, row);
        dumpRegInsts.push_back(absAddr & 0xFFFFFFFF);
        dumpRegInsts.push_back((absAddr >> 32) & 0xFFFFFFFF);
        count += 1;
      }
    }
    ipuInsts.push_back((DUMP_REGISTERS_OPCODE << 24) | count);
    ipuInsts.insert(ipuInsts.end(), dumpRegInsts.begin(), dumpRegInsts.end());

    xrt::bo npuInstructions(device, ipuInsts.size() * sizeof(uint32_t),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
    npuInstructions.write(ipuInsts.data());
    npuInstructions.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // group_id matches kernels.json
    auto c0 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(2));
    auto c1 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(3));
    auto c2 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(4));
    auto c3 = xrt::bo(device, sizeof(float), XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

    c0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    c3.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::run run_(kernel);
    run_.set_arg(0, npuInstructions);
    run_.set_arg(1, npuInstructions.size());
    run_.set_arg(2, c0);
    run_.set_arg(3, c1);
    run_.set_arg(4, c2);
    run_.set_arg(5, c3);
    run_.start();
    run_.wait2();

    c0.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    c3.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    std::cout << c0.map<float *>()[0] << ", ";
    std::cout << c1.map<float *>()[0] << ", ";
    std::cout << c2.map<float *>()[0] << ", ";
    std::cout << c3.map<float *>()[0] << ", ";
    std::cout << "\n";
  }

  auto boflags = XRT_BO_FLAGS_CACHEABLE;
  auto ext_boflags = XRT_BO_USE_DEBUG << 4;
  auto size = static_cast<size_t>(4096);

  {
    hw_ctx hwctx(device.get_handle().get(), xclbinFile);
    auto bo = hwctx.get()->alloc_bo(size, get_bo_flags(boflags, ext_boflags));
    auto dbg_p = static_cast<uint32_t *>(
        bo->map(xrt_core::buffer_handle::map_type::read));
    bo.get()->sync(buffer_handle::direction::device2host, 4096, 0);
    for (int i = 0; i < 4096; ++i)
      if (dbg_p[i] != 0)
        std::cout << dbg_p[i] << ", ";
    std::cout << "\n";
  }
}

int main() {
  //  testRepeatCount();
  //  dumpRegistersDPU();
  dumpRegistersTransaction();
}
