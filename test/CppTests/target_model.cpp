//===- target_model.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include <stdexcept>

using namespace xilinx;

static void checkControllerTopology(AIE::AIEDevice device, uint32_t perColumn,
                                    uint32_t total) {
  const auto &model = AIE::getTargetModel(device);
  if (model.getNumControllersPerColumn() != perColumn ||
      model.getNumControllers() != total) {
    throw std::runtime_error("Failed microcontroller topology check for " +
                             stringifyAIEDevice(device).str());
  }
}

void test() {

  // AIEDevice::xcvc1902
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'UsesSemaphoreLocks' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcvc1902 property check for "
                             "'UsesMultiDimensionalBDs' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcvc1902 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).columns() != 50) {
    throw std::runtime_error("Failed xcvc1902 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcvc1902).rows() != 9) {
    throw std::runtime_error("Failed xcvc1902 rows");
  }
  checkControllerTopology(AIE::AIEDevice::xcvc1902, 0, 0);

  // AIEDevice::xcve2302
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2302)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2302)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error("Failed xcve2302 property check for "
                             "'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcve2302 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).columns() != 17) {
    throw std::runtime_error("Failed xcve2302 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2302).rows() != 4) {
    throw std::runtime_error("Failed xcve2302 rows");
  }
  checkControllerTopology(AIE::AIEDevice::xcve2302, 0, 0);

  // AIEDevice::xcve2802
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2802)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::xcve2802)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802)
          .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error("Failed xcve2802 property check for "
                             "'IsNPU' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcve2802 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).columns() != 38) {
    throw std::runtime_error("Failed xcve2802 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::xcve2802).rows() != 11) {
    throw std::runtime_error("Failed xcve2802 rows");
  }
  checkControllerTopology(AIE::AIEDevice::xcve2802, 0, 0);

  // AIEDevice::npu_1col, npu_2col, npu_3col, npu_4col
  llvm::DenseMap<AIE::AIEDevice, int> npu1_devs;
  npu1_devs[AIE::AIEDevice::npu1_1col] = 1;
  npu1_devs[AIE::AIEDevice::npu1_2col] = 2;
  npu1_devs[AIE::AIEDevice::npu1_3col] = 3;
  npu1_devs[AIE::AIEDevice::npu1] = 4;
  for (auto &[dev, cols] : npu1_devs) {
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesSemaphoreLocks)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'UsesSemaphoreLocks' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'UsesMultiDimensionalBDs' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(AIE::AIETargetModel::IsNPU)) {
      throw std::runtime_error("Failed npu1_ncol property check for "
                               "'IsNPU' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::IsVirtualized)) {
      throw std::runtime_error(
          "Failed npu1_ncol property check for 'IsVirtualized' returns false");
    }
    if (AIE::getTargetModel(dev).columns() != cols) {
      throw std::runtime_error("Failed npu1_ncol columns");
    }
    if (AIE::getTargetModel(dev).rows() != 6) {
      throw std::runtime_error("Failed npu1_ncol rows");
    }
    checkControllerTopology(dev, 0, 0);
  }

  // AIEDevice::npu2
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'UsesSemaphoreLocks' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed npu2 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (!AIE::getTargetModel(AIE::AIEDevice::npu2)
           .hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'IsNPU' returns false");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2)
          .hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed npu2 property check for 'IsVirtualized' returns true");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2).columns() != 8) {
    throw std::runtime_error("Failed npu2 columns");
  }
  if (AIE::getTargetModel(AIE::AIEDevice::npu2).rows() != 6) {
    throw std::runtime_error("Failed npu2 rows");
  }
  checkControllerTopology(AIE::AIEDevice::npu2, 0, 0);

  // AIEDevice::npu2_1col, npu2_2col, npu2_3col, npu2_4col, npu2_5col,
  // npu2_6col, npu2_7col
  llvm::DenseMap<AIE::AIEDevice, int> npu2_devs;
  npu2_devs[AIE::AIEDevice::npu2_1col] = 1;
  npu2_devs[AIE::AIEDevice::npu2_2col] = 2;
  npu2_devs[AIE::AIEDevice::npu2_3col] = 3;
  npu2_devs[AIE::AIEDevice::npu2_4col] = 4;
  npu2_devs[AIE::AIEDevice::npu2_5col] = 5;
  npu2_devs[AIE::AIEDevice::npu2_6col] = 6;
  npu2_devs[AIE::AIEDevice::npu2_7col] = 7;
  for (auto &[dev, cols] : npu2_devs) {
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesSemaphoreLocks)) {
      throw std::runtime_error("Failed npu2_ncol property check for "
                               "'UsesSemaphoreLocks' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
      throw std::runtime_error("Failed npu2_ncol property check for "
                               "'UsesMultiDimensionalBDs' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(AIE::AIETargetModel::IsNPU)) {
      throw std::runtime_error("Failed npu2_ncol property check for "
                               "'IsNPU' returns false");
    }
    if (!AIE::getTargetModel(dev).hasProperty(
            AIE::AIETargetModel::IsVirtualized)) {
      throw std::runtime_error(
          "Failed npu2_ncol property check for 'IsVirtualized' returns false");
    }
    if (AIE::getTargetModel(dev).columns() != cols) {
      throw std::runtime_error("Failed npu2_ncol columns");
    }
    if (AIE::getTargetModel(dev).rows() != 6) {
      throw std::runtime_error("Failed npu2_ncol rows");
    }
    checkControllerTopology(dev, 0, 0);
  }

  // AIEDevice::xcve3858 (AIE2PS)
  const auto &ve3858 = AIE::getTargetModel(AIE::AIEDevice::xcve3858);
  if (!ve3858.hasProperty(AIE::AIETargetModel::UsesSemaphoreLocks)) {
    throw std::runtime_error("Failed xcve3858 property check for "
                             "'UsesSemaphoreLocks' returns false");
  }
  if (!ve3858.hasProperty(AIE::AIETargetModel::UsesMultiDimensionalBDs)) {
    throw std::runtime_error("Failed xcve3858 property check for "
                             "'UsesMultiDimensionalBDs' returns false");
  }
  if (!ve3858.hasProperty(AIE::AIETargetModel::IsNPU)) {
    throw std::runtime_error(
        "Failed xcve3858 property check for 'IsNPU' returns false");
  }
  checkControllerTopology(AIE::AIEDevice::xcve3858, 1, 36);
  // if (ve3858.getUcGroupId(0) != 0 || ve3858.getUcGroupId(35) != 35)
  //   throw std::runtime_error("Failed xcve3858 uC group-id mapping");

  if (ve3858.hasProperty(AIE::AIETargetModel::IsVirtualized)) {
    throw std::runtime_error(
        "Failed xcve3858 property check for 'IsVirtualized' returns true");
  }
  if (ve3858.columns() != 36) {
    throw std::runtime_error("Failed xcve3858 columns");
  }
  if (ve3858.rows() != 7) {
    throw std::runtime_error("Failed xcve3858 rows");
  }
  if (ve3858.getTargetArch() != AIE::AIEArch::AIE2ps) {
    throw std::runtime_error("Failed xcve3858 getTargetArch");
  }
  // Tile type checks: row 0 is shim, rows 1-2 are memory, and rows 3-6 are
  // cores.
  if (!ve3858.isShimNOCTile(0, 0)) {
    throw std::runtime_error("Failed xcve3858 isShimNOCTile(0,0)");
  }
  if (!ve3858.isMemTile(0, 1) || !ve3858.isMemTile(0, 2)) {
    throw std::runtime_error("Failed xcve3858 memory-tile layout");
  }
  if (!ve3858.isCoreTile(0, 3) || !ve3858.isCoreTile(35, 6)) {
    throw std::runtime_error("Failed xcve3858 core-tile layout");
  }
  // Memory sizes (inherited from AIE2, verified against spec)
  if (ve3858.getLocalMemorySize() != 0x10000) {
    throw std::runtime_error("Failed xcve3858 getLocalMemorySize (expected "
                             "64KB)");
  }
  if (ve3858.getMemTileSize() != 0x80000) {
    throw std::runtime_error("Failed xcve3858 getMemTileSize (expected 512KB)");
  }
  // Lock counts (inherited from AIE2, verified against spec)
  if (ve3858.getNumLocks(0, 0) != 16) {
    throw std::runtime_error("Failed xcve3858 getNumLocks shim");
  }
  if (ve3858.getNumLocks(0, 1) != 64) {
    throw std::runtime_error("Failed xcve3858 getNumLocks memtile");
  }
  if (ve3858.getNumLocks(0, 3) != 16) {
    throw std::runtime_error("Failed xcve3858 getNumLocks core");
  }
  // BD counts (inherited from AIE2, verified against spec)
  if (ve3858.getNumBDs(0, 0) != 16) {
    throw std::runtime_error("Failed xcve3858 getNumBDs shim");
  }
  if (ve3858.getNumBDs(0, 1) != 48) {
    throw std::runtime_error("Failed xcve3858 getNumBDs memtile");
  }
  if (ve3858.getNumBDs(0, 3) != 16) {
    throw std::runtime_error("Failed xcve3858 getNumBDs core");
  }
  // Burst encodings: AIE2PS supports 512B (4 encodings, not 3)
  auto bursts = ve3858.getShimBurstEncodingsAndLengths();
  if (bursts.size() != 4) {
    throw std::runtime_error("Failed xcve3858 burst encoding count "
                             "(expected 4, got " +
                             std::to_string(bursts.size()) + ")");
  }
  if (bursts[3].second != 512) {
    throw std::runtime_error("Failed xcve3858 burst[3] should be 512B");
  }

  // AIE2PS-specific shim DMA register layout.
  if (ve3858.getDmaBdAddress(2, 0, 3, 0, AIE::DMAChannelDir::S2MM) !=
      ((uint64_t{2} << 25) | 0x9090)) {
    throw std::runtime_error("Failed xcve3858 shim DMA BD address");
  }
  if (ve3858.getDmaControlAddress(2, 0, 1, AIE::DMAChannelDir::S2MM) !=
      ((uint32_t{2} << 25) | 0x9308)) {
    throw std::runtime_error("Failed xcve3858 shim S2MM control address");
  }
  if (ve3858.getDmaControlAddress(2, 0, 1, AIE::DMAChannelDir::MM2S) !=
      ((uint32_t{2} << 25) | 0x9318)) {
    throw std::runtime_error("Failed xcve3858 shim MM2S control address");
  }

  auto controllerMaster = ve3858.getStreamSwitchPortIndex(
      0, 0, AIE::WireBundle::Controller32, 0, true);
  auto controllerSlave = ve3858.getStreamSwitchPortIndex(
      0, 0, AIE::WireBundle::Controller32, 0, false);
  if (controllerMaster != 22 || controllerSlave != 23)
    throw std::runtime_error("Failed xcve3858 controller port mapping");
  if (ve3858.getNumDestSwitchboxConnections(
          0, 0, AIE::WireBundle::Controller32) != 1 ||
      ve3858.getNumSourceSwitchboxConnections(
          0, 0, AIE::WireBundle::Controller32) != 1 ||
      ve3858.getNumDestShimMuxConnections(0, 0,
                                          AIE::WireBundle::Controller32) != 0 ||
      ve3858.getNumSourceShimMuxConnections(0, 0,
                                            AIE::WireBundle::Controller32) != 0)
    throw std::runtime_error("Failed xcve3858 controller port connectivity");

  if (ve3858.getStreamSwitchPortIndex(0, 0, AIE::WireBundle::Controller32, 1,
                                      true))
    throw std::runtime_error("Failed xcve3858 controller channel bounds");

  // Cascade size
  if (ve3858.getAccumulatorCascadeSize() != 512) {
    throw std::runtime_error("Failed xcve3858 getAccumulatorCascadeSize");
  }
}

// Program memory is 16 KB on every generation, per aie-rt's XAie_CoreMod
// tables. The same number appears in python/utils/regdb.py's MEMORY_REGIONS --
// keep them in step.
void testProgramMemory() {
  for (auto dev :
       {AIE::AIEDevice::xcvc1902, AIE::AIEDevice::npu1, AIE::AIEDevice::npu2}) {
    const auto &tm = AIE::getTargetModel(dev);
    if (tm.getProgramMemorySize() != 0x4000)
      throw std::runtime_error("Failed program memory size");
  }
}

int main() {
  test();
  testProgramMemory();
  return 0;
}
