//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargets.h"
extern "C" {
#include "cdo-driver/cdo_driver.h"
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaiegbl.h"
}

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

static LogicalResult
generateCDOBinary(const StringRef outputPath,
                  const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

static LogicalResult generateCDOBinariesSeparately(AIERTControl &ctl,
                                                   const StringRef workDirPath,
                                                   DeviceOp &targetOp,
                                                   bool aieSim,
                                                   bool enableCores) {
  auto ps = std::filesystem::path::preferred_separator;

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_elfs.bin");
  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + targetOp.getSymName() + "_aie_cdo_elfs.bin")
              .str(),
          [&ctl, &targetOp, &workDirPath, &aieSim] {
            return ctl.addAieElfs(targetOp, workDirPath, aieSim);
          })))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_init.bin");
  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + targetOp.getSymName() + "_aie_cdo_init.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addInitConfig(targetOp); })))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_enable.bin");
  if (enableCores &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + targetOp.getSymName() + "_aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addCoreEnable(targetOp); })))
    return failure();

  return success();
}

static LogicalResult generateCDOUnified(AIERTControl &ctl,
                                        const StringRef workDirPath,
                                        DeviceOp &targetOp, bool aieSim,
                                        bool enableCores) {
  auto ps = std::filesystem::path::preferred_separator;


  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + targetOp.getSymName() + "_aie_cdo.bin").str(),
      [&ctl, &targetOp, &workDirPath, &aieSim, &enableCores] {
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfig(targetOp)))
          return failure();
        if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnable(targetOp)))
          return failure();
        return success();
      });
}

static LogicalResult
translateToCDODirect(ModuleOp m, llvm::StringRef workDirPath, llvm::StringRef deviceName,
                     byte_ordering endianness, bool emitUnified, bool cdoDebug,
                     bool aieSim, bool xaieDebug, bool enableCores) {

  DeviceOp targetOp = AIE::DeviceOp::getForSymbolInModuleOrError(m, deviceName);
  if (!targetOp) {
    return failure();
  }
  const AIETargetModel &targetModel =
      (const AIETargetModel &)targetOp.getTargetModel();

  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about NPU...
  assert(targetModel.hasProperty(AIETargetModel::IsNPU) &&
         "Only NPU currently supported");

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();
  initializeCDOGenerator(endianness, cdoDebug);

  auto result = [&]() {
    if (emitUnified) {
      return generateCDOUnified(ctl, workDirPath, targetOp, aieSim,
                                enableCores);
    }
    return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim,
                                         enableCores);
  }();
  return result;
}

LogicalResult xilinx::AIE::AIETranslateToCDODirect(
    ModuleOp m, llvm::StringRef workDirPath, llvm::StringRef deviceName,
    bool bigEndian, bool emitUnified, bool cdoDebug, bool aieSim, bool xaieDebug, bool enableCores) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return translateToCDODirect(m, workDirPath, deviceName, endianness, emitUnified, cdoDebug, aieSim, xaieDebug, enableCores);
}
