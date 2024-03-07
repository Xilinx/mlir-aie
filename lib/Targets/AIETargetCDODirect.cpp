//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Targets/AIERTX.h"
#include "aie/Targets/AIETargets.h"
extern "C" {
#include "cdo-driver/cdo_driver.h"
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstddef> // size_t
#include <cstdint> // uint
#include <cstdlib> // calloc
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <string>

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_dma.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_locks.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_ss.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

auto ps = std::filesystem::path::preferred_separator;

namespace xilinx::AIE {

LogicalResult addErrorHandlingToCDO(XAie_DevInst &devInst) {
  TRY_XAIE_API_LOGICAL_RESULT(XAie_ErrorHandlingInit, &devInst);
  return success();
}

LogicalResult addAieElfToCDO(XAie_DevInst &devInst, uint8_t col, uint8_t row,
                             const StringRef elfPath, bool aieSim) {
  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  TRY_XAIE_API_LOGICAL_RESULT(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                              elfPath.str().c_str(), /*loadSym*/ aieSim);
  return success();
}

LogicalResult addAieElfsToCDO(XAie_DevInst &devInst, DeviceOp &targetOp,
                              const StringRef workDirPath, bool aieSim) {
  for (auto tileOp : targetOp.getOps<TileOp>())
    if (tileOp.isShimNOCorPLTile()) {
      // Resets no needed with V2 kernel driver
    } else {
      int col = tileOp.colIndex();
      int row = tileOp.rowIndex();
      if (auto coreOp = tileOp.getCoreOp()) {
        std::string fileName;
        if (auto fileAttr = coreOp.getElfFile())
          fileName = fileAttr->str();
        else
          fileName = (llvm::Twine("core_") + std::to_string(col) + "_" +
                      std::to_string(row) + ".elf")
                         .str();
        if (failed(addAieElfToCDO(
                devInst, col, row,
                (llvm::Twine(workDirPath) + std::string(1, ps) + fileName)
                    .str(),
                aieSim)))
          return failure();
      }
    }
  return success();
}

LogicalResult addInitConfigToCDO(AIERTXControl &ctl, DeviceOp &targetOp) {
  if (failed(ctl.initLocks(targetOp)))
    return failure();

  auto memOps = llvm::to_vector_of<TileElement>(targetOp.getOps<MemOp>());
  llvm::append_range(memOps, targetOp.getOps<MemTileDMAOp>());
  llvm::append_range(memOps, targetOp.getOps<ShimDMAOp>());
  for (TileElement memOp : memOps) {
    int col = memOp.getTileID().col;
    int row = memOp.getTileID().row;
    XAie_LocType tileLoc = XAie_TileLoc(col, row);

    // handle DMA ops separately
    auto dmaOps = llvm::to_vector_of<DMAOp>(
        memOp.getOperation()->getRegion(0).getOps<DMAOp>());
    if (!dmaOps.empty()) {
      for (auto dmaOp : dmaOps)
        for (auto &bdRegion : dmaOp.getBds()) {
          Block &block = bdRegion.getBlocks().front();
          if (failed(ctl.configureLocksAndBd(block, tileLoc)))
            return failure();
        }
    } else {
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        if (block.getOps<DMABDOp>().empty())
          continue;
        if (failed(ctl.configureLocksAndBd(block, tileLoc)))
          return failure();
      }
    }

    if (!dmaOps.empty())
      for (auto dmaOp : dmaOps) {
        auto &block = dmaOp.getBds().front().getBlocks().front();
        DMABDOp bd = *block.getOps<DMABDOp>().begin();
        if (failed(ctl.pushToBdQueueAndEnable(
                *dmaOp.getOperation(), tileLoc, dmaOp.getChannelIndex(),
                dmaOp.getChannelDir(), bd.getBdId().value(),
                dmaOp.getRepeatCount())))
          return failure();
      }
    else
      for (Block &block : memOp.getOperation()->getRegion(0)) {
        for (auto op : block.getOps<DMAStartOp>()) {
          DMABDOp bd = *op.getDest()->getOps<DMABDOp>().begin();
          int chNum = op.getChannelIndex();
          auto channelDir = op.getChannelDir();
          if (failed(ctl.pushToBdQueueAndEnable(
                  *bd.getOperation(), tileLoc, chNum, channelDir,
                  bd.getBdId().value(), op.getRepeatCount())))
            return failure();
        }
      }
  }

  if (failed(ctl.configureSwitches(targetOp)))
    return failure();

  return success();
}

void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

LogicalResult generateCDOBinary(const StringRef outputPath,
                                const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

LogicalResult generateCDOBinariesSeparately(AIERTXControl &ctl,
                                            const StringRef workDirPath,
                                            DeviceOp &targetOp, bool aieSim,
                                            bool enableCores) {
  if (failed(generateCDOBinary((llvm::Twine(workDirPath) + std::string(1, ps) +
                                "aie_cdo_error_handling.bin")
                                   .str(),
                               std::bind(&addErrorHandlingToCDO, ctl.devInst))))
    return failure();

  if (!targetOp.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_elfs.bin")
              .str(),
          [&ctl, &targetOp, &workDirPath, &aieSim] {
            return addAieElfsToCDO(ctl.devInst, targetOp, workDirPath, aieSim);
          })))
    return failure();

  if (failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_init.bin")
              .str(),
          [&ctl, &targetOp] { return addInitConfigToCDO(ctl, targetOp); })))
    return failure();

  if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.enableCoresInDevice(targetOp); })))
    return failure();

  return success();
}

LogicalResult generateCDOUnified(AIERTXControl &ctl, const StringRef workDirPath,
                                 DeviceOp &targetOp, bool aieSim,
                                 bool enableCores) {
  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + "aie_cdo.bin").str(),
      [&ctl, &targetOp, &workDirPath, &aieSim, &enableCores] {
        if (failed(addErrorHandlingToCDO(ctl.devInst)))
          return failure();
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(addAieElfsToCDO(ctl.devInst, targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(addInitConfigToCDO(ctl, targetOp)))
          return failure();
        if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.enableCoresInDevice(targetOp)))
          return failure();
        return success();
      });
}

LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      byte_ordering endianness,
                                      bool emitUnified, bool cdoDebug,
                                      bool aieSim, bool xaieDebug,
                                      size_t partitionStartCol,
                                      bool enableCores) {
  auto devOps = m.getOps<DeviceOp>();
  assert(llvm::range_size(devOps) == 1 &&
         "only exactly 1 device op supported.");
  DeviceOp targetOp = *devOps.begin();
  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about IPU...
  assert(targetOp.getDevice() == AIEDevice::ipu &&
         "Only IPU currently supported");
  int maxCol = 0, minCol = 0;
  for (auto tileOp : targetOp.getOps<TileOp>()) {
    minCol = std::min(tileOp.getCol(), minCol);
    maxCol = std::max(tileOp.getCol(), maxCol);
  }
  size_t partitionNumCols = maxCol - minCol + 1;
  AIERTXControl ctl(partitionStartCol, partitionNumCols,
                 targetOp.getTargetModel());
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();
  initializeCDOGenerator(endianness, cdoDebug);
  if (emitUnified)
    return generateCDOUnified(ctl, workDirPath, targetOp, aieSim, enableCores);
  return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim,
                                       enableCores);
}
// Not sure why but defining this with xilinx::AIE will create a duplicate
// symbol in libAIETargets.a that then doesn't actually match the header?
LogicalResult AIETranslateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                                      bool bigEndian, bool emitUnified,
                                      bool cdoDebug, bool aieSim,
                                      bool xaieDebug, size_t partitionStartCol,
                                      bool enableCores) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return AIETranslateToCDODirect(m, workDirPath, endianness, emitUnified,
                                 cdoDebug, aieSim, xaieDebug, partitionStartCol,
                                 enableCores);
}
} // namespace xilinx::AIE
