//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIETargets.h"
#include "aie/Targets/cdo_driver.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#include <filesystem>
#include <map>
#include <stdint.h>
#include <string>

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaiegbl.h"
#include "xaiengine/xaiegbl_defs.h"
}

#define DEBUG_TYPE "aie-generate-cdo-direct"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

// So that we can use the pattern if(auto r = ...) { // r is nonzero }
static_assert(XAIE_OK == 0);

#define AIERC_STR(x) x, #x
static const std::map<AieRC, std::string> AIERCTOSTR = {
    {AIERC_STR(XAIE_OK)},
    {AIERC_STR(XAIE_ERR)},
    {AIERC_STR(XAIE_INVALID_DEVICE)},
    {AIERC_STR(XAIE_INVALID_RANGE)},
    {AIERC_STR(XAIE_INVALID_ARGS)},
    {AIERC_STR(XAIE_INVALID_TILE)},
    {AIERC_STR(XAIE_ERR_STREAM_PORT)},
    {AIERC_STR(XAIE_INVALID_DMA_TILE)},
    {AIERC_STR(XAIE_INVALID_BD_NUM)},
    {AIERC_STR(XAIE_ERR_OUTOFBOUND)},
    {AIERC_STR(XAIE_INVALID_DATA_MEM_ADDR)},
    {AIERC_STR(XAIE_INVALID_ELF)},
    {AIERC_STR(XAIE_CORE_STATUS_TIMEOUT)},
    {AIERC_STR(XAIE_INVALID_CHANNEL_NUM)},
    {AIERC_STR(XAIE_INVALID_LOCK)},
    {AIERC_STR(XAIE_INVALID_DMA_DIRECTION)},
    {AIERC_STR(XAIE_INVALID_PLIF_WIDTH)},
    {AIERC_STR(XAIE_INVALID_LOCK_ID)},
    {AIERC_STR(XAIE_INVALID_LOCK_VALUE)},
    {AIERC_STR(XAIE_LOCK_RESULT_FAILED)},
    {AIERC_STR(XAIE_INVALID_DMA_DESC)},
    {AIERC_STR(XAIE_INVALID_ADDRESS)},
    {AIERC_STR(XAIE_FEATURE_NOT_SUPPORTED)},
    {AIERC_STR(XAIE_INVALID_BURST_LENGTH)},
    {AIERC_STR(XAIE_INVALID_BACKEND)},
    {AIERC_STR(XAIE_INSUFFICIENT_BUFFER_SIZE)},
    {AIERC_STR(XAIE_ERR_MAX)}};
#undef AIERC_STR

#define TRY_XAIE_API(API, ...)                                                 \
  if (auto r = API(__VA_ARGS__))                                               \
  report_fatal_error(llvm::Twine(#API " failed with ") + AIERCTOSTR.at(r))

namespace xilinx::AIE {

struct AIEControl {
  XAie_Config configPtr;
  XAie_DevInst devInst;
  std::string workDirPath;
  byte_ordering endianness;

  AIEControl(const std::string &workDirPath, byte_ordering endianness,
             uint8_t hwGen = XAIE_DEV_GEN_AIEML,
             uint64_t xaieBaseAddr = 0x40000000, uint8_t xaieColShift = 25,
             uint8_t xaieRowShift = 20, uint8_t xaieNumCols = 5,
             uint8_t xaieNumRows = 6, uint8_t xaieShimRow = 0,
             uint8_t xaieMemTileRowStart = 1, uint8_t xaieMemTileNumRows = 1,
             uint8_t xaieAieTileRowStart = 2, uint8_t xaieAieTileNumRows = 4,
             uint64_t xaiePartitionBaseAddr = 0x0, uint64_t npiAddr = 0x0)
      : workDirPath(workDirPath), endianness(endianness),
        configPtr({
            .AieGen = hwGen,
            .BaseAddr = xaieBaseAddr,
            .ColShift = xaieColShift,
            .RowShift = xaieRowShift,
            .NumRows = xaieNumRows,
            .NumCols = xaieNumCols,
            .ShimRowNum = xaieShimRow,
            .MemTileRowStart = xaieMemTileRowStart,
            .MemTileNumRows = xaieMemTileNumRows,
            .AieTileRowStart = xaieAieTileRowStart,
            .AieTileNumRows = xaieAieTileNumRows,
            .PartProp = {0},
        }) {
    // Quoting: The instance of a device must be always declared using this
    //		macro. In future, the same macro will be expanded to allocate
    //		more memory from the user application for resource management.
    XAie_InstDeclare(_devInst, &configPtr); // Declare global device instance
    devInst = _devInst;
    // TODO(max): what is the "partition"?
    TRY_XAIE_API(XAie_SetupPartitionConfig, &devInst, xaiePartitionBaseAddr,
                 /*PartStartCol=*/1,
                 /*PartNumCols=*/1);
    TRY_XAIE_API(XAie_CfgInitialize, &devInst, &configPtr);
    TRY_XAIE_API(XAie_SetIOBackend, &devInst, XAIE_IO_BACKEND_CDO);
    TRY_XAIE_API(XAie_UpdateNpiAddr, &devInst, npiAddr);
  }

  std::string prependWorkDir(const std::string &path) {
    return workDirPath + std::filesystem::path::preferred_separator + path;
  }

  void initializeCDOGenerator() {
#ifndef NDEBUG
    EnAXIdebug(); // Enables AXI-MM prints for configs being added in CDO,
#endif
    setEndianness(endianness);
  };

  // void addInitConfigToCDO(const std::string &workDirPath) {
  //   ppgraphInit(workDirPath);
  // }
  //
  // void addCoreEnableToCDO() { ppgraphCoreEnable(); }
  //
  // void addErrorHandlingToCDO() { enableErrorHandling(); }

  // loadSym: Load symbols from .map file. This argument is not used when
  // __AIESIM__ is not defined.
  void addAieElfToCDO(uint8_t col, uint8_t row, const std::string &elfPath,
                      bool loadSym) {
    TRY_XAIE_API(XAie_LoadElf, &devInst, XAie_TileLoc(col, row),
                 elfPath.c_str(), loadSym);
  }

  void generateCDOBinariesSeparately() {
    startCDOFileStream(prependWorkDir("aie_cdo_error_handling.bin").c_str());
    FileHeader();
    TRY_XAIE_API(XAie_ErrorHandlingInit, &devInst);
    configureHeader();
    endCurrentCDOFileStream();

    startCDOFileStream(prependWorkDir("aie_cdo_elfs.bin").c_str());
    FileHeader();
    // addAieElfsToCDO(workDirPath);
    configureHeader();
    endCurrentCDOFileStream();

    startCDOFileStream(prependWorkDir("aie_cdo_init.bin").c_str());
    FileHeader();
    // addInitConfigToCDO(workDirPath);
    configureHeader();
    endCurrentCDOFileStream();

    startCDOFileStream(prependWorkDir("aie_cdo_enable.bin").c_str());
    FileHeader();
    // addCoreEnableToCDO();
    configureHeader();
    endCurrentCDOFileStream();
  }
};

} // namespace xilinx::AIE

LogicalResult AIE::AIETranslateToCDODirect(ModuleOp m,
                                           const std::string &workDirPath,
                                           byte_ordering endianness) {
  AIEControl ctl(workDirPath, endianness);
  return failure();
}
