//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AIETargetShared.h"
#include "aie/Targets/AIETargets.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"

extern "C" {
#include "xaiengine/xaie_core.h"
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaie_plif.h"
#include "xaiengine/xaie_ss.h"
}

#define HW_GEN XAIE_DEV_GEN_AIEML
#define XAIE_NUM_ROWS 6
#define XAIE_NUM_COLS 5
#define XAIE_BASE_ADDR 0x40000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_MEM_TILE_ROW_START 1
#define XAIE_MEM_TILE_NUM_ROWS 1
#define XAIE_AIE_TILE_ROW_START 2
#define XAIE_AIE_TILE_NUM_ROWS 4
#define FOR_WRITE 0
#define FOR_READ 1
#define XAIE_PARTITION_BASE_ADDR 0x0

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

XAie_InstDeclare(DevInst, &ConfigPtr); // Declare global device instance

class InitializeAIEControl {
public:
  InitializeAIEControl() {
    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT,
                     XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_MEM_TILE_ROW_START,
                     XAIE_MEM_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    XAie_SetupPartitionConfig(&DevInst, XAIE_PARTITION_BASE_ADDR,
                              /*PartStartCol=*/1, /*PartNumCols=*/1);

    XAie_CfgInitialize(&DevInst, &ConfigPtr);

    XAie_SetIOBackend(
        &DevInst,
        XAIE_IO_BACKEND_CDO); // Set aiengine driver library to run for CDO Mode
    XAie_UpdateNpiAddr(&DevInst, /*NpiAddr=*/0x0);
  }
} initAIEControl;

namespace xilinx::AIE {} // namespace xilinx::AIE

mlir::LogicalResult xilinx::AIE::AIETranslateToCDODirect(ModuleOp m,
                                                         raw_ostream &output) {
  return failure();
}
