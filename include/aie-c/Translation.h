//===- Translation.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_C_TRANSLATION_H
#define AIE_C_TRANSLATION_H

#include "aie-c/TargetModel.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Wrap.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirStringRef aieTranslateAIEVecToCpp(MlirOperation op,
                                                         bool aie2);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateNpuToBinary(MlirOperation op,
                                                         MlirStringRef name);
MLIR_CAPI_EXPORTED MlirStringRef
aieTranslateControlPacketsToUI32Vec(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToXAIEV2(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToHSA(MlirOperation op);
MLIR_CAPI_EXPORTED MlirStringRef aieTranslateToBCF(MlirOperation op, int col,
                                                   int row);
MLIR_CAPI_EXPORTED MlirStringRef aieLLVMLink(MlirStringRef *modules,
                                             int nModules);
MLIR_CAPI_EXPORTED MlirLogicalResult
aieTranslateToCDODirect(MlirOperation moduleOp, MlirStringRef workDirPath,
                        bool bigEndian, bool emitUnified, bool cdoDebug,
                        bool aieSim, bool xaieDebug, bool enableCores);
MLIR_CAPI_EXPORTED MlirOperation aieTranslateBinaryToTxn(MlirContext ctx,
                                                         MlirStringRef binary);

MLIR_CAPI_EXPORTED MlirOperation aieRuntimeSequenceCreate(MlirStringRef name, int dev);
MLIR_CAPI_EXPORTED MlirStringRef aieRuntimeSequenceAddNpuDmaMempy(
  MlirOperation runtime_sequence, uint32_t id, uint32_t direction, uint32_t channel, uint32_t column, uint64_t addr,
    uint32_t offsets[4], uint32_t sizes[4], uint32_t strides[4]);
MLIR_CAPI_EXPORTED MlirLogicalResult
aieRuntimeSequenceAddNpuDmaWait(MlirOperation runtime_sequence, MlirStringRef symbol);

struct AieRtControl {
  void *ptr;
};
using AieRtControl = struct AieRtControl;

MLIR_CAPI_EXPORTED AieRtControl getAieRtControl(AieTargetModel tm);
MLIR_CAPI_EXPORTED void freeAieRtControl(AieRtControl aieCtl);
MLIR_CAPI_EXPORTED void aieRtStartTransaction(AieRtControl aieCtl);
MLIR_CAPI_EXPORTED void aieRtDmaUpdateBdAddr(AieRtControl aieCtl, int col,
                                             int row, size_t addr, size_t bdId);
MLIR_CAPI_EXPORTED void aieRtExportSerializedTransaction(AieRtControl aieCtl);

#ifdef __cplusplus
}
#endif

#endif // AIE_C_TRANSLATION_H
