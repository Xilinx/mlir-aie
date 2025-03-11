//===- Translation.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "aie-c/Translation.h"

#include "aie/Conversion/AIEToConfiguration/AIEToConfiguration.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargets.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace llvm;
using namespace mlir;
using namespace xilinx::AIE;

MlirStringRef aieTranslateAIEVecToCpp(MlirOperation moduleOp, bool aie2) {
  std::string cpp;
  llvm::raw_string_ostream os(cpp);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(xilinx::aievec::translateAIEVecToCpp(mod, aie2, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(cpp.size()));
  cpp.copy(cStr, cpp.size());
  return mlirStringRefCreate(cStr, cpp.size());
};

MlirStringRef aieTranslateModuleToLLVMIR(MlirOperation moduleOp) {
  std::string llvmir;
  llvm::raw_string_ostream os(llvmir);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(mod, llvmContext);
  if (!llvmModule)
    return mlirStringRefCreate(nullptr, 0);
  llvmModule->print(os, nullptr);
  char *cStr = static_cast<char *>(malloc(llvmir.size()));
  llvmir.copy(cStr, llvmir.size());
  return mlirStringRefCreate(cStr, llvmir.size());
}

MlirLogicalResult aieTranslateToCDODirect(MlirOperation moduleOp,
                                          MlirStringRef workDirPath,
                                          bool bigEndian, bool emitUnified,
                                          bool cdoDebug, bool aieSim,
                                          bool xaieDebug, bool enableCores) {
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  auto status = AIETranslateToCDODirect(
      mod, llvm::StringRef(workDirPath.data, workDirPath.length), bigEndian,
      emitUnified, cdoDebug, aieSim, xaieDebug, enableCores);
  std::vector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(mod.getContext(), [&](Diagnostic &d) {
    llvm::raw_string_ostream(diagnostics.emplace_back())
        << d.getLocation() << ": " << d;
  });
  if (failed(status))
    for (const auto &diagnostic : diagnostics)
      std::cerr << diagnostic << "\n";

  return wrap(status);
}

MlirOperation aieTranslateBinaryToTxn(MlirContext ctx, MlirStringRef binary) {
  std::vector<uint8_t> binaryData(binary.data, binary.data + binary.length);
  auto mod = convertTransactionBinaryToMLIR(unwrap(ctx), binaryData);
  if (!mod)
    return wrap(ModuleOp().getOperation());
  return wrap(mod->getOperation());
}

MlirStringRef aieTranslateNpuToBinary(MlirOperation moduleOp,
                                      MlirStringRef sequenceName) {
  std::string npu;
  llvm::raw_string_ostream os(npu);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  llvm::StringRef name(sequenceName.data, sequenceName.length);
  if (failed(AIETranslateNpuToBinary(mod, os, name)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(npu.size()));
  npu.copy(cStr, npu.size());
  return mlirStringRefCreate(cStr, npu.size());
}

MlirStringRef aieTranslateControlPacketsToUI32Vec(MlirOperation moduleOp) {
  std::string npu;
  llvm::raw_string_ostream os(npu);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateControlPacketsToUI32Vec(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(npu.size()));
  npu.copy(cStr, npu.size());
  return mlirStringRefCreate(cStr, npu.size());
}

MlirStringRef aieTranslateToXAIEV2(MlirOperation moduleOp) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToXAIEV2(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToHSA(MlirOperation moduleOp) {
  std::string xaie;
  llvm::raw_string_ostream os(xaie);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToHSA(mod, os)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(xaie.size()));
  xaie.copy(cStr, xaie.size());
  return mlirStringRefCreate(cStr, xaie.size());
}

MlirStringRef aieTranslateToBCF(MlirOperation moduleOp, int col, int row) {
  std::string bcf;
  llvm::raw_string_ostream os(bcf);
  ModuleOp mod = llvm::cast<ModuleOp>(unwrap(moduleOp));
  if (failed(AIETranslateToBCF(mod, os, col, row)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(bcf.size()));
  bcf.copy(cStr, bcf.size());
  return mlirStringRefCreate(cStr, bcf.size());
}

MlirStringRef aieLLVMLink(MlirStringRef *modules, int nModules) {
  std::string ll;
  llvm::raw_string_ostream os(ll);
  std::vector<std::string> files;
  files.reserve(nModules);
  for (int i = 0; i < nModules; ++i)
    files.emplace_back(modules[i].data, modules[i].length);
  if (failed(AIELLVMLink(os, files)))
    return mlirStringRefCreate(nullptr, 0);
  char *cStr = static_cast<char *>(malloc(ll.size()));
  ll.copy(cStr, ll.size());
  return mlirStringRefCreate(cStr, ll.size());
}

MlirOperation aieRuntimeSequenceCreate(MlirStringRef name, int dev) {

  std::string seqName(name.data, name.length);

  // Create an MLIR context using the registry
  auto *context = new MLIRContext;
  context->allowUnregisteredDialects();
  context->loadDialect<memref::MemRefDialect>();
  context->loadDialect<xilinx::AIE::AIEDialect>();
  context->loadDialect<xilinx::AIEX::AIEXDialect>();

  // create a new ModuleOp and set the insertion point
  auto loc = mlir::UnknownLoc::get(context);
  OpBuilder builder(context);

  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto device = builder.create<DeviceOp>(loc, AIEDevice(dev));

  device.getRegion().emplaceBlock();
  DeviceOp::ensureTerminator(device.getBodyRegion(), builder, loc);
  builder.setInsertionPointToStart(device.getBody());

  StringAttr seq_sym_name = builder.getStringAttr(seqName);
  auto seq = builder.create<xilinx::AIEX::RuntimeSequenceOp>(loc, seq_sym_name);
  seq.getBody().push_back(new Block);

  return wrap(module.getOperation());
}

MlirStringRef aieRuntimeSequenceAddNpuDmaMempy(
    MlirOperation m, uint32_t id, uint32_t direction, uint32_t channel,
    uint32_t column, uint64_t addr, uint32_t offsets[4], uint32_t sizes[4],
    uint32_t strides[4]) {

  auto module = cast<ModuleOp>(unwrap(m));

  // get the runtime sequence from the module
  DeviceOp deviceOp = *(module.getOps<DeviceOp>().begin());
  xilinx::AIEX::RuntimeSequenceOp seq =
      *(deviceOp.getOps<xilinx::AIEX::RuntimeSequenceOp>().begin());

  auto loc = seq->getLoc();
  auto ctx = seq->getContext();
  OpBuilder builder(module);
  builder.setInsertionPoint(seq);

  auto memrefType =
      MemRefType::get({1}, mlir::IntegerType::get(ctx, 32), nullptr, 0);
  auto arg = seq.getBody().addArgument(memrefType, loc);
  const std::vector<int64_t> staticOffsets = {offsets[0], offsets[1],
                                              offsets[2], offsets[3]};
  const std::vector<int64_t> staticSizes = {sizes[0], sizes[1], sizes[2],
                                            sizes[3]};
  const std::vector<int64_t> staticStrides = {strides[0], strides[1],
                                              strides[2], strides[3]};
  std::string metadata = "memcpy" + std::to_string(id);

  if (!deviceOp.lookupSymbol(metadata))
    builder.create<memref::GlobalOp>(builder.getUnknownLoc(), metadata,
                                    builder.getStringAttr("public"), memrefType,
                                    nullptr, false, nullptr);
  builder.create<ShimDMAAllocationOp>(loc, metadata, DMAChannelDir(direction),
                                      channel, column);

  builder.setInsertionPointToEnd(&seq.getBody().front());
  builder.create<xilinx::AIEX::NpuDmaMemcpyNdOp>(
      loc, arg, SmallVector<mlir::Value>{}, SmallVector<mlir::Value>{},
      SmallVector<mlir::Value>{}, mlir::ArrayRef(staticOffsets),
      mlir::ArrayRef(staticSizes), mlir::ArrayRef(staticStrides), nullptr,
      metadata, id);

  char *cStr = static_cast<char *>(malloc(metadata.size()));
  metadata.copy(cStr, metadata.size());
  return mlirStringRefCreate(cStr, metadata.size());
}

MlirLogicalResult aieRuntimeSequenceAddNpuDmaWait(MlirOperation m,
                                                  MlirStringRef symbol) {

  auto module = cast<ModuleOp>(unwrap(m));

  // get the runtime sequence from the module
  DeviceOp deviceOp = *(module.getOps<DeviceOp>().begin());
  xilinx::AIEX::RuntimeSequenceOp seq =
      *(deviceOp.getOps<xilinx::AIEX::RuntimeSequenceOp>().begin());

  auto loc = seq->getLoc();
  OpBuilder builder(module);
  builder.setInsertionPointToEnd(&seq.getBody().front());
  builder.create<xilinx::AIEX::NpuDmaWaitOp>(
      loc, StringRef{symbol.data, symbol.length});
  return {1};
}

DEFINE_C_API_PTR_METHODS(AieRtControl, xilinx::AIE::AIERTControl)

AieRtControl getAieRtControl(AieTargetModel tm) {
  // unwrap the target model
  const BaseNPUTargetModel &targetModel =
      *reinterpret_cast<const BaseNPUTargetModel *>(tm.d);
  AIERTControl *ctl = new AIERTControl(targetModel);
  return wrap(ctl);
}

void freeAieRtControl(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  delete ctl;
}

void aieRtDmaUpdateBdAddr(AieRtControl aieCtl, int col, int row, size_t addr,
                          size_t bdId) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->dmaUpdateBdAddr(col, row, addr, bdId);
}

void aieRtStartTransaction(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->startTransaction();
}

void aieRtExportSerializedTransaction(AieRtControl aieCtl) {
  AIERTControl *ctl = unwrap(aieCtl);
  ctl->exportSerializedTransaction();
}