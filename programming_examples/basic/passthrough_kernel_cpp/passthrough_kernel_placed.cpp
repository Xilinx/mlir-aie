//===- passthrough_kernel_placed.cpp ----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/InitialAllDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include <cstdint>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;
using namespace xilinx::AIEX;

// Command line options
static cl::opt<std::string> deviceOpt("d", cl::desc("AIE Device (npu or npu2)"),
                                       cl::init("npu"));
static cl::opt<int64_t> in1SizeOpt("i1s", cl::desc("Input 1 size in bytes"),
                                    cl::init(4096));
static cl::opt<int64_t> outSizeOpt("os", cl::desc("Output size in bytes"),
                                    cl::init(4096));
static cl::opt<int64_t> traceSizeOpt("t", cl::desc("Trace buffer size"),
                                      cl::init(0));

namespace {

// Helper function to create a memref type
MemRefType createMemRefType(MLIRContext *ctx, ArrayRef<int64_t> shape,
                             Type elementType) {
  return MemRefType::get(shape, elementType);
}

// Main function to generate the passthrough kernel
void generatePassthroughKernel(ModuleOp module, AIEDevice device,
                                int64_t in1Size, int64_t outSize,
                                int64_t traceSize) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();

  // Data type
  Type i8Type = builder.getI8Type();
  Type i32Type = builder.getI32Type();

  // Calculate sizes
  int64_t N = in1Size; // N elements of uint8
  int64_t lineWidthInBytes = N / 4; // chop input in 4 sub-tensors

  // Check that output size matches input size
  if (outSize != in1Size) {
    llvm::errs() << "Error: Output buffer size must be equal to input buffer "
                    "size.\n";
    exit(1);
  }

  // Check input size constraints
  if (in1Size % 64 != 0 || in1Size < 512) {
    llvm::errs() << "Error: In1 buffer size (" << in1Size
                 << ") must be a multiple of 64 and greater than or equal to "
                    "512\n";
    exit(1);
  }

  // Create device operation
  builder.setInsertionPointToStart(module.getBody());
  auto deviceOp =
      builder.create<DeviceOp>(loc, AIEDeviceAttr::get(ctx, device));
  Block *deviceBlock = builder.createBlock(&deviceOp.getRegion());
  builder.setInsertionPointToStart(deviceBlock);

  // Define types
  auto vectorTy = createMemRefType(ctx, {N}, i8Type);
  auto lineTy = createMemRefType(ctx, {lineWidthInBytes}, i8Type);

  // Tile declarations
  auto shimTile = builder.create<TileOp>(loc, 0, 0);
  auto computeTile2 = builder.create<TileOp>(loc, 0, 2);

  // AIE-array data movement with object fifos
  auto ofIn = builder.create<ObjectFifoCreateOp>(
      loc, builder.getStringAttr("in"), shimTile, computeTile2,
      builder.getI32IntegerAttr(2), lineTy);

  auto ofOut = builder.create<ObjectFifoCreateOp>(
      loc, builder.getStringAttr("out"), computeTile2, shimTile,
      builder.getI32IntegerAttr(2), lineTy);

  // Create core operation
  auto coreOp = builder.create<CoreOp>(loc, computeTile2);
  Block *coreBlock = builder.createBlock(&coreOp.getBody());
  builder.setInsertionPointToStart(coreBlock);

  // Create constants for the loop
  auto cZero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto cMaxSize = builder.create<arith::ConstantIndexOp>(loc, INT32_MAX);
  auto cOne = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto cLineWidth =
      builder.create<arith::ConstantOp>(loc, builder.getI32IntegerAttr(lineWidthInBytes));

  // Create the infinite loop
  auto forOp = builder.create<scf::ForOp>(loc, cZero, cMaxSize, cOne);
  Block *loopBlock = &forOp.getRegion().front();
  builder.setInsertionPointToStart(loopBlock);

  // Acquire output and input
  auto elemOut = builder.create<ObjectFifoAcquireOp>(
      loc, TypeRange{lineTy}, ObjectFifoPort::Produce,
      builder.getStringAttr("out"), builder.getI32IntegerAttr(1));

  auto elemIn = builder.create<ObjectFifoAcquireOp>(
      loc, TypeRange{lineTy}, ObjectFifoPort::Consume,
      builder.getStringAttr("in"), builder.getI32IntegerAttr(1));

  // Call external function passThroughLine
  // First, we need to declare the function
  builder.setInsertionPointToStart(deviceBlock);
  auto funcType = builder.getFunctionType(
      {lineTy, lineTy, i32Type}, {});
  auto funcOp = builder.create<func::FuncOp>(
      loc, "passThroughLine", funcType);
  funcOp.setPrivate();

  // Go back to the loop body
  builder.setInsertionPointToEnd(loopBlock);

  // Call the function
  builder.create<func::CallOp>(loc, funcOp,
                                ValueRange{elemIn.getResult(0),
                                           elemOut.getResult(0), cLineWidth});

  // Release input and output
  builder.create<ObjectFifoReleaseOp>(loc, ObjectFifoPort::Consume,
                                       builder.getStringAttr("in"),
                                       builder.getI32IntegerAttr(1));

  builder.create<ObjectFifoReleaseOp>(loc, ObjectFifoPort::Produce,
                                       builder.getStringAttr("out"),
                                       builder.getI32IntegerAttr(1));

  // Yield for the for loop
  builder.create<scf::YieldOp>(loc);

  // End the core
  builder.setInsertionPointToEnd(coreBlock);
  builder.create<EndOp>(loc);

  // Set link_with attribute on core
  coreOp->setAttr("link_with",
                  builder.getStringAttr("passThrough.cc.o"));

  // Create runtime sequence
  builder.setInsertionPointToEnd(deviceBlock);
  auto seqOp = builder.create<AIEX::RuntimeSequenceOp>(
      loc, builder.getFunctionType({vectorTy, vectorTy, vectorTy}, {}),
      "sequence");
  Block *seqBlock = builder.createBlock(&seqOp.getBody());
  seqBlock->addArguments({vectorTy, vectorTy, vectorTy},
                          {loc, loc, loc});
  builder.setInsertionPointToStart(seqBlock);

  Value inTensor = seqBlock->getArgument(0);
  Value outTensor = seqBlock->getArgument(1);
  // notUsed = seqBlock->getArgument(2);

  // Create DMA BD operations for input
  auto inTask = builder.create<AIEX::NpuDmaBdTaskOp>(
      loc, ofIn.getSymNameAttr(), /*bd_id=*/0,
      /*offset=*/builder.getI32IntegerAttr(0),
      /*sizes=*/
      DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{1, 1, 1, (int32_t)N}),
      /*strides=*/DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{0, 0, 0}),
      /*iteration_current=*/nullptr, /*iteration_size=*/nullptr,
      /*iteration_stride=*/nullptr);

  // Create DMA BD operations for output
  auto outTask = builder.create<AIEX::NpuDmaBdTaskOp>(
      loc, ofOut.getSymNameAttr(), /*bd_id=*/0,
      /*offset=*/builder.getI32IntegerAttr(0),
      /*sizes=*/
      DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{1, 1, 1, (int32_t)N}),
      /*strides=*/DenseI32ArrayAttr::get(ctx, ArrayRef<int32_t>{0, 0, 0}),
      /*iteration_current=*/nullptr, /*iteration_size=*/nullptr,
      /*iteration_stride=*/nullptr);

  // Write DMA operations
  builder.create<AIEX::NpuWriteChannelOp>(
      loc, inTask, inTensor, /*channel=*/0, /*column=*/0, /*bd_id=*/0);
  builder.create<AIEX::NpuWriteChannelOp>(
      loc, outTask, outTensor, /*channel=*/0, /*column=*/0, /*bd_id=*/0);

  // Sync operations
  builder.create<AIEX::NpuSyncOp>(loc, /*column=*/0, /*row=*/0,
                                   /*direction=*/
                                   AIEX::AIEDmaDirection::S2MM,
                                   /*channel=*/0);
  builder.create<AIEX::NpuSyncOp>(loc, /*column=*/0, /*row=*/0,
                                   /*direction=*/
                                   AIEX::AIEDmaDirection::MM2S,
                                   /*channel=*/0);

  // End sequence
  builder.create<AIEX::EndOp>(loc);

  // Add terminator to device block
  builder.setInsertionPointToEnd(deviceBlock);
  builder.create<EndOp>(loc);
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register dialects
  DialectRegistry registry;
  xilinx::registerAllDialects(registry);
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Parse command line options
  cl::ParseCommandLineOptions(argc, argv, "AIE passthrough kernel generator\n");

  // Determine device
  AIEDevice device;
  if (deviceOpt == "npu") {
    device = AIEDevice::npu1_1col;
  } else if (deviceOpt == "npu2") {
    device = AIEDevice::npu2;
  } else {
    llvm::errs() << "Error: Device name " << deviceOpt << " is unknown\n";
    return 1;
  }

  // Get sizes
  int64_t in1Size = in1SizeOpt;
  int64_t outSize = outSizeOpt;
  int64_t traceSize = traceSizeOpt;

  // Create module
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);

  // Generate the kernel
  generatePassthroughKernel(module, device, in1Size, outSize, traceSize);

  // Verify the module
  if (failed(verify(module))) {
    llvm::errs() << "Module verification failed\n";
    return 1;
  }

  // Print the module
  module.print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
