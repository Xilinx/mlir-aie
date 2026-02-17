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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include <cstdint>
#include <limits>
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

// Main function to generate the passthrough kernel using MLIR C++ API
void generatePassthroughKernel(ModuleOp module, AIEDevice device,
                                int64_t in1Size, int64_t outSize,
                                int64_t traceSize) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  Location loc = builder.getUnknownLoc();

  // Data types
  Type i8Type = builder.getI8Type();
  Type i32Type = builder.getI32Type();
  Type indexType = builder.getIndexType();

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

  // Note: Tracing support not yet implemented
  if (traceSize > 0) {
    llvm::errs() << "Warning: Trace support not yet implemented in C++ version\n";
  }

  // Define types
  auto vectorTy = MemRefType::get({N}, i8Type);
  auto lineTy = MemRefType::get({lineWidthInBytes}, i8Type);

  // Create device operation
  builder.setInsertionPointToStart(module.getBody());
  auto deviceOp = builder.create<DeviceOp>(loc, AIEDeviceAttr::get(ctx, device));
  Block *deviceBlock = builder.createBlock(&deviceOp.getRegion());
  builder.setInsertionPointToStart(deviceBlock);

  // Create tiles
  auto shimTile = TileOp::create(builder, loc, 0, 0);
  auto computeTile2 = TileOp::create(builder, loc, 0, 2);

  // Create ObjectFIFOs
  // ObjectFifo "in" from shim to compute tile
  auto ofInName = builder.getStringAttr("in");
  auto ofInElemType = TypeAttr::get(ObjectFifoType::get(lineTy));
  auto ofIn = builder.create<ObjectFifoCreateOp>(
      loc, ofInName, shimTile, ValueRange{computeTile2},
      builder.getI32IntegerAttr(2), ofInElemType);

  // ObjectFifo "out" from compute tile to shim
  auto ofOutName = builder.getStringAttr("out");
  auto ofOutElemType = TypeAttr::get(ObjectFifoType::get(lineTy));
  auto ofOut = builder.create<ObjectFifoCreateOp>(
      loc, ofOutName, computeTile2, ValueRange{shimTile},
      builder.getI32IntegerAttr(2), ofOutElemType);

  // Create function declaration for external kernel
  auto funcType = builder.getFunctionType({lineTy, lineTy, i32Type}, {});
  auto funcOp = builder.create<func::FuncOp>(loc, "passThroughLine", funcType);
  funcOp.setPrivate();

  // Create core operation
  auto coreOp = CoreOp::create(builder, loc, indexType, computeTile2);
  Region &coreRegion = coreOp.getBody();
  coreRegion.push_back(new Block);
  Block *coreBlock = &coreRegion.back();
  builder.setInsertionPointToStart(coreBlock);

  // Create constants for the loop
  auto cZero = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto cMaxSize = builder.create<arith::ConstantIndexOp>(loc, std::numeric_limits<int32_t>::max());
  auto cOne = builder.create<arith::ConstantIndexOp>(loc, 1);

  // Create the infinite loop
  auto forOp = builder.create<scf::ForOp>(loc, cZero, cMaxSize, cOne);
  Block *loopBlock = &forOp.getRegion().front();
  builder.setInsertionPointToStart(loopBlock);

  // Acquire output and input
  auto subviewType = ObjectFifoSubviewType::get(lineTy);
  auto elemOut = builder.create<ObjectFifoAcquireOp>(
      loc, subviewType, ObjectFifoPort::Produce, ofOutName,
      builder.getI32IntegerAttr(1));

  auto elemIn = builder.create<ObjectFifoAcquireOp>(
      loc, subviewType, ObjectFifoPort::Consume, ofInName,
      builder.getI32IntegerAttr(1));

  // Access the subviews
  auto elemOutMem = builder.create<ObjectFifoSubviewAccessOp>(
      loc, lineTy, elemOut.getSubview(), builder.getI32IntegerAttr(0));

  auto elemInMem = builder.create<ObjectFifoSubviewAccessOp>(
      loc, lineTy, elemIn.getSubview(), builder.getI32IntegerAttr(0));

  // Create constant for line width
  auto cLineWidth = builder.create<arith::ConstantOp>(
      loc, i32Type, builder.getI32IntegerAttr(lineWidthInBytes));

  // Call the external function
  builder.create<func::CallOp>(loc, funcOp,
                                ValueRange{elemInMem, elemOutMem, cLineWidth});

  // Release input and output
  builder.create<ObjectFifoReleaseOp>(loc, ObjectFifoPort::Consume, ofInName,
                                       builder.getI32IntegerAttr(1));

  builder.create<ObjectFifoReleaseOp>(loc, ObjectFifoPort::Produce, ofOutName,
                                       builder.getI32IntegerAttr(1));

  // Yield for the for loop
  builder.create<scf::YieldOp>(loc);

  // End the core
  builder.setInsertionPointToEnd(coreBlock);
  EndOp::create(builder, loc);

  // Set link_with attribute on core
  coreOp->setAttr("link_with", builder.getStringAttr("passThrough.cc.o"));

  // Create runtime sequence function
  builder.setInsertionPointToEnd(deviceBlock);
  auto seqFuncType = builder.getFunctionType({vectorTy, vectorTy, vectorTy}, {});
  auto seqFuncOp = builder.create<func::FuncOp>(loc, "sequence", seqFuncType);
  Block *seqBlock = seqFuncOp.addEntryBlock();
  builder.setInsertionPointToStart(seqBlock);

  Value inTensor = seqBlock->getArgument(0);
  Value outTensor = seqBlock->getArgument(1);
  // Value notUsed = seqBlock->getArgument(2);

  // Create DMA operations for input
  SmallVector<int64_t> staticOffsets = {0, 0, 0, 0};
  SmallVector<int64_t> staticSizes = {1, 1, 1, (int64_t)N};
  SmallVector<int64_t> staticStrides = {0, 0, 0};

  NpuDmaMemcpyNdOp::create(
      builder, loc, inTensor,
      /*offsets=*/SmallVector<Value>{},
      /*sizes=*/SmallVector<Value>{},
      /*strides=*/SmallVector<Value>{},
      ArrayRef(staticOffsets), ArrayRef(staticSizes), ArrayRef(staticStrides),
      /*packet=*/nullptr, /*metadata=*/ofInName, /*id=*/0,
      /*issue_token=*/false,
      /*d0_zero_before=*/0, /*d1_zero_before=*/0, /*d2_zero_before=*/0,
      /*d0_zero_after=*/0, /*d1_zero_after=*/0, /*d2_zero_after=*/0,
      /*burst_length=*/0);

  // Create DMA operations for output
  NpuDmaMemcpyNdOp::create(
      builder, loc, outTensor,
      /*offsets=*/SmallVector<Value>{},
      /*sizes=*/SmallVector<Value>{},
      /*strides=*/SmallVector<Value>{},
      ArrayRef(staticOffsets), ArrayRef(staticSizes), ArrayRef(staticStrides),
      /*packet=*/nullptr, /*metadata=*/ofOutName, /*id=*/1,
      /*issue_token=*/false,
      /*d0_zero_before=*/0, /*d1_zero_before=*/0, /*d2_zero_before=*/0,
      /*d0_zero_after=*/0, /*d1_zero_after=*/0, /*d2_zero_after=*/0,
      /*burst_length=*/0);

  // Sync operation
  NpuSyncOp::create(builder, loc,
                     /*column=*/builder.getI32IntegerAttr(0),
                     /*row=*/builder.getI32IntegerAttr(0),
                     /*direction=*/builder.getI32IntegerAttr(0),
                     /*channel=*/builder.getI32IntegerAttr(0),
                     /*column_num=*/builder.getI32IntegerAttr(1),
                     /*row_num=*/builder.getI32IntegerAttr(1));

  // Return
  builder.create<func::ReturnOp>(loc);

  // Add terminator to device block
  builder.setInsertionPointToEnd(deviceBlock);
  EndOp::create(builder, loc);
}

} // namespace

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  // Register dialects
  DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::AIEX::AIEXDialect>();

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
