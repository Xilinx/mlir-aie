//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_CONVERSION_PASSES
#define AIE_CONVERSION_PASSES

include "mlir/IR/EnumAttr.td"
include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AIEVecToLLVM
//===----------------------------------------------------------------------===//
def Aie2Fp32EmulationType : I32EnumAttr<"Aie2Fp32Emulation", "AIE2 FP32 Emulation",
    [
      I32EnumAttrCase<"AccuracySafe", 0, "accuracy-safe">,
      I32EnumAttrCase<"AccuracyFast", 1, "accuracy-fast">,
      I32EnumAttrCase<"AccuracyLow", 2, "accuracy-low">
    ]>{
  let cppNamespace = "xilinx::aievec";
}

def ConvertAIEVecToLLVM : Pass<"convert-aievec-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert AIEVec dialect to LLVM dialect";
  let description = [{
    This pass converts AIEVec dialect ops to LLVM dialect calls to builtins.
  }];
  let constructor = "xilinx::aievec::createConvertAIEVecToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "xilinx::xllvm::XLLVMDialect"];
  let options = [
      Option<"aie2Fp32Emulation", "aie2-fp32-emulation-strategy", "xilinx::aievec::Aie2Fp32Emulation",
             /*default=*/"xilinx::aievec::Aie2Fp32Emulation::AccuracySafe",
             "Set the AIE2 FP32 emulation strategy. Elementwise multiplication and matrix multiplication intrinsics for FP32 input type are emulated using bfloat16 data-path.",
             [{::llvm::cl::values(
               clEnumValN(xilinx::aievec::Aie2Fp32Emulation::AccuracySafe, "accuracy-safe",
                "Most accurate option since input fp32 number is split into 3 bfloat16 numbers. float a*b would require 9 mac operations due to 3 bfloat16 splits each."),
               clEnumValN(xilinx::aievec::Aie2Fp32Emulation::AccuracyFast, "accuracy-fast",
                "Fast and Accurate option. Input fp32 number is split in to 3 bfloat16 numbers. In the 9 mac operations to emulate fp32 mul, mac operations with LSBs are ignored. (3 last terms)."),
               clEnumValN(xilinx::aievec::Aie2Fp32Emulation::AccuracyLow, "accuracy-low",
                "Fast and least accurate option. Input fp32 number is split in to 2 bfloat16 numbers. In the 4 mac operations to emulate fp32 mul, mac operations with LSBs are ignored. (1 last term).")
              )}]>
   ];
}

//===----------------------------------------------------------------------===//
// AIEToTransaction
//===----------------------------------------------------------------------===//

def ConvertAIEToTransaction : Pass<"convert-aie-to-transaction",
                                   "xilinx::AIE::DeviceOp"> {
  let summary = "Convert AIE dialect to npu transaction operations";
  let description = [{
    This pass converts aie.device operations to a sequence of transaction binary
    operations (writes, masked writes, and block writes) that can be used to
    to configure the npu device. The transaction operations are emitted as
    `npu.write32`, `npu.maskwrite32`, and `npu.blockwrite` operations. A new
    `aiex.runtime_sequence` operation is inserted into the `aie.device` to
    contain the new transaction operations sequence.
  }];
  let constructor = "xilinx::AIE::createConvertAIEToTransactionPass()";
  let dependentDialects = ["xilinx::AIE::AIEDialect",
                           "xilinx::AIEX::AIEXDialect"];
  let options = [
      Option<"clElfDir", "elf-dir", "std::string", /*default=*/"",
             "Where to find ELF files">,
  ];
}

//===----------------------------------------------------------------------===//
// AIEToControlPackets
//===----------------------------------------------------------------------===//

def ConvertAIEToControlPackets : Pass<"convert-aie-to-control-packets",
                                   "xilinx::AIE::DeviceOp"> {
  let summary = "Convert AIE dialect to npu control packet operations";
  let description = [{
    This pass converts aie.device operations to a sequence of control packet
    operations (writes, masked writes, and block writes) that can be used to
    to configure the npu device. A new `aiex.runtime_sequence` operation is
    inserted into the `aie.device` to contain the new control packet sequence.
  }];
  let constructor = "xilinx::AIE::createConvertAIEToControlPacketsPass()";
  let dependentDialects = ["xilinx::AIE::AIEDialect",
                           "xilinx::AIEX::AIEXDialect"];
  let options = [
      Option<"clElfDir", "elf-dir", "std::string", /*default=*/"",
             "Where to find ELF files">,
  ];
}

#endif // AIE_CONVERSION_PASSES
