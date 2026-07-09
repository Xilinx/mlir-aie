//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
      Option<"aieTarget", "aie-target", "std::string",
             /*default=*/"\"aie2\"",
             "Select AIE version: \\\"aie\\\", \\\"aie2\\\", or \\\"aie2p\\\". This will determine which XLLVM intrinsics to use.">,
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
    configure the npu device. The transaction operations are emitted as
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
      Option<"clDeviceName", "device-name", "std::string", /*default=*/"",
             "Which device to generate code for; if none specified, process all devices">,
  ];
}

//===----------------------------------------------------------------------===//
// AIEXToEmitC
//===----------------------------------------------------------------------===//

def ConvertAIEXToEmitC : Pass<"convert-aiex-to-emitc", "mlir::ModuleOp"> {
  let summary = "Convert AIEX runtime sequence ops to EmitC for C++ TXN generation";
  let description = [{
    This pass converts the straight-line npu transaction ops in an
    `aie.runtime_sequence` (`npu.write32`, `npu.maskwrite32`, `npu.sync`,
    `npu.address_patch`, `npu.blockwrite`) into EmitC dialect calls naming the
    functions in `aie/Runtime/TxnEncoding.h`. Running MLIR's `translateToCpp()`
    on the result produces a standalone `generate_txn_<seq>(...)` C++ function
    that assembles the TXN binary at run time -- the same words the compile-time
    binary emitter (`AIETargetNPU.cpp`) produces, but from a C++ function so
    runtime-valued operands can flow through to a single compiled xclbin.

    Control flow (`scf.for`/`scf.if`) inside a runtime sequence is rejected with
    a diagnostic; it is handled by later passes in the dynamic-sequences work.
  }];
  let constructor = "xilinx::createConvertAIEXToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect",
                           "xilinx::AIE::AIEDialect",
                           "xilinx::AIEX::AIEXDialect"];
}

//===----------------------------------------------------------------------===//
// AIEToControlPackets
//===----------------------------------------------------------------------===//

def ConvertAIEToControlPackets : Pass<"convert-aie-to-control-packets",
                                   "xilinx::AIE::DeviceOp"> {
  let summary = "Convert AIE dialect to npu control packet operations";
  let description = [{
    This pass converts aie.device operations to a sequence of control packet
    operations (`npu.control_packet`) that can be used to configure the npu
    device. A new `aiex.runtime_sequence` operation is inserted into the
    `aie.device` to contain the new control packet sequence.
  }];
  let constructor = "xilinx::AIE::createConvertAIEToControlPacketsPass()";
  let dependentDialects = ["xilinx::AIE::AIEDialect",
                           "xilinx::AIEX::AIEXDialect"];
  let options = [
      Option<"clElfDir", "elf-dir", "std::string", /*default=*/"",
             "Where to find ELF files">,
      Option<"clDeviceName", "device-name", "std::string", /*default=*/"",
             "Which device to generate code for; if none specified, process all devices">,
  ];
}

#endif // AIE_CONVERSION_PASSES
