//===- else_condition_check.mlir ---------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses : Fails
// 
// malloc(): unaligned tcache chunk detected
// PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses else_condition_check.mlir
// malloc(): unaligned tcache chunk detected
// Aborted (core dumped)


// RUN: aie-opt --aie-objectFifo-stateful-transform else_condition_check.mlir | aie-opt --aie-assign-buffer-addresses : Passes

// After merging the main with alloc-flags branch and using all the Passes, rather than just the 2.
// free(): invalid next size (fast)
// PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: aie-opt --aie-canonicalize-device --aie-assign-lock-ids --aie-register-objectFifos --aie-objectFifo-stateful-transform --aie-assign-bd-ids --aie-assign-buffer-addresses else_condition_check.mlir
//  #0 0x000060e11514cc30 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) Signals.cpp:0:0
//  #1 0x000060e11514a56e SignalHandler(int) Signals.cpp:0:0
//  #2 0x00007d0a82a42520 (/lib/x86_64-linux-gnu/libc.so.6+0x42520)
//  #3 0x00007d0a82a969fc __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
//  #4 0x00007d0a82a969fc __pthread_kill_internal ./nptl/pthread_kill.c:78:10
//  #5 0x00007d0a82a969fc pthread_kill ./nptl/pthread_kill.c:89:10
//  #6 0x00007d0a82a42476 gsignal ./signal/../sysdeps/posix/raise.c:27:6
//  #7 0x00007d0a82a287f3 abort ./stdlib/abort.c:81:7
//  #8 0x00007d0a82a89676 __libc_message ./libio/../sysdeps/posix/libc_fatal.c:155:5
//  #9 0x00007d0a82aa0cfc ./malloc/malloc.c:5668:3
// #10 0x00007d0a82aa2a9d _int_free ./malloc/malloc.c:4522:4
// #11 0x00007d0a82aa5453 __libc_free ./malloc/malloc.c:3394:3
// #12 0x000060e114ef3f2e mlir::detail::OpToOpPassAdaptor::~OpToOpPassAdaptor() Pass.cpp:0:0
// #13 0x000060e114ef5e8b mlir::PassManager::~PassManager() Pass.cpp:0:0
// #14 0x000060e114a78453 performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
// #15 0x000060e114a79683 processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
// #16 0x000060e114a797bd mlir::LogicalResult llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
// #17 0x000060e1150db529 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) ToolUtilities.cpp:0:0
// #18 0x000060e114a72677 mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
// #19 0x000060e114a7990c mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) MlirOptMain.cpp:0:0
// #20 0x000060e114a79e17 mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) MlirOptMain.cpp:0:0
// #21 0x000060e112a3a526 main aie-opt.cpp:0:0
// #22 0x00007d0a82a29d90 __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
// #23 0x00007d0a82a29e40 call_init ./csu/../csu/libc-start.c:128:20
// #24 0x00007d0a82a29e40 __libc_start_main ./csu/../csu/libc-start.c:379:5
// #25 0x000060e112a298e5 _start (/scratch/pvasired/mlir-aie/install/bin/aie-opt+0xbd18e5)
// Aborted (core dumped)

// Jeff's Output:
// #11 0x000055b3a1a27224 std::vector<long, std::allocator<long>>::operator[](unsigned long) /usr/bin/../lib/gcc/x86_64-linux-gnu/13/../../../../include/c++/13/bits/stl_vector.h:1125:2
// #12 0x000055b3a1a22ba0 setBufferAddress(xilinx::AIE::BufferOp, int, int, std::vector<long, std::allocator<long>>&, std::vector<BankLimits, std::allocator<BankLimits>>&) 
// /work/acdc/aie/lib/Dialect/AIE/Transforms/AIEAssignBuffers.cpp:205:25


// Justifies that error occurs only from bank-aware allocation fail
// else_condition_check.mlir:63:13: error: 'aie.tile' op allocated buffers exceeded available memory: Bank aware
// (no stack allocated)

// malloc(): unaligned tcache chunk detected
// PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: aie-opt --aie-objectFifo-stateful-transform --aie-assign-buffer-addresses=alloc-scheme=bank-aware else_condition_check.mlir
// malloc(): unaligned tcache chunk detected
// Aborted (core dumped)

// module @test {
//  aie.device(xcvc1902) {
//   %tile12 = aie.tile(1, 2)
//   %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
//   %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> // 32 bytes
//   %tile13 = aie.tile(1, 3)
//   aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>>
//  }
// }


// If objectFifo's source and destination is the same tile
// corrupted size vs. prev_size
// PLEASE submit a bug report to https://github.com/llvm/llvm-project/issues/ and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: aie-opt --aie-canonicalize-device --aie-assign-lock-ids --aie-register-objectFifos --aie-objectFifo-stateful-transform --aie-assign-bd-ids --aie-assign-buffer-addresses else_condition_check.mlir
//  #0 0x0000576916c1fd00 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) Signals.cpp:0:0
//  #1 0x0000576916c1d63e SignalHandler(int) Signals.cpp:0:0
//  #2 0x0000708a2bc42520 (/lib/x86_64-linux-gnu/libc.so.6+0x42520)
//  #3 0x0000708a2bc969fc __pthread_kill_implementation ./nptl/pthread_kill.c:44:76
//  #4 0x0000708a2bc969fc __pthread_kill_internal ./nptl/pthread_kill.c:78:10
//  #5 0x0000708a2bc969fc pthread_kill ./nptl/pthread_kill.c:89:10
//  #6 0x0000708a2bc42476 gsignal ./signal/../sysdeps/posix/raise.c:27:6
//  #7 0x0000708a2bc287f3 abort ./stdlib/abort.c:81:7
//  #8 0x0000708a2bc89676 __libc_message ./libio/../sysdeps/posix/libc_fatal.c:155:5
//  #9 0x0000708a2bca0cfc ./malloc/malloc.c:5668:3
// #10 0x0000708a2bca17e2 unlink_chunk ./malloc/malloc.c:1643:2
// #11 0x0000708a2bca1969 malloc_consolidate ./malloc/malloc.c:4780:6
// #12 0x0000708a2bca3bdb _int_malloc ./malloc/malloc.c:3965:9
// #13 0x0000708a2bca5262 __libc_malloc ./malloc/malloc.c:3322:7
// #14 0x0000708a2c0b751c operator new(unsigned long) (/lib/x86_64-linux-gnu/libstdc++.so.6+0xb751c)
// #15 0x0000576916c00ef4 llvm::raw_ostream::SetBuffered() raw_ostream.cpp:0:0
// #16 0x0000576916c01bad llvm::raw_ostream::write(char const*, unsigned long) raw_ostream.cpp:0:0
// #17 0x0000576916b18351 void llvm::detail::UniqueFunctionBase<void, mlir::Operation*, mlir::OpAsmPrinter&, llvm::StringRef>::CallImpl<mlir::Op<mlir::ModuleOp, mlir::OpTrait::OneRegion, mlir::OpTrait::ZeroResults, mlir::OpTrait::ZeroSuccessors, mlir::OpTrait::ZeroOperands, mlir::OpTrait::NoRegionArguments, mlir::OpTrait::NoTerminator, mlir::OpTrait::SingleBlock, mlir::OpTrait::OpInvariants, mlir::BytecodeOpInterface::Trait, mlir::OpTrait::AffineScope, mlir::OpTrait::IsIsolatedFromAbove, mlir::OpTrait::SymbolTable, mlir::SymbolOpInterface::Trait, mlir::OpAsmOpInterface::Trait, mlir::RegionKindInterface::Trait, mlir::OpTrait::HasOnlyGraphRegion>::getPrintAssemblyFn()::'lambda'(mlir::Operation*, mlir::OpAsmPrinter&, llvm::StringRef) const>(void*, mlir::Operation*, mlir::OpAsmPrinter&, llvm::StringRef) BuiltinDialect.cpp:0:0
// #18 0x0000576916b10314 mlir::RegisteredOperationName::Model<mlir::ModuleOp>::printAssembly(mlir::Operation*, mlir::OpAsmPrinter&, llvm::StringRef) BuiltinDialect.cpp:0:0
// #19 0x0000576916ad712e mlir::Operation::print(llvm::raw_ostream&, mlir::AsmState&) AsmPrinter.cpp:0:0
// #20 0x000057691655162e performActions(llvm::raw_ostream&, std::shared_ptr<llvm::SourceMgr> const&, mlir::MLIRContext*, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
// #21 0x0000576916551bd3 processBuffer(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::MlirOptMainConfig const&, mlir::DialectRegistry&, llvm::ThreadPoolInterface*) MlirOptMain.cpp:0:0
// #22 0x0000576916551d0d mlir::LogicalResult llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>::callback_fn<mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&)::'lambda'(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>(long, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&) MlirOptMain.cpp:0:0
// #23 0x0000576916bacc49 mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) ToolUtilities.cpp:0:0
// #24 0x000057691654abc7 mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) MlirOptMain.cpp:0:0
// #25 0x0000576916551e5c mlir::MlirOptMain(int, char**, llvm::StringRef, llvm::StringRef, mlir::DialectRegistry&) MlirOptMain.cpp:0:0
// #26 0x0000576916552367 mlir::MlirOptMain(int, char**, llvm::StringRef, mlir::DialectRegistry&) MlirOptMain.cpp:0:0
// #27 0x0000576914485056 main aie-opt.cpp:0:0
// #28 0x0000708a2bc29d90 __libc_start_call_main ./csu/../sysdeps/nptl/libc_start_call_main.h:58:16
// #29 0x0000708a2bc29e40 call_init ./csu/../csu/libc-start.c:128:20
// #30 0x0000708a2bc29e40 __libc_start_main ./csu/../csu/libc-start.c:379:5
// #31 0x0000576914473fa5 _start (/scratch/pvasired/mlir-aie/install/bin/aie-opt+0xbf2fa5)
// Aborted (core dumped)
// module @test {
//  aie.device(xcvc1902) {
//   %tile12 = aie.tile(1, 2)
//   %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
//   %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> // 32 bytes
//   %tile13 = aie.tile(1, 3)
//   aie.objectfifo @act_3_4(%tile12, {%tile12}, 4 : i32) : !aie.objectfifo<memref<8xi32>>
//  }
// }

module @test {
 aie.device(xcvc1902) {
  %tile12 = aie.tile(1, 2)
  %1 = aie.buffer(%tile12) { sym_name = "a" } : memref<4096xi32>  //16384 bytes
  %b1 = aie.buffer(%tile12) { sym_name = "b" } : memref<16xi16> // 32 bytes
  %tile13 = aie.tile(1, 3)
  aie.objectfifo @act_3_4(%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<8xi32>>
 }
}
