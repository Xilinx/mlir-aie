// RUN: aie-opt --aie-assign-buffer-addresses %s > %t.temp1
// RUN: aie-opt --aie-llvm-lowering="tilecol=1 tilerow=1" %t.temp1 | aie-translate --aie-generate-llvmir | opt -strip -S | llc -O0 --march=aie --filetype=obj -o=%t.o
// RUN: aie-translate --aie-generate-mmap %t.temp1
// RUN: ld.lld %t.o %S/../../runtime_lib/me_basic.o -T %S/ld.script -o %t.out
// RUN: llvm-objdump -dr --arch-name=aie %t.out | FileCheck -check-prefix=CHECK11 %s

// Test LLVM lowering for lock accesses and memory accesses (LockOp, UseLockOp, and BufferOp)
// Things to make sure:
//   - LockID: depending on which tile (or memory module) a lock is instantiated, create a lock ID
//             that has correct offset from a core's view (based on cardinal direction)
//   - Buffer: depending on which tile (or memory module) a buffer is instantiated, create an LLVM
//             static allocation (for now) for each core that can access to the buffer
// _main_init comes from me_basic.o.  This has to exist at address 0.
// CHECK11: 00000000 <_main_init>:
// CHECK11: <core11>:
// CHECK11: acq
// CHECK11: st
// CHECK11: st
// CHECK11: rel

module @test_core_llvm1 {
  %tile11 = AIE.tile(1, 1)
  %tile21 = AIE.tile(1, 2)

  %lock11_8 = AIE.lock(%tile11, 8)
  %buf11_0  = AIE.buffer(%tile11) { sym_name = "a" } : memref<256xi32>
  %buf12_0  = AIE.buffer(%tile11) { sym_name = "b" } : memref<256xi32>

  %core11 = AIE.core(%tile11) {
    AIE.useLock(%lock11_8, Acquire, 0, 0)
    %0 = constant 1 : i32
    %i = constant 16 : index
    store %0, %buf11_0[%i] : memref<256xi32>
    store %0, %buf12_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 1, 0)
    AIE.end
  }

  %core21 = AIE.core(%tile21) {
    AIE.useLock(%lock11_8, Acquire, 1, 0)
    %i = constant 16 : index
    %0 = load %buf11_0[%i] : memref<256xi32>
    AIE.useLock(%lock11_8, Release, 0, 0)
    AIE.end
  }
}
