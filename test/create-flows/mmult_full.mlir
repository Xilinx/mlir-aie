// RUN: aie-opt --aie-create-flows --aie-find-flows %s | FileCheck %s
// CHECK: %[[T1:.*]] = AIE.tile(7, 0)
// CHECK: %[[T3:.*]] = AIE.tile(8, 3)
// CHECK: %[[T15:.*]] = AIE.tile(6, 0)
// CHECK: %[[T17:.*]] = AIE.tile(7, 3)
// CHECK: %[[T29:.*]] = AIE.tile(3, 0)
// CHECK: %[[T31:.*]] = AIE.tile(8, 2)
// CHECK: %[[T43:.*]] = AIE.tile(2, 0)
// CHECK: %[[T45:.*]] = AIE.tile(7, 2)
//
// CHECK: AIE.flow(%[[T1]], DMA : 0, %[[T3]], DMA : 0)
// CHECK: AIE.flow(%[[T1]], DMA : 1, %[[T3]], DMA : 1)
// CHECK: AIE.flow(%[[T3]], DMA : 0, %[[T29]], DMA : 1)
// CHECK: AIE.flow(%[[T15]], DMA : 0, %[[T17]], DMA : 0)
// CHECK: AIE.flow(%[[T15]], DMA : 1, %[[T17]], DMA : 1)
// CHECK: AIE.flow(%[[T17]], DMA : 0, %[[T29]], DMA : 0)
// CHECK: AIE.flow(%[[T29]], DMA : 0, %[[T31]], DMA : 0)
// CHECK: AIE.flow(%[[T29]], DMA : 1, %[[T31]], DMA : 1)
// CHECK: AIE.flow(%[[T31]], DMA : 0, %[[T43]], DMA : 1)
// CHECK: AIE.flow(%[[T43]], DMA : 0, %[[T45]], DMA : 0)
// CHECK: AIE.flow(%[[T43]], DMA : 1, %[[T45]], DMA : 1)
// CHECK: AIE.flow(%[[T45]], DMA : 0, %[[T43]], DMA : 0)

module @aie.herd_0  {
  %0 = AIE.tile(7, 1)
  %1 = AIE.tile(7, 0)
  %2 = AIE.tile(1, 1)
  %3 = AIE.tile(8, 3)
  %4 = AIE.lock(%3, 1)
  %5 = AIE.lock(%3, 3)
  %6 = AIE.buffer(%3) {sym_name = "buf11"} : memref<16x16xf32, 2>
  %7 = AIE.lock(%3, 2)
  %8 = AIE.buffer(%3) {sym_name = "buf10"} : memref<16x16xf32, 2>
  %9 = AIE.lock(%3, 0)
  %10 = AIE.buffer(%3) {sym_name = "buf9"} : memref<16x16xf32, 2>
  %11 = AIE.mem(%3)  {
    %63 = AIE.dmaStart(S2MM0, ^bb1, ^bb5)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%9, Acquire, 0, 0)
    AIE.dmaBd(<%10 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%9, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%4, Acquire, 0, 0)
    AIE.dmaBd(<%6 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%4, Release, 1, 0)
    br ^bb1
  ^bb3:  // pred: ^bb5
    %64 = AIE.dmaStart(S2MM1, ^bb4, ^bb7)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%7, Acquire, 0, 0)
    AIE.dmaBd(<%8 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%7, Release, 1, 0)
    br ^bb4
  ^bb5:  // pred: ^bb0
    %65 = AIE.dmaStart(MM2S0, ^bb6, ^bb3)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    AIE.useLock(%5, Acquire, 1, 0)
    AIE.dmaBd(<%6 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%5, Release, 0, 0)
    br ^bb6
  ^bb7:  // pred: ^bb3
    AIE.end
  }
  %12 = AIE.core(%3)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    scf.for %arg0 = %c0 to %c32 step %c16 {
      AIE.useLock(%9, Acquire, 1, 0)
      AIE.useLock(%7, Acquire, 1, 0)
      AIE.useLock(%4, Acquire, 1, 0)
      AIE.useLock(%5, Acquire, 0, 0)
      %c0_0 = constant 0 : index
      %c16_1 = constant 16 : index
      %c1 = constant 1 : index
      scf.for %arg1 = %c0_0 to %c16_1 step %c1 {
        %c0_2 = constant 0 : index
        %c16_3 = constant 16 : index
        %c1_4 = constant 1 : index
        scf.for %arg2 = %c0_2 to %c16_3 step %c1_4 {
          %c0_5 = constant 0 : index
          %c16_6 = constant 16 : index
          %c1_7 = constant 1 : index
          scf.for %arg3 = %c0_5 to %c16_6 step %c1_7 {
            %63 = memref.load %10[%arg1, %arg3] : memref<16x16xf32, 2>
            %64 = memref.load %8[%arg3, %arg2] : memref<16x16xf32, 2>
            %65 = memref.load %6[%arg1, %arg2] : memref<16x16xf32, 2>
            %66 = mulf %63, %64 : f32
            %67 = addf %65, %66 : f32
            memref.store %67, %6[%arg1, %arg2] : memref<16x16xf32, 2>
          }
        }
      }
      AIE.useLock(%9, Release, 0, 0)
      AIE.useLock(%7, Release, 0, 0)
      AIE.useLock(%4, Release, 0, 0)
      AIE.useLock(%5, Release, 1, 0)
    }
    AIE.end
  }
  %13 = AIE.tile(6, 2)
  %14 = AIE.tile(6, 1)
  %15 = AIE.tile(6, 0)
  %16 = AIE.tile(0, 1)
  %17 = AIE.tile(7, 3)
  %18 = AIE.lock(%17, 1)
  %19 = AIE.lock(%17, 3)
  %20 = AIE.buffer(%17) {sym_name = "buf8"} : memref<16x16xf32, 2>
  %21 = AIE.lock(%17, 2)
  %22 = AIE.buffer(%17) {sym_name = "buf7"} : memref<16x16xf32, 2>
  %23 = AIE.lock(%17, 0)
  %24 = AIE.buffer(%17) {sym_name = "buf6"} : memref<16x16xf32, 2>
  %25 = AIE.mem(%17)  {
    %63 = AIE.dmaStart(S2MM0, ^bb1, ^bb5)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%23, Acquire, 0, 0)
    AIE.dmaBd(<%24 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%23, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%18, Acquire, 0, 0)
    AIE.dmaBd(<%20 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%18, Release, 1, 0)
    br ^bb1
  ^bb3:  // pred: ^bb5
    %64 = AIE.dmaStart(S2MM1, ^bb4, ^bb7)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%21, Acquire, 0, 0)
    AIE.dmaBd(<%22 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%21, Release, 1, 0)
    br ^bb4
  ^bb5:  // pred: ^bb0
    %65 = AIE.dmaStart(MM2S0, ^bb6, ^bb3)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    AIE.useLock(%19, Acquire, 1, 0)
    AIE.dmaBd(<%20 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%19, Release, 0, 0)
    br ^bb6
  ^bb7:  // pred: ^bb3
    AIE.end
  }
  %26 = AIE.core(%17)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    scf.for %arg0 = %c0 to %c32 step %c16 {
      AIE.useLock(%23, Acquire, 1, 0)
      AIE.useLock(%21, Acquire, 1, 0)
      AIE.useLock(%18, Acquire, 1, 0)
      AIE.useLock(%19, Acquire, 0, 0)
      %c0_0 = constant 0 : index
      %c16_1 = constant 16 : index
      %c1 = constant 1 : index
      scf.for %arg1 = %c0_0 to %c16_1 step %c1 {
        %c0_2 = constant 0 : index
        %c16_3 = constant 16 : index
        %c1_4 = constant 1 : index
        scf.for %arg2 = %c0_2 to %c16_3 step %c1_4 {
          %c0_5 = constant 0 : index
          %c16_6 = constant 16 : index
          %c1_7 = constant 1 : index
          scf.for %arg3 = %c0_5 to %c16_6 step %c1_7 {
            %63 = memref.load %24[%arg1, %arg3] : memref<16x16xf32, 2>
            %64 = memref.load %22[%arg3, %arg2] : memref<16x16xf32, 2>
            %65 = memref.load %20[%arg1, %arg2] : memref<16x16xf32, 2>
            %66 = mulf %63, %64 : f32
            %67 = addf %65, %66 : f32
            memref.store %67, %20[%arg1, %arg2] : memref<16x16xf32, 2>
          }
        }
      }
      AIE.useLock(%23, Release, 0, 0)
      AIE.useLock(%21, Release, 0, 0)
      AIE.useLock(%18, Release, 0, 0)
      AIE.useLock(%19, Release, 1, 0)
    }
    AIE.end
  }
  %27 = AIE.tile(3, 2)
  %28 = AIE.tile(3, 1)
  %29 = AIE.tile(3, 0)
  %30 = AIE.tile(1, 0)
  %31 = AIE.tile(8, 2)
  %32 = AIE.lock(%31, 1)
  %33 = AIE.lock(%31, 3)
  %34 = AIE.buffer(%31) {sym_name = "buf5"} : memref<16x16xf32, 2>
  %35 = AIE.lock(%31, 2)
  %36 = AIE.buffer(%31) {sym_name = "buf4"} : memref<16x16xf32, 2>
  %37 = AIE.lock(%31, 0)
  %38 = AIE.buffer(%31) {sym_name = "buf3"} : memref<16x16xf32, 2>
  %39 = AIE.mem(%31)  {
    %63 = AIE.dmaStart(S2MM0, ^bb1, ^bb5)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%37, Acquire, 0, 0)
    AIE.dmaBd(<%38 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%37, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%32, Acquire, 0, 0)
    AIE.dmaBd(<%34 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%32, Release, 1, 0)
    br ^bb1
  ^bb3:  // pred: ^bb5
    %64 = AIE.dmaStart(S2MM1, ^bb4, ^bb7)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%35, Acquire, 0, 0)
    AIE.dmaBd(<%36 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%35, Release, 1, 0)
    br ^bb4
  ^bb5:  // pred: ^bb0
    %65 = AIE.dmaStart(MM2S0, ^bb6, ^bb3)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    AIE.useLock(%33, Acquire, 1, 0)
    AIE.dmaBd(<%34 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%33, Release, 0, 0)
    br ^bb6
  ^bb7:  // pred: ^bb3
    AIE.end
  }
  %40 = AIE.core(%31)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    scf.for %arg0 = %c0 to %c32 step %c16 {
      AIE.useLock(%37, Acquire, 1, 0)
      AIE.useLock(%35, Acquire, 1, 0)
      AIE.useLock(%32, Acquire, 1, 0)
      AIE.useLock(%33, Acquire, 0, 0)
      %c0_0 = constant 0 : index
      %c16_1 = constant 16 : index
      %c1 = constant 1 : index
      scf.for %arg1 = %c0_0 to %c16_1 step %c1 {
        %c0_2 = constant 0 : index
        %c16_3 = constant 16 : index
        %c1_4 = constant 1 : index
        scf.for %arg2 = %c0_2 to %c16_3 step %c1_4 {
          %c0_5 = constant 0 : index
          %c16_6 = constant 16 : index
          %c1_7 = constant 1 : index
          scf.for %arg3 = %c0_5 to %c16_6 step %c1_7 {
            %63 = memref.load %38[%arg1, %arg3] : memref<16x16xf32, 2>
            %64 = memref.load %36[%arg3, %arg2] : memref<16x16xf32, 2>
            %65 = memref.load %34[%arg1, %arg2] : memref<16x16xf32, 2>
            %66 = mulf %63, %64 : f32
            %67 = addf %65, %66 : f32
            memref.store %67, %34[%arg1, %arg2] : memref<16x16xf32, 2>
          }
        }
      }
      AIE.useLock(%37, Release, 0, 0)
      AIE.useLock(%35, Release, 0, 0)
      AIE.useLock(%32, Release, 0, 0)
      AIE.useLock(%33, Release, 1, 0)
    }
    AIE.end
  }
  %41 = AIE.tile(2, 2)
  %42 = AIE.tile(2, 1)
  %43 = AIE.tile(2, 0)
  %44 = AIE.tile(0, 0)
  %45 = AIE.tile(7, 2)
  %46 = AIE.lock(%45, 1)
  %47 = AIE.lock(%45, 3)
  %48 = AIE.buffer(%45) {sym_name = "buf2"} : memref<16x16xf32, 2>
  %49 = AIE.lock(%45, 2)
  %50 = AIE.buffer(%45) {sym_name = "buf1"} : memref<16x16xf32, 2>
  %51 = AIE.lock(%45, 0)
  %52 = AIE.buffer(%45) {sym_name = "buf0"} : memref<16x16xf32, 2>
  %53 = AIE.mem(%45)  {
    %63 = AIE.dmaStart(S2MM0, ^bb1, ^bb5)
  ^bb1:  // 2 preds: ^bb0, ^bb2
    AIE.useLock(%51, Acquire, 0, 0)
    AIE.dmaBd(<%52 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%51, Release, 1, 0)
    br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%46, Acquire, 0, 0)
    AIE.dmaBd(<%48 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%46, Release, 1, 0)
    br ^bb1
  ^bb3:  // pred: ^bb5
    %64 = AIE.dmaStart(S2MM1, ^bb4, ^bb7)
  ^bb4:  // 2 preds: ^bb3, ^bb4
    AIE.useLock(%49, Acquire, 0, 0)
    AIE.dmaBd(<%50 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%49, Release, 1, 0)
    br ^bb4
  ^bb5:  // pred: ^bb0
    %65 = AIE.dmaStart(MM2S0, ^bb6, ^bb3)
  ^bb6:  // 2 preds: ^bb5, ^bb6
    AIE.useLock(%47, Acquire, 1, 0)
    AIE.dmaBd(<%48 : memref<16x16xf32, 2>, 0, 256>, 0)
    AIE.useLock(%47, Release, 0, 0)
    br ^bb6
  ^bb7:  // pred: ^bb3
    AIE.end
  }
  %54 = AIE.core(%45)  {
    br ^bb1
  ^bb1:  // pred: ^bb0
    br ^bb2
  ^bb2:  // pred: ^bb1
    %c32 = constant 32 : index
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    scf.for %arg0 = %c0 to %c32 step %c16 {
      AIE.useLock(%51, Acquire, 1, 0)
      AIE.useLock(%49, Acquire, 1, 0)
      AIE.useLock(%46, Acquire, 1, 0)
      AIE.useLock(%47, Acquire, 0, 0)
      %c0_0 = constant 0 : index
      %c16_1 = constant 16 : index
      %c1 = constant 1 : index
      scf.for %arg1 = %c0_0 to %c16_1 step %c1 {
        %c0_2 = constant 0 : index
        %c16_3 = constant 16 : index
        %c1_4 = constant 1 : index
        scf.for %arg2 = %c0_2 to %c16_3 step %c1_4 {
          %c0_5 = constant 0 : index
          %c16_6 = constant 16 : index
          %c1_7 = constant 1 : index
          scf.for %arg3 = %c0_5 to %c16_6 step %c1_7 {
            %63 = memref.load %52[%arg1, %arg3] : memref<16x16xf32, 2>
            %64 = memref.load %50[%arg3, %arg2] : memref<16x16xf32, 2>
            %65 = memref.load %48[%arg1, %arg2] : memref<16x16xf32, 2>
            %66 = mulf %63, %64 : f32
            %67 = addf %65, %66 : f32
            memref.store %67, %48[%arg1, %arg2] : memref<16x16xf32, 2>
          }
        }
      }
      AIE.useLock(%51, Release, 0, 0)
      AIE.useLock(%49, Release, 0, 0)
      AIE.useLock(%46, Release, 0, 0)
      AIE.useLock(%47, Release, 1, 0)
    }
    AIE.end
  }

  AIE.flow(%43, DMA : 0, %45, DMA : 0)
  AIE.flow(%43, DMA : 1, %45, DMA : 1)
  AIE.flow(%45, DMA : 0, %43, DMA : 0)
 
  AIE.flow(%29, DMA : 0, %31, DMA : 0)
  AIE.flow(%29, DMA : 1, %31, DMA : 1)
  AIE.flow(%31, DMA : 0, %43, DMA : 1)
 
  AIE.flow(%15, DMA : 0, %17, DMA : 0)
  AIE.flow(%15, DMA : 1, %17, DMA : 1)
  AIE.flow(%3, DMA : 0, %29, DMA : 1)
 
  AIE.flow(%1, DMA : 0, %3, DMA : 0)
  AIE.flow(%1, DMA : 1, %3, DMA : 1)
  AIE.flow(%17, DMA : 0, %29, DMA : 0)
  
}
