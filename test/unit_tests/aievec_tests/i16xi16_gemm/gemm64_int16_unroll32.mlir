// REQUIRES: valid_xchess_license
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" --convert-vector-to-aievec="aie-target=aie2" -lower-affine | aie-translate -aie2=true --aievec-to-cpp -o gen_aie-ml.cc
// RUN: xchesscc -f -g +s -p me -P %aietools/data/aie_ml/lib/ +w work +o work -I%S -I. -c %S/kernel.cc -o kernel.o
// RUN: xchesscc -f -g +s -p me -P %aietools/data/aie_ml/lib/ +w work +o work -I%S -I. %S/testbench.cc work/kernel.o
// RUN: cp -r %S/data . && xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out"

module {
  func.func @matmul(%arg0: memref<64x64xi16>, %arg1: memref<64x64xi16>, %arg2: memref<64x64xi16>) {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 step 32 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<64x64xi16>
          %1 = affine.load %arg1[%arg5, %arg4] : memref<64x64xi16>
          %2 = arith.muli %0, %1 : i16
          %3 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %4 = arith.addi %3, %2 : i16
          affine.store %4, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %5 = affine.load %arg0[%arg3, %arg5 + 1] : memref<64x64xi16>
          %6 = affine.load %arg1[%arg5 + 1, %arg4] : memref<64x64xi16>
          %7 = arith.muli %5, %6 : i16
          %8 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %9 = arith.addi %8, %7 : i16
          affine.store %9, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %10 = affine.load %arg0[%arg3, %arg5 + 2] : memref<64x64xi16>
          %11 = affine.load %arg1[%arg5 + 2, %arg4] : memref<64x64xi16>
          %12 = arith.muli %10, %11 : i16
          %13 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %14 = arith.addi %13, %12 : i16
          affine.store %14, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %15 = affine.load %arg0[%arg3, %arg5 + 3] : memref<64x64xi16>
          %16 = affine.load %arg1[%arg5 + 3, %arg4] : memref<64x64xi16>
          %17 = arith.muli %15, %16 : i16
          %18 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %19 = arith.addi %18, %17 : i16
          affine.store %19, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %20 = affine.load %arg0[%arg3, %arg5 + 4] : memref<64x64xi16>
          %21 = affine.load %arg1[%arg5 + 4, %arg4] : memref<64x64xi16>
          %22 = arith.muli %20, %21 : i16
          %23 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %24 = arith.addi %23, %22 : i16
          affine.store %24, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %25 = affine.load %arg0[%arg3, %arg5 + 5] : memref<64x64xi16>
          %26 = affine.load %arg1[%arg5 + 5, %arg4] : memref<64x64xi16>
          %27 = arith.muli %25, %26 : i16
          %28 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %29 = arith.addi %28, %27 : i16
          affine.store %29, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %30 = affine.load %arg0[%arg3, %arg5 + 6] : memref<64x64xi16>
          %31 = affine.load %arg1[%arg5 + 6, %arg4] : memref<64x64xi16>
          %32 = arith.muli %30, %31 : i16
          %33 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %34 = arith.addi %33, %32 : i16
          affine.store %34, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %35 = affine.load %arg0[%arg3, %arg5 + 7] : memref<64x64xi16>
          %36 = affine.load %arg1[%arg5 + 7, %arg4] : memref<64x64xi16>
          %37 = arith.muli %35, %36 : i16
          %38 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %39 = arith.addi %38, %37 : i16
          affine.store %39, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %40 = affine.load %arg0[%arg3, %arg5 + 8] : memref<64x64xi16>
          %41 = affine.load %arg1[%arg5 + 8, %arg4] : memref<64x64xi16>
          %42 = arith.muli %40, %41 : i16
          %43 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %44 = arith.addi %43, %42 : i16
          affine.store %44, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %45 = affine.load %arg0[%arg3, %arg5 + 9] : memref<64x64xi16>
          %46 = affine.load %arg1[%arg5 + 9, %arg4] : memref<64x64xi16>
          %47 = arith.muli %45, %46 : i16
          %48 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %49 = arith.addi %48, %47 : i16
          affine.store %49, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %50 = affine.load %arg0[%arg3, %arg5 + 10] : memref<64x64xi16>
          %51 = affine.load %arg1[%arg5 + 10, %arg4] : memref<64x64xi16>
          %52 = arith.muli %50, %51 : i16
          %53 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %54 = arith.addi %53, %52 : i16
          affine.store %54, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %55 = affine.load %arg0[%arg3, %arg5 + 11] : memref<64x64xi16>
          %56 = affine.load %arg1[%arg5 + 11, %arg4] : memref<64x64xi16>
          %57 = arith.muli %55, %56 : i16
          %58 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %59 = arith.addi %58, %57 : i16
          affine.store %59, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %60 = affine.load %arg0[%arg3, %arg5 + 12] : memref<64x64xi16>
          %61 = affine.load %arg1[%arg5 + 12, %arg4] : memref<64x64xi16>
          %62 = arith.muli %60, %61 : i16
          %63 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %64 = arith.addi %63, %62 : i16
          affine.store %64, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %65 = affine.load %arg0[%arg3, %arg5 + 13] : memref<64x64xi16>
          %66 = affine.load %arg1[%arg5 + 13, %arg4] : memref<64x64xi16>
          %67 = arith.muli %65, %66 : i16
          %68 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %69 = arith.addi %68, %67 : i16
          affine.store %69, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %70 = affine.load %arg0[%arg3, %arg5 + 14] : memref<64x64xi16>
          %71 = affine.load %arg1[%arg5 + 14, %arg4] : memref<64x64xi16>
          %72 = arith.muli %70, %71 : i16
          %73 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %74 = arith.addi %73, %72 : i16
          affine.store %74, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %75 = affine.load %arg0[%arg3, %arg5 + 15] : memref<64x64xi16>
          %76 = affine.load %arg1[%arg5 + 15, %arg4] : memref<64x64xi16>
          %77 = arith.muli %75, %76 : i16
          %78 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %79 = arith.addi %78, %77 : i16
          affine.store %79, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %80 = affine.load %arg0[%arg3, %arg5 + 16] : memref<64x64xi16>
          %81 = affine.load %arg1[%arg5 + 16, %arg4] : memref<64x64xi16>
          %82 = arith.muli %80, %81 : i16
          %83 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %84 = arith.addi %83, %82 : i16
          affine.store %84, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %85 = affine.load %arg0[%arg3, %arg5 + 17] : memref<64x64xi16>
          %86 = affine.load %arg1[%arg5 + 17, %arg4] : memref<64x64xi16>
          %87 = arith.muli %85, %86 : i16
          %88 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %89 = arith.addi %88, %87 : i16
          affine.store %89, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %90 = affine.load %arg0[%arg3, %arg5 + 18] : memref<64x64xi16>
          %91 = affine.load %arg1[%arg5 + 18, %arg4] : memref<64x64xi16>
          %92 = arith.muli %90, %91 : i16
          %93 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %94 = arith.addi %93, %92 : i16
          affine.store %94, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %95 = affine.load %arg0[%arg3, %arg5 + 19] : memref<64x64xi16>
          %96 = affine.load %arg1[%arg5 + 19, %arg4] : memref<64x64xi16>
          %97 = arith.muli %95, %96 : i16
          %98 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %99 = arith.addi %98, %97 : i16
          affine.store %99, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %100 = affine.load %arg0[%arg3, %arg5 + 20] : memref<64x64xi16>
          %101 = affine.load %arg1[%arg5 + 20, %arg4] : memref<64x64xi16>
          %102 = arith.muli %100, %101 : i16
          %103 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %104 = arith.addi %103, %102 : i16
          affine.store %104, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %105 = affine.load %arg0[%arg3, %arg5 + 21] : memref<64x64xi16>
          %106 = affine.load %arg1[%arg5 + 21, %arg4] : memref<64x64xi16>
          %107 = arith.muli %105, %106 : i16
          %108 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %109 = arith.addi %108, %107 : i16
          affine.store %109, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %110 = affine.load %arg0[%arg3, %arg5 + 22] : memref<64x64xi16>
          %111 = affine.load %arg1[%arg5 + 22, %arg4] : memref<64x64xi16>
          %112 = arith.muli %110, %111 : i16
          %113 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %114 = arith.addi %113, %112 : i16
          affine.store %114, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %115 = affine.load %arg0[%arg3, %arg5 + 23] : memref<64x64xi16>
          %116 = affine.load %arg1[%arg5 + 23, %arg4] : memref<64x64xi16>
          %117 = arith.muli %115, %116 : i16
          %118 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %119 = arith.addi %118, %117 : i16
          affine.store %119, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %120 = affine.load %arg0[%arg3, %arg5 + 24] : memref<64x64xi16>
          %121 = affine.load %arg1[%arg5 + 24, %arg4] : memref<64x64xi16>
          %122 = arith.muli %120, %121 : i16
          %123 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %124 = arith.addi %123, %122 : i16
          affine.store %124, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %125 = affine.load %arg0[%arg3, %arg5 + 25] : memref<64x64xi16>
          %126 = affine.load %arg1[%arg5 + 25, %arg4] : memref<64x64xi16>
          %127 = arith.muli %125, %126 : i16
          %128 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %129 = arith.addi %128, %127 : i16
          affine.store %129, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %130 = affine.load %arg0[%arg3, %arg5 + 26] : memref<64x64xi16>
          %131 = affine.load %arg1[%arg5 + 26, %arg4] : memref<64x64xi16>
          %132 = arith.muli %130, %131 : i16
          %133 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %134 = arith.addi %133, %132 : i16
          affine.store %134, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %135 = affine.load %arg0[%arg3, %arg5 + 27] : memref<64x64xi16>
          %136 = affine.load %arg1[%arg5 + 27, %arg4] : memref<64x64xi16>
          %137 = arith.muli %135, %136 : i16
          %138 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %139 = arith.addi %138, %137 : i16
          affine.store %139, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %140 = affine.load %arg0[%arg3, %arg5 + 28] : memref<64x64xi16>
          %141 = affine.load %arg1[%arg5 + 28, %arg4] : memref<64x64xi16>
          %142 = arith.muli %140, %141 : i16
          %143 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %144 = arith.addi %143, %142 : i16
          affine.store %144, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %145 = affine.load %arg0[%arg3, %arg5 + 29] : memref<64x64xi16>
          %146 = affine.load %arg1[%arg5 + 29, %arg4] : memref<64x64xi16>
          %147 = arith.muli %145, %146 : i16
          %148 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %149 = arith.addi %148, %147 : i16
          affine.store %149, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %150 = affine.load %arg0[%arg3, %arg5 + 30] : memref<64x64xi16>
          %151 = affine.load %arg1[%arg5 + 30, %arg4] : memref<64x64xi16>
          %152 = arith.muli %150, %151 : i16
          %153 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %154 = arith.addi %153, %152 : i16
          affine.store %154, %arg2[%arg3, %arg4] : memref<64x64xi16>
          %155 = affine.load %arg0[%arg3, %arg5 + 31] : memref<64x64xi16>
          %156 = affine.load %arg1[%arg5 + 31, %arg4] : memref<64x64xi16>
          %157 = arith.muli %155, %156 : i16
          %158 = affine.load %arg2[%arg3, %arg4] : memref<64x64xi16>
          %159 = arith.addi %158, %157 : i16
          affine.store %159, %arg2[%arg3, %arg4] : memref<64x64xi16>
        }
      }
    }
    return
  }
}
