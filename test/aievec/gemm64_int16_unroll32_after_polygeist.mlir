// RUN: aie-opt %s -affine-super-vectorize="virtual-vector-size=32" -aieml=true --aie-vectorize -canonicalize | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @matmul(%arg0: memref<?x64xi16>, %arg1: memref<?x64xi16>, %arg2: memref<?x64xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 64 {
        affine.for %arg5 = 0 to 64 step 32 {
          %0 = affine.load %arg0[%arg3, %arg5] : memref<?x64xi16>
          %1 = arith.extsi %0 : i16 to i32
          %2 = affine.load %arg1[%arg5, %arg4] : memref<?x64xi16>
          %3 = arith.extsi %2 : i16 to i32
          %4 = arith.muli %1, %3 : i32
          %5 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %6 = arith.trunci %4 : i32 to i16
          %7 = arith.addi %5, %6 : i16
          affine.store %7, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %8 = affine.load %arg0[%arg3, %arg5 + 1] : memref<?x64xi16>
          %9 = arith.extsi %8 : i16 to i32
          %10 = affine.load %arg1[%arg5 + 1, %arg4] : memref<?x64xi16>
          %11 = arith.extsi %10 : i16 to i32
          %12 = arith.muli %9, %11 : i32
          %13 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %14 = arith.trunci %12 : i32 to i16
          %15 = arith.addi %13, %14 : i16
          affine.store %15, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %16 = affine.load %arg0[%arg3, %arg5 + 2] : memref<?x64xi16>
          %17 = arith.extsi %16 : i16 to i32
          %18 = affine.load %arg1[%arg5 + 2, %arg4] : memref<?x64xi16>
          %19 = arith.extsi %18 : i16 to i32
          %20 = arith.muli %17, %19 : i32
          %21 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %22 = arith.trunci %20 : i32 to i16
          %23 = arith.addi %21, %22 : i16
          affine.store %23, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %24 = affine.load %arg0[%arg3, %arg5 + 3] : memref<?x64xi16>
          %25 = arith.extsi %24 : i16 to i32
          %26 = affine.load %arg1[%arg5 + 3, %arg4] : memref<?x64xi16>
          %27 = arith.extsi %26 : i16 to i32
          %28 = arith.muli %25, %27 : i32
          %29 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %30 = arith.trunci %28 : i32 to i16
          %31 = arith.addi %29, %30 : i16
          affine.store %31, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %32 = affine.load %arg0[%arg3, %arg5 + 4] : memref<?x64xi16>
          %33 = arith.extsi %32 : i16 to i32
          %34 = affine.load %arg1[%arg5 + 4, %arg4] : memref<?x64xi16>
          %35 = arith.extsi %34 : i16 to i32
          %36 = arith.muli %33, %35 : i32
          %37 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %38 = arith.trunci %36 : i32 to i16
          %39 = arith.addi %37, %38 : i16
          affine.store %39, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %40 = affine.load %arg0[%arg3, %arg5 + 5] : memref<?x64xi16>
          %41 = arith.extsi %40 : i16 to i32
          %42 = affine.load %arg1[%arg5 + 5, %arg4] : memref<?x64xi16>
          %43 = arith.extsi %42 : i16 to i32
          %44 = arith.muli %41, %43 : i32
          %45 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %46 = arith.trunci %44 : i32 to i16
          %47 = arith.addi %45, %46 : i16
          affine.store %47, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %48 = affine.load %arg0[%arg3, %arg5 + 6] : memref<?x64xi16>
          %49 = arith.extsi %48 : i16 to i32
          %50 = affine.load %arg1[%arg5 + 6, %arg4] : memref<?x64xi16>
          %51 = arith.extsi %50 : i16 to i32
          %52 = arith.muli %49, %51 : i32
          %53 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %54 = arith.trunci %52 : i32 to i16
          %55 = arith.addi %53, %54 : i16
          affine.store %55, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %56 = affine.load %arg0[%arg3, %arg5 + 7] : memref<?x64xi16>
          %57 = arith.extsi %56 : i16 to i32
          %58 = affine.load %arg1[%arg5 + 7, %arg4] : memref<?x64xi16>
          %59 = arith.extsi %58 : i16 to i32
          %60 = arith.muli %57, %59 : i32
          %61 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %62 = arith.trunci %60 : i32 to i16
          %63 = arith.addi %61, %62 : i16
          affine.store %63, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %64 = affine.load %arg0[%arg3, %arg5 + 8] : memref<?x64xi16>
          %65 = arith.extsi %64 : i16 to i32
          %66 = affine.load %arg1[%arg5 + 8, %arg4] : memref<?x64xi16>
          %67 = arith.extsi %66 : i16 to i32
          %68 = arith.muli %65, %67 : i32
          %69 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %70 = arith.trunci %68 : i32 to i16
          %71 = arith.addi %69, %70 : i16
          affine.store %71, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %72 = affine.load %arg0[%arg3, %arg5 + 9] : memref<?x64xi16>
          %73 = arith.extsi %72 : i16 to i32
          %74 = affine.load %arg1[%arg5 + 9, %arg4] : memref<?x64xi16>
          %75 = arith.extsi %74 : i16 to i32
          %76 = arith.muli %73, %75 : i32
          %77 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %78 = arith.trunci %76 : i32 to i16
          %79 = arith.addi %77, %78 : i16
          affine.store %79, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %80 = affine.load %arg0[%arg3, %arg5 + 10] : memref<?x64xi16>
          %81 = arith.extsi %80 : i16 to i32
          %82 = affine.load %arg1[%arg5 + 10, %arg4] : memref<?x64xi16>
          %83 = arith.extsi %82 : i16 to i32
          %84 = arith.muli %81, %83 : i32
          %85 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %86 = arith.trunci %84 : i32 to i16
          %87 = arith.addi %85, %86 : i16
          affine.store %87, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %88 = affine.load %arg0[%arg3, %arg5 + 11] : memref<?x64xi16>
          %89 = arith.extsi %88 : i16 to i32
          %90 = affine.load %arg1[%arg5 + 11, %arg4] : memref<?x64xi16>
          %91 = arith.extsi %90 : i16 to i32
          %92 = arith.muli %89, %91 : i32
          %93 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %94 = arith.trunci %92 : i32 to i16
          %95 = arith.addi %93, %94 : i16
          affine.store %95, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %96 = affine.load %arg0[%arg3, %arg5 + 12] : memref<?x64xi16>
          %97 = arith.extsi %96 : i16 to i32
          %98 = affine.load %arg1[%arg5 + 12, %arg4] : memref<?x64xi16>
          %99 = arith.extsi %98 : i16 to i32
          %100 = arith.muli %97, %99 : i32
          %101 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %102 = arith.trunci %100 : i32 to i16
          %103 = arith.addi %101, %102 : i16
          affine.store %103, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %104 = affine.load %arg0[%arg3, %arg5 + 13] : memref<?x64xi16>
          %105 = arith.extsi %104 : i16 to i32
          %106 = affine.load %arg1[%arg5 + 13, %arg4] : memref<?x64xi16>
          %107 = arith.extsi %106 : i16 to i32
          %108 = arith.muli %105, %107 : i32
          %109 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %110 = arith.trunci %108 : i32 to i16
          %111 = arith.addi %109, %110 : i16
          affine.store %111, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %112 = affine.load %arg0[%arg3, %arg5 + 14] : memref<?x64xi16>
          %113 = arith.extsi %112 : i16 to i32
          %114 = affine.load %arg1[%arg5 + 14, %arg4] : memref<?x64xi16>
          %115 = arith.extsi %114 : i16 to i32
          %116 = arith.muli %113, %115 : i32
          %117 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %118 = arith.trunci %116 : i32 to i16
          %119 = arith.addi %117, %118 : i16
          affine.store %119, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %120 = affine.load %arg0[%arg3, %arg5 + 15] : memref<?x64xi16>
          %121 = arith.extsi %120 : i16 to i32
          %122 = affine.load %arg1[%arg5 + 15, %arg4] : memref<?x64xi16>
          %123 = arith.extsi %122 : i16 to i32
          %124 = arith.muli %121, %123 : i32
          %125 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %126 = arith.trunci %124 : i32 to i16
          %127 = arith.addi %125, %126 : i16
          affine.store %127, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %128 = affine.load %arg0[%arg3, %arg5 + 16] : memref<?x64xi16>
          %129 = arith.extsi %128 : i16 to i32
          %130 = affine.load %arg1[%arg5 + 16, %arg4] : memref<?x64xi16>
          %131 = arith.extsi %130 : i16 to i32
          %132 = arith.muli %129, %131 : i32
          %133 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %134 = arith.trunci %132 : i32 to i16
          %135 = arith.addi %133, %134 : i16
          affine.store %135, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %136 = affine.load %arg0[%arg3, %arg5 + 17] : memref<?x64xi16>
          %137 = arith.extsi %136 : i16 to i32
          %138 = affine.load %arg1[%arg5 + 17, %arg4] : memref<?x64xi16>
          %139 = arith.extsi %138 : i16 to i32
          %140 = arith.muli %137, %139 : i32
          %141 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %142 = arith.trunci %140 : i32 to i16
          %143 = arith.addi %141, %142 : i16
          affine.store %143, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %144 = affine.load %arg0[%arg3, %arg5 + 18] : memref<?x64xi16>
          %145 = arith.extsi %144 : i16 to i32
          %146 = affine.load %arg1[%arg5 + 18, %arg4] : memref<?x64xi16>
          %147 = arith.extsi %146 : i16 to i32
          %148 = arith.muli %145, %147 : i32
          %149 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %150 = arith.trunci %148 : i32 to i16
          %151 = arith.addi %149, %150 : i16
          affine.store %151, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %152 = affine.load %arg0[%arg3, %arg5 + 19] : memref<?x64xi16>
          %153 = arith.extsi %152 : i16 to i32
          %154 = affine.load %arg1[%arg5 + 19, %arg4] : memref<?x64xi16>
          %155 = arith.extsi %154 : i16 to i32
          %156 = arith.muli %153, %155 : i32
          %157 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %158 = arith.trunci %156 : i32 to i16
          %159 = arith.addi %157, %158 : i16
          affine.store %159, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %160 = affine.load %arg0[%arg3, %arg5 + 20] : memref<?x64xi16>
          %161 = arith.extsi %160 : i16 to i32
          %162 = affine.load %arg1[%arg5 + 20, %arg4] : memref<?x64xi16>
          %163 = arith.extsi %162 : i16 to i32
          %164 = arith.muli %161, %163 : i32
          %165 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %166 = arith.trunci %164 : i32 to i16
          %167 = arith.addi %165, %166 : i16
          affine.store %167, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %168 = affine.load %arg0[%arg3, %arg5 + 21] : memref<?x64xi16>
          %169 = arith.extsi %168 : i16 to i32
          %170 = affine.load %arg1[%arg5 + 21, %arg4] : memref<?x64xi16>
          %171 = arith.extsi %170 : i16 to i32
          %172 = arith.muli %169, %171 : i32
          %173 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %174 = arith.trunci %172 : i32 to i16
          %175 = arith.addi %173, %174 : i16
          affine.store %175, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %176 = affine.load %arg0[%arg3, %arg5 + 22] : memref<?x64xi16>
          %177 = arith.extsi %176 : i16 to i32
          %178 = affine.load %arg1[%arg5 + 22, %arg4] : memref<?x64xi16>
          %179 = arith.extsi %178 : i16 to i32
          %180 = arith.muli %177, %179 : i32
          %181 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %182 = arith.trunci %180 : i32 to i16
          %183 = arith.addi %181, %182 : i16
          affine.store %183, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %184 = affine.load %arg0[%arg3, %arg5 + 23] : memref<?x64xi16>
          %185 = arith.extsi %184 : i16 to i32
          %186 = affine.load %arg1[%arg5 + 23, %arg4] : memref<?x64xi16>
          %187 = arith.extsi %186 : i16 to i32
          %188 = arith.muli %185, %187 : i32
          %189 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %190 = arith.trunci %188 : i32 to i16
          %191 = arith.addi %189, %190 : i16
          affine.store %191, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %192 = affine.load %arg0[%arg3, %arg5 + 24] : memref<?x64xi16>
          %193 = arith.extsi %192 : i16 to i32
          %194 = affine.load %arg1[%arg5 + 24, %arg4] : memref<?x64xi16>
          %195 = arith.extsi %194 : i16 to i32
          %196 = arith.muli %193, %195 : i32
          %197 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %198 = arith.trunci %196 : i32 to i16
          %199 = arith.addi %197, %198 : i16
          affine.store %199, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %200 = affine.load %arg0[%arg3, %arg5 + 25] : memref<?x64xi16>
          %201 = arith.extsi %200 : i16 to i32
          %202 = affine.load %arg1[%arg5 + 25, %arg4] : memref<?x64xi16>
          %203 = arith.extsi %202 : i16 to i32
          %204 = arith.muli %201, %203 : i32
          %205 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %206 = arith.trunci %204 : i32 to i16
          %207 = arith.addi %205, %206 : i16
          affine.store %207, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %208 = affine.load %arg0[%arg3, %arg5 + 26] : memref<?x64xi16>
          %209 = arith.extsi %208 : i16 to i32
          %210 = affine.load %arg1[%arg5 + 26, %arg4] : memref<?x64xi16>
          %211 = arith.extsi %210 : i16 to i32
          %212 = arith.muli %209, %211 : i32
          %213 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %214 = arith.trunci %212 : i32 to i16
          %215 = arith.addi %213, %214 : i16
          affine.store %215, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %216 = affine.load %arg0[%arg3, %arg5 + 27] : memref<?x64xi16>
          %217 = arith.extsi %216 : i16 to i32
          %218 = affine.load %arg1[%arg5 + 27, %arg4] : memref<?x64xi16>
          %219 = arith.extsi %218 : i16 to i32
          %220 = arith.muli %217, %219 : i32
          %221 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %222 = arith.trunci %220 : i32 to i16
          %223 = arith.addi %221, %222 : i16
          affine.store %223, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %224 = affine.load %arg0[%arg3, %arg5 + 28] : memref<?x64xi16>
          %225 = arith.extsi %224 : i16 to i32
          %226 = affine.load %arg1[%arg5 + 28, %arg4] : memref<?x64xi16>
          %227 = arith.extsi %226 : i16 to i32
          %228 = arith.muli %225, %227 : i32
          %229 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %230 = arith.trunci %228 : i32 to i16
          %231 = arith.addi %229, %230 : i16
          affine.store %231, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %232 = affine.load %arg0[%arg3, %arg5 + 29] : memref<?x64xi16>
          %233 = arith.extsi %232 : i16 to i32
          %234 = affine.load %arg1[%arg5 + 29, %arg4] : memref<?x64xi16>
          %235 = arith.extsi %234 : i16 to i32
          %236 = arith.muli %233, %235 : i32
          %237 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %238 = arith.trunci %236 : i32 to i16
          %239 = arith.addi %237, %238 : i16
          affine.store %239, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %240 = affine.load %arg0[%arg3, %arg5 + 30] : memref<?x64xi16>
          %241 = arith.extsi %240 : i16 to i32
          %242 = affine.load %arg1[%arg5 + 30, %arg4] : memref<?x64xi16>
          %243 = arith.extsi %242 : i16 to i32
          %244 = arith.muli %241, %243 : i32
          %245 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %246 = arith.trunci %244 : i32 to i16
          %247 = arith.addi %245, %246 : i16
          affine.store %247, %arg2[%arg3, %arg4] : memref<?x64xi16>
          %248 = affine.load %arg0[%arg3, %arg5 + 31] : memref<?x64xi16>
          %249 = arith.extsi %248 : i16 to i32
          %250 = affine.load %arg1[%arg5 + 31, %arg4] : memref<?x64xi16>
          %251 = arith.extsi %250 : i16 to i32
          %252 = arith.muli %249, %251 : i32
          %253 = affine.load %arg2[%arg3, %arg4] : memref<?x64xi16>
          %254 = arith.trunci %252 : i32 to i16
          %255 = arith.addi %253, %254 : i16
          affine.store %255, %arg2[%arg3, %arg4] : memref<?x64xi16>
        }
      }
    }
    return
  }
}
// CHECK-LABEL: @matmul
// CHECK-SAME: %[[A0:[0-9a-zA-Z]*]]: memref<?x64xi16>
// CHECK-SAME: %[[A1:[0-9a-zA-Z]*]]: memref<?x64xi16>
// CHECK-SAME: %[[A2:[0-9a-zA-Z]*]]: memref<?x64xi16>
//      CHECK:    %[[C31:.*]] = arith.constant 31 : index
//      CHECK:    %[[C30:.*]] = arith.constant 30 : index
//      CHECK:    %[[C29:.*]] = arith.constant 29 : index
//      CHECK:    %[[C28:.*]] = arith.constant 28 : index
//      CHECK:    %[[C27:.*]] = arith.constant 27 : index
//      CHECK:    %[[C26:.*]] = arith.constant 26 : index
//      CHECK:    %[[C25:.*]] = arith.constant 25 : index
//      CHECK:    %[[C24:.*]] = arith.constant 24 : index
//      CHECK:    %[[C23:.*]] = arith.constant 23 : index
//      CHECK:    %[[C22:.*]] = arith.constant 22 : index
//      CHECK:    %[[C21:.*]] = arith.constant 21 : index
//      CHECK:    %[[C20:.*]] = arith.constant 20 : index
//      CHECK:    %[[C19:.*]] = arith.constant 19 : index
//      CHECK:    %[[C18:.*]] = arith.constant 18 : index
//      CHECK:    %[[C17:.*]] = arith.constant 17 : index
//      CHECK:    %[[C16:.*]] = arith.constant 16 : index
//      CHECK:    %[[C15:.*]] = arith.constant 15 : index
//      CHECK:    %[[C14:.*]] = arith.constant 14 : index
//      CHECK:    %[[C13:.*]] = arith.constant 13 : index
//      CHECK:    %[[C12:.*]] = arith.constant 12 : index
//      CHECK:    %[[C11:.*]] = arith.constant 11 : index
//      CHECK:    %[[C10:.*]] = arith.constant 10 : index
//      CHECK:    %[[C9:.*]] = arith.constant 9 : index
//      CHECK:    %[[C8:.*]] = arith.constant 8 : index
//      CHECK:    %[[C7:.*]] = arith.constant 7 : index
//      CHECK:    %[[C6:.*]] = arith.constant 6 : index
//      CHECK:    %[[C5:.*]] = arith.constant 5 : index
//      CHECK:    %[[C4:.*]] = arith.constant 4 : index
//      CHECK:    %[[C3:.*]] = arith.constant 3 : index
//      CHECK:    %[[C2:.*]] = arith.constant 2 : index
//      CHECK:    %[[C32:.*]] = arith.constant 32 : index
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[C64:.*]] = arith.constant 64 : index
//      CHECK:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK:    scf.for %[[A3:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C1:.*]] {
//      CHECK:      scf.for %[[A4:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C32:.*]] {
//      CHECK:        %[[T0:.*]] = aievec.upd %[[A2]][%[[A3:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:        %[[T1:.*]] = aievec.ups %[[T0:.*]] {shift = 0 : i8} : vector<32xi16>, vector<32xi32>
//      CHECK:        scf.for %[[A5:.*]] = %[[C0:.*]] to %[[C64:.*]] step %[[C32:.*]] {
//      CHECK:          %[[T2:.*]] = aievec.upd %[[A0]][%[[A3:.*]], %[[A5:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T3:.*]] = aievec.upd %[[A1]][%[[A5:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T4:.*]] = aievec.broadcast %[[T2:.*]] {idx = 0 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T5:.*]] = aievec.mac_elem %[[T3:.*]], %[[T4:.*]], %[[T1:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T6:.*]] = arith.addi %[[A5:.*]], %[[C1:.*]] : index
//      CHECK:          %[[T7:.*]] = aievec.upd %[[A1]][%[[T6:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T8:.*]] = aievec.broadcast %[[T2:.*]] {idx = 1 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T9:.*]] = aievec.mac_elem %[[T7:.*]], %[[T8:.*]], %[[T5:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T10:.*]] = arith.addi %[[A5:.*]], %[[C2:.*]] : index
//      CHECK:          %[[T11:.*]] = aievec.upd %[[A1]][%[[T10:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T12:.*]] = aievec.broadcast %[[T2:.*]] {idx = 2 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T13:.*]] = aievec.mac_elem %[[T11:.*]], %[[T12:.*]], %[[T9:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T14:.*]] = arith.addi %[[A5:.*]], %[[C3:.*]] : index
//      CHECK:          %[[T15:.*]] = aievec.upd %[[A1]][%[[T14:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T16:.*]] = aievec.broadcast %[[T2:.*]] {idx = 3 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T17:.*]] = aievec.mac_elem %[[T15:.*]], %[[T16:.*]], %[[T13:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T18:.*]] = arith.addi %[[A5:.*]], %[[C4:.*]] : index
//      CHECK:          %[[T19:.*]] = aievec.upd %[[A1]][%[[T18:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T20:.*]] = aievec.broadcast %[[T2:.*]] {idx = 4 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T21:.*]] = aievec.mac_elem %[[T19:.*]], %[[T20:.*]], %[[T17:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T22:.*]] = arith.addi %[[A5:.*]], %[[C5:.*]] : index
//      CHECK:          %[[T23:.*]] = aievec.upd %[[A1]][%[[T22:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T24:.*]] = aievec.broadcast %[[T2:.*]] {idx = 5 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T25:.*]] = aievec.mac_elem %[[T23:.*]], %[[T24:.*]], %[[T21:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T26:.*]] = arith.addi %[[A5:.*]], %[[C6:.*]] : index
//      CHECK:          %[[T27:.*]] = aievec.upd %[[A1]][%[[T26:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T28:.*]] = aievec.broadcast %[[T2:.*]] {idx = 6 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T29:.*]] = aievec.mac_elem %[[T27:.*]], %[[T28:.*]], %[[T25:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T30:.*]] = arith.addi %[[A5:.*]], %[[C7:.*]] : index
//      CHECK:          %[[T31:.*]] = aievec.upd %[[A1]][%[[T30:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T32:.*]] = aievec.broadcast %[[T2:.*]] {idx = 7 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T33:.*]] = aievec.mac_elem %[[T31:.*]], %[[T32:.*]], %[[T29:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T34:.*]] = arith.addi %[[A5:.*]], %[[C8:.*]] : index
//      CHECK:          %[[T35:.*]] = aievec.upd %[[A1]][%[[T34:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T36:.*]] = aievec.broadcast %[[T2:.*]] {idx = 8 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T37:.*]] = aievec.mac_elem %[[T35:.*]], %[[T36:.*]], %[[T33:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T38:.*]] = arith.addi %[[A5:.*]], %[[C9:.*]] : index
//      CHECK:          %[[T39:.*]] = aievec.upd %[[A1]][%[[T38:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T40:.*]] = aievec.broadcast %[[T2:.*]] {idx = 9 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T41:.*]] = aievec.mac_elem %[[T39:.*]], %[[T40:.*]], %[[T37:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T42:.*]] = arith.addi %[[A5:.*]], %[[C10:.*]] : index
//      CHECK:          %[[T43:.*]] = aievec.upd %[[A1]][%[[T42:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T44:.*]] = aievec.broadcast %[[T2:.*]] {idx = 10 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T45:.*]] = aievec.mac_elem %[[T43:.*]], %[[T44:.*]], %[[T41:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T46:.*]] = arith.addi %[[A5:.*]], %[[C11:.*]] : index
//      CHECK:          %[[T47:.*]] = aievec.upd %[[A1]][%[[T46:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T48:.*]] = aievec.broadcast %[[T2:.*]] {idx = 11 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T49:.*]] = aievec.mac_elem %[[T47:.*]], %[[T48:.*]], %[[T45:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T50:.*]] = arith.addi %[[A5:.*]], %[[C12:.*]] : index
//      CHECK:          %[[T51:.*]] = aievec.upd %[[A1]][%[[T50:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T52:.*]] = aievec.broadcast %[[T2:.*]] {idx = 12 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T53:.*]] = aievec.mac_elem %[[T51:.*]], %[[T52:.*]], %[[T49:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T54:.*]] = arith.addi %[[A5:.*]], %[[C13:.*]] : index
//      CHECK:          %[[T55:.*]] = aievec.upd %[[A1]][%[[T54:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T56:.*]] = aievec.broadcast %[[T2:.*]] {idx = 13 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T57:.*]] = aievec.mac_elem %[[T55:.*]], %[[T56:.*]], %[[T53:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T58:.*]] = arith.addi %[[A5:.*]], %[[C14:.*]] : index
//      CHECK:          %[[T59:.*]] = aievec.upd %[[A1]][%[[T58:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T60:.*]] = aievec.broadcast %[[T2:.*]] {idx = 14 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T61:.*]] = aievec.mac_elem %[[T59:.*]], %[[T60:.*]], %[[T57:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T62:.*]] = arith.addi %[[A5:.*]], %[[C15:.*]] : index
//      CHECK:          %[[T63:.*]] = aievec.upd %[[A1]][%[[T62:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T64:.*]] = aievec.broadcast %[[T2:.*]] {idx = 15 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T65:.*]] = aievec.mac_elem %[[T63:.*]], %[[T64:.*]], %[[T61:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T66:.*]] = arith.addi %[[A5:.*]], %[[C16:.*]] : index
//      CHECK:          %[[T67:.*]] = aievec.upd %[[A1]][%[[T66:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T68:.*]] = aievec.broadcast %[[T2:.*]] {idx = 16 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T69:.*]] = aievec.mac_elem %[[T67:.*]], %[[T68:.*]], %[[T65:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T70:.*]] = arith.addi %[[A5:.*]], %[[C17:.*]] : index
//      CHECK:          %[[T71:.*]] = aievec.upd %[[A1]][%[[T70:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T72:.*]] = aievec.broadcast %[[T2:.*]] {idx = 17 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T73:.*]] = aievec.mac_elem %[[T71:.*]], %[[T72:.*]], %[[T69:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T74:.*]] = arith.addi %[[A5:.*]], %[[C18:.*]] : index
//      CHECK:          %[[T75:.*]] = aievec.upd %[[A1]][%[[T74:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T76:.*]] = aievec.broadcast %[[T2:.*]] {idx = 18 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T77:.*]] = aievec.mac_elem %[[T75:.*]], %[[T76:.*]], %[[T73:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T78:.*]] = arith.addi %[[A5:.*]], %[[C19:.*]] : index
//      CHECK:          %[[T79:.*]] = aievec.upd %[[A1]][%[[T78:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T80:.*]] = aievec.broadcast %[[T2:.*]] {idx = 19 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T81:.*]] = aievec.mac_elem %[[T79:.*]], %[[T80:.*]], %[[T77:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T82:.*]] = arith.addi %[[A5:.*]], %[[C20:.*]] : index
//      CHECK:          %[[T83:.*]] = aievec.upd %[[A1]][%[[T82:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T84:.*]] = aievec.broadcast %[[T2:.*]] {idx = 20 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T85:.*]] = aievec.mac_elem %[[T83:.*]], %[[T84:.*]], %[[T81:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T86:.*]] = arith.addi %[[A5:.*]], %[[C21:.*]] : index
//      CHECK:          %[[T87:.*]] = aievec.upd %[[A1]][%[[T86:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T88:.*]] = aievec.broadcast %[[T2:.*]] {idx = 21 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T89:.*]] = aievec.mac_elem %[[T87:.*]], %[[T88:.*]], %[[T85:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T90:.*]] = arith.addi %[[A5:.*]], %[[C22:.*]] : index
//      CHECK:          %[[T91:.*]] = aievec.upd %[[A1]][%90, %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T92:.*]] = aievec.broadcast %[[T2:.*]] {idx = 22 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T93:.*]] = aievec.mac_elem %[[T91:.*]], %[[T92:.*]], %[[T89:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T94:.*]] = arith.addi %[[A5:.*]], %[[C23:.*]] : index
//      CHECK:          %[[T95:.*]] = aievec.upd %[[A1]][%[[T94:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T96:.*]] = aievec.broadcast %[[T2:.*]] {idx = 23 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T97:.*]] = aievec.mac_elem %[[T95:.*]], %[[T96:.*]], %[[T93:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T98:.*]] = arith.addi %[[A5:.*]], %[[C24:.*]] : index
//      CHECK:          %[[T99:.*]] = aievec.upd %[[A1]][%[[T98:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T100:.*]] = aievec.broadcast %[[T2:.*]] {idx = 24 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T101:.*]] = aievec.mac_elem %[[T99:.*]], %[[T100:.*]], %[[T97:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T102:.*]] = arith.addi %[[A5:.*]], %[[C25:.*]] : index
//      CHECK:          %[[T103:.*]] = aievec.upd %[[A1]][%[[T102:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T104:.*]] = aievec.broadcast %[[T2:.*]] {idx = 25 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T105:.*]] = aievec.mac_elem %[[T103:.*]], %[[T104:.*]], %[[T101:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T106:.*]] = arith.addi %[[A5:.*]], %[[C26:.*]] : index
//      CHECK:          %[[T107:.*]] = aievec.upd %[[A1]][%[[T106:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T108:.*]] = aievec.broadcast %[[T2:.*]] {idx = 26 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T109:.*]] = aievec.mac_elem %[[T107:.*]], %[[T108:.*]], %[[T105:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T110:.*]] = arith.addi %[[A5:.*]], %[[C27:.*]] : index
//      CHECK:          %[[T111:.*]] = aievec.upd %[[A1]][%[[T110:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T112:.*]] = aievec.broadcast %[[T2:.*]] {idx = 27 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T113:.*]] = aievec.mac_elem %[[T111:.*]], %[[T112:.*]], %[[T109:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T114:.*]] = arith.addi %[[A5:.*]], %[[C28:.*]] : index
//      CHECK:          %[[T115:.*]] = aievec.upd %[[A1]][%[[T114:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T116:.*]] = aievec.broadcast %[[T2:.*]] {idx = 28 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T117:.*]] = aievec.mac_elem %[[T115:.*]], %[[T116:.*]], %[[T113:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T118:.*]] = arith.addi %[[A5:.*]], %[[C29:.*]] : index
//      CHECK:          %[[T119:.*]] = aievec.upd %[[A1]][%[[T118:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T120:.*]] = aievec.broadcast %[[T2:.*]] {idx = 29 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T121:.*]] = aievec.mac_elem %[[T119:.*]], %[[T120:.*]], %[[T117:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T122:.*]] = arith.addi %[[A5:.*]], %[[C30:.*]] : index
//      CHECK:          %[[T123:.*]] = aievec.upd %[[A1]][%[[T122:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T124:.*]] = aievec.broadcast %[[T2:.*]] {idx = 30 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T125:.*]] = aievec.mac_elem %[[T123:.*]], %[[T124:.*]], %[[T121:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T126:.*]] = arith.addi %[[A5:.*]], %[[C31:.*]] : index
//      CHECK:          %[[T127:.*]] = aievec.upd %[[A1]][%[[T126:.*]], %[[A4:.*]]] {index = 0 : i8, offset = 0 : i32} : memref<?x64xi16>, vector<32xi16>
//      CHECK:          %[[T128:.*]] = aievec.broadcast %[[T2:.*]] {idx = 31 : i8} : vector<32xi16>, vector<32xi16>
//      CHECK:          %[[T129:.*]] = aievec.mac_elem %[[T127:.*]], %[[T128:.*]], %[[T125:.*]] : vector<32xi16>, vector<32xi16>, vector<32xi32>
//      CHECK:          %[[T130:.*]] = aievec.srs %[[T129:.*]] {shift = 0 : i8} : vector<32xi32>, vector<32xi16>
//      CHECK:          vector.transfer_write %[[T130:.*]], %[[A2]][%[[A3:.*]], %[[A4:.*]]] {in_bounds = [true]} : vector<32xi16>, memref<?x64xi16>
