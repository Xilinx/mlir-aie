; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target triple = "aie"

@of_1_buff_0 = external global [256 x float]
@of_1_buff_1 = external global [256 x float]
@of_1_buff_2 = external global [256 x float]
@of_1_buff_3 = external global [256 x float]
@of_2_buff_0 = external global [256 x float]
@of_2_buff_1 = external global [256 x float]

declare i8* @malloc(i64)

declare void @free(i8*)

declare void @debug_i32(i32)

declare void @llvm.aie.put.ms(i32, i32)

declare void @llvm.aie.put.wms(i32, i128)

declare void @llvm.aie.put.fms(i32, float)

declare i32 @llvm.aie.get.ss(i32)

declare i128 @llvm.aie.get.wss(i32)

declare float @llvm.aie.get.fss(i32)

declare void @llvm.aie.put.mcd(i384)

declare i384 @llvm.aie.get.scd()

declare void @llvm.aie.lock.acquire.reg(i32, i32)

declare void @llvm.aie.lock.release.reg(i32, i32)

declare void @stencil_2d_7point_fp32(float*, float*, float*, float*)

define void @core_7_1() {
  call void @llvm.aie.lock.acquire.reg(i32 62, i32 0)
  call void @llvm.aie.lock.acquire.reg(i32 48, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 49, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 50, i32 1)
  call void @llvm.aie.lock.acquire.reg(i32 52, i32 0)
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_0 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_1 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_1_buff_2 to i64), i64 31), i64 0))
  call void @llvm.assume(i1 icmp eq (i64 and (i64 ptrtoint ([256 x float]* @of_2_buff_0 to i64), i64 31), i64 0))
  call void @stencil_2d_7point_fp32(float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_0, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_1, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_1_buff_2, i32 0, i32 0), float* getelementptr inbounds ([256 x float], [256 x float]* @of_2_buff_0, i32 0, i32 0))
  call void @llvm.aie.lock.release.reg(i32 48, i32 0)
  call void @llvm.aie.lock.release.reg(i32 52, i32 1)
  call void @llvm.aie.lock.release.reg(i32 62, i32 0)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 ) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
