; ModuleID = '/home/muhaawad/git/amd/iron/mlir-aie/test/npu-xrt/vec_vec_add_objfifo_init/aie2.mlir.prj/input.llpeanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2"

@in1_buff_0 = local_unnamed_addr global [256 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79, i32 80, i32 81, i32 82, i32 83, i32 84, i32 85, i32 86, i32 87, i32 88, i32 89, i32 90, i32 91, i32 92, i32 93, i32 94, i32 95, i32 96, i32 97, i32 98, i32 99, i32 100, i32 101, i32 102, i32 103, i32 104, i32 105, i32 106, i32 107, i32 108, i32 109, i32 110, i32 111, i32 112, i32 113, i32 114, i32 115, i32 116, i32 117, i32 118, i32 119, i32 120, i32 121, i32 122, i32 123, i32 124, i32 125, i32 126, i32 127, i32 128, i32 129, i32 130, i32 131, i32 132, i32 133, i32 134, i32 135, i32 136, i32 137, i32 138, i32 139, i32 140, i32 141, i32 142, i32 143, i32 144, i32 145, i32 146, i32 147, i32 148, i32 149, i32 150, i32 151, i32 152, i32 153, i32 154, i32 155, i32 156, i32 157, i32 158, i32 159, i32 160, i32 161, i32 162, i32 163, i32 164, i32 165, i32 166, i32 167, i32 168, i32 169, i32 170, i32 171, i32 172, i32 173, i32 174, i32 175, i32 176, i32 177, i32 178, i32 179, i32 180, i32 181, i32 182, i32 183, i32 184, i32 185, i32 186, i32 187, i32 188, i32 189, i32 190, i32 191, i32 192, i32 193, i32 194, i32 195, i32 196, i32 197, i32 198, i32 199, i32 200, i32 201, i32 202, i32 203, i32 204, i32 205, i32 206, i32 207, i32 208, i32 209, i32 210, i32 211, i32 212, i32 213, i32 214, i32 215, i32 216, i32 217, i32 218, i32 219, i32 220, i32 221, i32 222, i32 223, i32 224, i32 225, i32 226, i32 227, i32 228, i32 229, i32 230, i32 231, i32 232, i32 233, i32 234, i32 235, i32 236, i32 237, i32 238, i32 239, i32 240, i32 241, i32 242, i32 243, i32 244, i32 245, i32 246, i32 247, i32 248, i32 249, i32 250, i32 251, i32 252, i32 253, i32 254, i32 255, i32 256]
@in1_cons_buff_0 = external local_unnamed_addr global [256 x i32]
@in2_cons_buff_1 = external local_unnamed_addr global [16 x i32]
@in2_cons_buff_0 = external local_unnamed_addr global [16 x i32]
@out_buff_1 = external local_unnamed_addr global [16 x i32]
@out_buff_0 = external local_unnamed_addr global [16 x i32]

; Function Attrs: nounwind
declare void @llvm.aie2.acquire(i32, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2.release(i32, i32) #0

; Function Attrs: nounwind
define void @core_0_2() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %183
  %2 = phi i64 [ 0, %0 ], [ %184, %183 ]
  tail call void @llvm.aie2.acquire(i32 49, i32 -1)
  br label %3

3:                                                ; preds = %1, %3
  %4 = phi i64 [ 0, %1 ], [ %181, %3 ]
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  %5 = shl nuw nsw i64 %4, 4
  %6 = trunc nuw i64 %5 to i20
  %7 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %6
  %8 = load i32, ptr %7, align 4
  %9 = load i32, ptr @in2_cons_buff_0, align 4
  %10 = add i32 %9, %8
  store i32 %10, ptr @out_buff_0, align 4
  %11 = trunc i64 %5 to i20
  %12 = or disjoint i20 %11, 1
  %13 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %12
  %14 = load i32, ptr %13, align 4
  %15 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 4), align 4
  %16 = add i32 %15, %14
  store i32 %16, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 4), align 4
  %17 = trunc i64 %5 to i20
  %18 = or disjoint i20 %17, 2
  %19 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %18
  %20 = load i32, ptr %19, align 4
  %21 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 8), align 4
  %22 = add i32 %21, %20
  store i32 %22, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 8), align 4
  %23 = trunc i64 %5 to i20
  %24 = or disjoint i20 %23, 3
  %25 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %24
  %26 = load i32, ptr %25, align 4
  %27 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 12), align 4
  %28 = add i32 %27, %26
  store i32 %28, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 12), align 4
  %29 = trunc i64 %5 to i20
  %30 = or disjoint i20 %29, 4
  %31 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %30
  %32 = load i32, ptr %31, align 4
  %33 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 16), align 4
  %34 = add i32 %33, %32
  store i32 %34, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 16), align 4
  %35 = trunc i64 %5 to i20
  %36 = or disjoint i20 %35, 5
  %37 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %36
  %38 = load i32, ptr %37, align 4
  %39 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 20), align 4
  %40 = add i32 %39, %38
  store i32 %40, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 20), align 4
  %41 = trunc i64 %5 to i20
  %42 = or disjoint i20 %41, 6
  %43 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %42
  %44 = load i32, ptr %43, align 4
  %45 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 24), align 4
  %46 = add i32 %45, %44
  store i32 %46, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 24), align 4
  %47 = trunc i64 %5 to i20
  %48 = or disjoint i20 %47, 7
  %49 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %48
  %50 = load i32, ptr %49, align 4
  %51 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 28), align 4
  %52 = add i32 %51, %50
  store i32 %52, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 28), align 4
  %53 = trunc i64 %5 to i20
  %54 = or disjoint i20 %53, 8
  %55 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %54
  %56 = load i32, ptr %55, align 4
  %57 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 32), align 4
  %58 = add i32 %57, %56
  store i32 %58, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 32), align 4
  %59 = trunc i64 %5 to i20
  %60 = or disjoint i20 %59, 9
  %61 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %60
  %62 = load i32, ptr %61, align 4
  %63 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 36), align 4
  %64 = add i32 %63, %62
  store i32 %64, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 36), align 4
  %65 = trunc i64 %5 to i20
  %66 = or disjoint i20 %65, 10
  %67 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %66
  %68 = load i32, ptr %67, align 4
  %69 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 40), align 4
  %70 = add i32 %69, %68
  store i32 %70, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 40), align 4
  %71 = trunc i64 %5 to i20
  %72 = or disjoint i20 %71, 11
  %73 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %72
  %74 = load i32, ptr %73, align 4
  %75 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 44), align 4
  %76 = add i32 %75, %74
  store i32 %76, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 44), align 4
  %77 = trunc i64 %5 to i20
  %78 = or disjoint i20 %77, 12
  %79 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %78
  %80 = load i32, ptr %79, align 4
  %81 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 48), align 4
  %82 = add i32 %81, %80
  store i32 %82, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 48), align 4
  %83 = trunc i64 %5 to i20
  %84 = or disjoint i20 %83, 13
  %85 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %84
  %86 = load i32, ptr %85, align 4
  %87 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 52), align 4
  %88 = add i32 %87, %86
  store i32 %88, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 52), align 4
  %89 = trunc i64 %5 to i20
  %90 = or disjoint i20 %89, 14
  %91 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %90
  %92 = load i32, ptr %91, align 4
  %93 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 56), align 4
  %94 = add i32 %93, %92
  store i32 %94, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 56), align 4
  %95 = trunc i64 %5 to i20
  %96 = or disjoint i20 %95, 15
  %97 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %96
  %98 = load i32, ptr %97, align 4
  %99 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_0, i20 60), align 4
  %100 = add i32 %99, %98
  store i32 %100, ptr getelementptr inbounds (i8, ptr @out_buff_0, i20 60), align 4
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  tail call void @llvm.aie2.acquire(i32 51, i32 -1)
  tail call void @llvm.aie2.acquire(i32 52, i32 -1)
  %101 = or disjoint i20 %6, 16
  %102 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %101
  %103 = load i32, ptr %102, align 4
  %104 = load i32, ptr @in2_cons_buff_1, align 4
  %105 = add i32 %104, %103
  store i32 %105, ptr @out_buff_1, align 4
  %106 = or disjoint i20 %11, 17
  %107 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %106
  %108 = load i32, ptr %107, align 4
  %109 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 4), align 4
  %110 = add i32 %109, %108
  store i32 %110, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 4), align 4
  %111 = or disjoint i20 %17, 18
  %112 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %111
  %113 = load i32, ptr %112, align 4
  %114 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 8), align 4
  %115 = add i32 %114, %113
  store i32 %115, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 8), align 4
  %116 = or disjoint i20 %23, 19
  %117 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %116
  %118 = load i32, ptr %117, align 4
  %119 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 12), align 4
  %120 = add i32 %119, %118
  store i32 %120, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 12), align 4
  %121 = or disjoint i20 %29, 20
  %122 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %121
  %123 = load i32, ptr %122, align 4
  %124 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 16), align 4
  %125 = add i32 %124, %123
  store i32 %125, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 16), align 4
  %126 = or disjoint i20 %35, 21
  %127 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %126
  %128 = load i32, ptr %127, align 4
  %129 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 20), align 4
  %130 = add i32 %129, %128
  store i32 %130, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 20), align 4
  %131 = or disjoint i20 %41, 22
  %132 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %131
  %133 = load i32, ptr %132, align 4
  %134 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 24), align 4
  %135 = add i32 %134, %133
  store i32 %135, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 24), align 4
  %136 = or disjoint i20 %47, 23
  %137 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %136
  %138 = load i32, ptr %137, align 4
  %139 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 28), align 4
  %140 = add i32 %139, %138
  store i32 %140, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 28), align 4
  %141 = or disjoint i20 %53, 24
  %142 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %141
  %143 = load i32, ptr %142, align 4
  %144 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 32), align 4
  %145 = add i32 %144, %143
  store i32 %145, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 32), align 4
  %146 = or disjoint i20 %59, 25
  %147 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %146
  %148 = load i32, ptr %147, align 4
  %149 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 36), align 4
  %150 = add i32 %149, %148
  store i32 %150, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 36), align 4
  %151 = or disjoint i20 %65, 26
  %152 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %151
  %153 = load i32, ptr %152, align 4
  %154 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 40), align 4
  %155 = add i32 %154, %153
  store i32 %155, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 40), align 4
  %156 = or disjoint i20 %71, 27
  %157 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %156
  %158 = load i32, ptr %157, align 4
  %159 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 44), align 4
  %160 = add i32 %159, %158
  store i32 %160, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 44), align 4
  %161 = or disjoint i20 %77, 28
  %162 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %161
  %163 = load i32, ptr %162, align 4
  %164 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 48), align 4
  %165 = add i32 %164, %163
  store i32 %165, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 48), align 4
  %166 = or disjoint i20 %83, 29
  %167 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %166
  %168 = load i32, ptr %167, align 4
  %169 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 52), align 4
  %170 = add i32 %169, %168
  store i32 %170, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 52), align 4
  %171 = or disjoint i20 %89, 30
  %172 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %171
  %173 = load i32, ptr %172, align 4
  %174 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 56), align 4
  %175 = add i32 %174, %173
  store i32 %175, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 56), align 4
  %176 = or disjoint i20 %95, 31
  %177 = getelementptr inbounds i32, ptr @in1_cons_buff_0, i20 %176
  %178 = load i32, ptr %177, align 4
  %179 = load i32, ptr getelementptr inbounds (i8, ptr @in2_cons_buff_1, i20 60), align 4
  %180 = add i32 %179, %178
  store i32 %180, ptr getelementptr inbounds (i8, ptr @out_buff_1, i20 60), align 4
  tail call void @llvm.aie2.release(i32 50, i32 1)
  tail call void @llvm.aie2.release(i32 53, i32 1)
  %181 = add nuw nsw i64 %4, 2
  %182 = icmp ult i64 %4, 14
  br i1 %182, label %3, label %183

183:                                              ; preds = %3
  tail call void @llvm.aie2.release(i32 48, i32 1)
  %184 = add nuw nsw i64 %2, 1
  %.not = icmp eq i64 %184, 9223372036854775807
  br i1 %.not, label %185, label %1

185:                                              ; preds = %183
  ret void
}

attributes #0 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
