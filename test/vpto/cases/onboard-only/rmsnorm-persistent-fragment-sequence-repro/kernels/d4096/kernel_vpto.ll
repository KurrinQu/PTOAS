; ModuleID = 'ptoas.hivm.official.vector'
source_filename = "ptoas.hivm.official.vector"

; Unknown intrinsic
declare void @llvm.hivm.SET.FLAG.IMM(i64, i64, i64) #0

; Unknown intrinsic
declare void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6), ptr addrspace(1), i64, i64) #0

; Unknown intrinsic
declare void @llvm.hivm.WAIT.FLAG.IMM(i64, i64, i64) #0

; Unknown intrinsic
declare void @llvm.hivm.store.vfsimt.info(i64) #0

; Unknown intrinsic
declare void @llvm.hivm.WAIT.FLAG.REG(i64, i64, i64) #0

; Unknown intrinsic
declare i64 @llvm.hivm.GET.BLOCK.IDX() #0

; Unknown intrinsic
declare void @llvm.hivm.SET.FLAG.REG(i64, i64, i64) #0

; Unknown intrinsic
declare void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1), ptr addrspace(6), i64, i64) #0

; Unknown intrinsic
declare i32 @llvm.hivm.get.TID.X() #0

; Unknown intrinsic
declare i32 @llvm.hivm.get.laneID() #0

; Unknown intrinsic
declare float @llvm.hivm.redux.add.f32(float) #0

; Unknown intrinsic
declare void @llvm.hivm.sync.workitems() #0

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #1

define void @rmsnorm_d4096_kernel_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, float %4) #2 {
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) null, ptr addrspace(1) %1, i64 549755813904, i64 18014398509498368)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.store.vfsimt.info(i64 4295032960)
  call simt_entry void @rmsnorm_d4096_kernel_simt_0(ptr addrspace(6) null)
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %27, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 64
  br i1 %8, label %9, label %28

9:                                                ; preds = %6
  %10 = and i64 %7, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %10)
  %11 = mul i64 %7, 262144
  %12 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %13 = mul i64 %12, 4096
  %14 = add i64 %11, %13
  %15 = getelementptr float, ptr addrspace(1) %2, i64 %14
  %16 = mul i64 %10, 4096
  %17 = add i64 %16, 4096
  %18 = getelementptr float, ptr addrspace(6) null, i64 %17
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) %18, ptr addrspace(1) %15, i64 549755813904, i64 18014398509498368)
  call void @llvm.hivm.SET.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.store.vfsimt.info(i64 4295032960)
  call simt_entry void @rmsnorm_d4096_kernel_simt_1(i64 %16, ptr addrspace(6) null, float %4, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 5, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 5, i64 %10)
  %19 = mul i64 %10, 8
  %20 = getelementptr float, ptr addrspace(6) null, i64 %19
  %21 = mul i64 %7, 64
  %22 = add i64 %21, %12
  %23 = getelementptr float, ptr addrspace(1) %0, i64 %22
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %23, ptr addrspace(6) %20, i64 134217744, i64 4398046511108)
  %24 = add i64 %16, 12288
  %25 = getelementptr float, ptr addrspace(6) null, i64 %24
  %26 = getelementptr float, ptr addrspace(1) %3, i64 %14
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %26, ptr addrspace(6) %25, i64 549755813904, i64 18014398509498368)
  call void @llvm.hivm.SET.FLAG.REG(i64 5, i64 1, i64 %10)
  %27 = add i64 %7, 1
  br label %6

28:                                               ; preds = %6
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 1, i64 4, i64 1)
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @rmsnorm_d4096_kernel_simt_0(ptr addrspace(6) %0) #3 !annotation !8 !annotation !9 {
  %2 = call i32 @llvm.hivm.get.TID.X()
  %3 = mul i32 %2, 2
  %4 = sext i32 %3 to i64
  %5 = mul i64 %4, 4
  %6 = ptrtoint ptr addrspace(6) %0 to i64
  %7 = inttoptr i64 %6 to ptr addrspace(6)
  %8 = getelementptr i8, ptr addrspace(6) %7, i64 %5
  %9 = load <2 x float>, ptr addrspace(6) %8, align 8
  %10 = extractelement <2 x float> %9, i32 0
  %11 = extractelement <2 x float> %9, i32 1
  %12 = add i32 %3, 256
  %13 = sext i32 %12 to i64
  %14 = mul i64 %13, 4
  %15 = getelementptr i8, ptr addrspace(6) %7, i64 %14
  %16 = load <2 x float>, ptr addrspace(6) %15, align 8
  %17 = extractelement <2 x float> %16, i32 0
  %18 = extractelement <2 x float> %16, i32 1
  %19 = add i32 %3, 512
  %20 = sext i32 %19 to i64
  %21 = mul i64 %20, 4
  %22 = getelementptr i8, ptr addrspace(6) %7, i64 %21
  %23 = load <2 x float>, ptr addrspace(6) %22, align 8
  %24 = extractelement <2 x float> %23, i32 0
  %25 = extractelement <2 x float> %23, i32 1
  %26 = add i32 %3, 768
  %27 = sext i32 %26 to i64
  %28 = mul i64 %27, 4
  %29 = getelementptr i8, ptr addrspace(6) %7, i64 %28
  %30 = load <2 x float>, ptr addrspace(6) %29, align 8
  %31 = extractelement <2 x float> %30, i32 0
  %32 = extractelement <2 x float> %30, i32 1
  %33 = add i32 %3, 1024
  %34 = sext i32 %33 to i64
  %35 = mul i64 %34, 4
  %36 = getelementptr i8, ptr addrspace(6) %7, i64 %35
  %37 = load <2 x float>, ptr addrspace(6) %36, align 8
  %38 = extractelement <2 x float> %37, i32 0
  %39 = extractelement <2 x float> %37, i32 1
  %40 = add i32 %3, 1280
  %41 = sext i32 %40 to i64
  %42 = mul i64 %41, 4
  %43 = getelementptr i8, ptr addrspace(6) %7, i64 %42
  %44 = load <2 x float>, ptr addrspace(6) %43, align 8
  %45 = extractelement <2 x float> %44, i32 0
  %46 = extractelement <2 x float> %44, i32 1
  %47 = add i32 %3, 1536
  %48 = sext i32 %47 to i64
  %49 = mul i64 %48, 4
  %50 = getelementptr i8, ptr addrspace(6) %7, i64 %49
  %51 = load <2 x float>, ptr addrspace(6) %50, align 8
  %52 = extractelement <2 x float> %51, i32 0
  %53 = extractelement <2 x float> %51, i32 1
  %54 = add i32 %3, 1792
  %55 = sext i32 %54 to i64
  %56 = mul i64 %55, 4
  %57 = getelementptr i8, ptr addrspace(6) %7, i64 %56
  %58 = load <2 x float>, ptr addrspace(6) %57, align 8
  %59 = extractelement <2 x float> %58, i32 0
  %60 = extractelement <2 x float> %58, i32 1
  %61 = add i32 %3, 2048
  %62 = sext i32 %61 to i64
  %63 = mul i64 %62, 4
  %64 = getelementptr i8, ptr addrspace(6) %7, i64 %63
  %65 = load <2 x float>, ptr addrspace(6) %64, align 8
  %66 = extractelement <2 x float> %65, i32 0
  %67 = extractelement <2 x float> %65, i32 1
  %68 = add i32 %3, 2304
  %69 = sext i32 %68 to i64
  %70 = mul i64 %69, 4
  %71 = getelementptr i8, ptr addrspace(6) %7, i64 %70
  %72 = load <2 x float>, ptr addrspace(6) %71, align 8
  %73 = extractelement <2 x float> %72, i32 0
  %74 = extractelement <2 x float> %72, i32 1
  %75 = add i32 %3, 2560
  %76 = sext i32 %75 to i64
  %77 = mul i64 %76, 4
  %78 = getelementptr i8, ptr addrspace(6) %7, i64 %77
  %79 = load <2 x float>, ptr addrspace(6) %78, align 8
  %80 = extractelement <2 x float> %79, i32 0
  %81 = extractelement <2 x float> %79, i32 1
  %82 = add i32 %3, 2816
  %83 = sext i32 %82 to i64
  %84 = mul i64 %83, 4
  %85 = getelementptr i8, ptr addrspace(6) %7, i64 %84
  %86 = load <2 x float>, ptr addrspace(6) %85, align 8
  %87 = extractelement <2 x float> %86, i32 0
  %88 = extractelement <2 x float> %86, i32 1
  %89 = add i32 %3, 3072
  %90 = sext i32 %89 to i64
  %91 = mul i64 %90, 4
  %92 = getelementptr i8, ptr addrspace(6) %7, i64 %91
  %93 = load <2 x float>, ptr addrspace(6) %92, align 8
  %94 = extractelement <2 x float> %93, i32 0
  %95 = extractelement <2 x float> %93, i32 1
  %96 = add i32 %3, 3328
  %97 = sext i32 %96 to i64
  %98 = mul i64 %97, 4
  %99 = getelementptr i8, ptr addrspace(6) %7, i64 %98
  %100 = load <2 x float>, ptr addrspace(6) %99, align 8
  %101 = extractelement <2 x float> %100, i32 0
  %102 = extractelement <2 x float> %100, i32 1
  %103 = add i32 %3, 3584
  %104 = sext i32 %103 to i64
  %105 = mul i64 %104, 4
  %106 = getelementptr i8, ptr addrspace(6) %7, i64 %105
  %107 = load <2 x float>, ptr addrspace(6) %106, align 8
  %108 = extractelement <2 x float> %107, i32 0
  %109 = extractelement <2 x float> %107, i32 1
  %110 = add i32 %3, 3840
  %111 = sext i32 %110 to i64
  %112 = mul i64 %111, 4
  %113 = getelementptr i8, ptr addrspace(6) %7, i64 %112
  %114 = load <2 x float>, ptr addrspace(6) %113, align 8
  %115 = extractelement <2 x float> %114, i32 0
  %116 = extractelement <2 x float> %114, i32 1
  %117 = bitcast float %10 to i32
  %118 = bitcast float %11 to i32
  %119 = bitcast float %17 to i32
  %120 = bitcast float %18 to i32
  %121 = bitcast float %24 to i32
  %122 = bitcast float %25 to i32
  %123 = bitcast float %31 to i32
  %124 = bitcast float %32 to i32
  %125 = bitcast float %38 to i32
  %126 = bitcast float %39 to i32
  %127 = bitcast float %45 to i32
  %128 = bitcast float %46 to i32
  %129 = bitcast float %52 to i32
  %130 = bitcast float %53 to i32
  %131 = bitcast float %59 to i32
  %132 = bitcast float %60 to i32
  %133 = bitcast float %66 to i32
  %134 = bitcast float %67 to i32
  %135 = bitcast float %73 to i32
  %136 = bitcast float %74 to i32
  %137 = bitcast float %80 to i32
  %138 = bitcast float %81 to i32
  %139 = bitcast float %87 to i32
  %140 = bitcast float %88 to i32
  %141 = bitcast float %94 to i32
  %142 = bitcast float %95 to i32
  %143 = bitcast float %101 to i32
  %144 = bitcast float %102 to i32
  %145 = bitcast float %108 to i32
  %146 = bitcast float %109 to i32
  %147 = bitcast float %115 to i32
  %148 = bitcast float %116 to i32
  %149 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31},={TPER32},={TPER33},={TPER34},={TPER35},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"(i32 %117, i32 %118, i32 %119, i32 %120, i32 %121, i32 %122, i32 %123, i32 %124, i32 %125, i32 %126, i32 %127, i32 %128, i32 %129, i32 %130, i32 %131, i32 %132, i32 %133, i32 %134, i32 %135, i32 %136, i32 %137, i32 %138, i32 %139, i32 %140, i32 %141, i32 %142, i32 %143, i32 %144, i32 %145, i32 %146, i32 %147, i32 %148)
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @rmsnorm_d4096_kernel_simt_1(i64 %0, ptr addrspace(6) %1, float %2, i64 %3) #3 !annotation !8 !annotation !9 {
  %5 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31},={TPER32},={TPER33},={TPER34},={TPER35}"()
  %6 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 0
  %7 = bitcast i32 %6 to float
  %8 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 1
  %9 = bitcast i32 %8 to float
  %10 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 2
  %11 = bitcast i32 %10 to float
  %12 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 3
  %13 = bitcast i32 %12 to float
  %14 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 4
  %15 = bitcast i32 %14 to float
  %16 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 5
  %17 = bitcast i32 %16 to float
  %18 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 6
  %19 = bitcast i32 %18 to float
  %20 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 7
  %21 = bitcast i32 %20 to float
  %22 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 8
  %23 = bitcast i32 %22 to float
  %24 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 9
  %25 = bitcast i32 %24 to float
  %26 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 10
  %27 = bitcast i32 %26 to float
  %28 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 11
  %29 = bitcast i32 %28 to float
  %30 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 12
  %31 = bitcast i32 %30 to float
  %32 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 13
  %33 = bitcast i32 %32 to float
  %34 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 14
  %35 = bitcast i32 %34 to float
  %36 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 15
  %37 = bitcast i32 %36 to float
  %38 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 16
  %39 = bitcast i32 %38 to float
  %40 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 17
  %41 = bitcast i32 %40 to float
  %42 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 18
  %43 = bitcast i32 %42 to float
  %44 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 19
  %45 = bitcast i32 %44 to float
  %46 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 20
  %47 = bitcast i32 %46 to float
  %48 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 21
  %49 = bitcast i32 %48 to float
  %50 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 22
  %51 = bitcast i32 %50 to float
  %52 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 23
  %53 = bitcast i32 %52 to float
  %54 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 24
  %55 = bitcast i32 %54 to float
  %56 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 25
  %57 = bitcast i32 %56 to float
  %58 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 26
  %59 = bitcast i32 %58 to float
  %60 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 27
  %61 = bitcast i32 %60 to float
  %62 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 28
  %63 = bitcast i32 %62 to float
  %64 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 29
  %65 = bitcast i32 %64 to float
  %66 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 30
  %67 = bitcast i32 %66 to float
  %68 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %5, 31
  %69 = bitcast i32 %68 to float
  %70 = alloca float, i32 32, align 4
  %71 = alloca float, align 4
  %72 = call i32 @llvm.hivm.get.TID.X()
  %73 = mul i32 %72, 2
  %74 = sext i32 %73 to i64
  %75 = add i64 %0, %74
  %76 = add i64 %75, 4096
  %77 = mul i64 %76, 4
  %78 = ptrtoint ptr addrspace(6) %1 to i64
  %79 = inttoptr i64 %78 to ptr addrspace(6)
  %80 = getelementptr i8, ptr addrspace(6) %79, i64 %77
  %81 = load <2 x float>, ptr addrspace(6) %80, align 8
  store <2 x float> %81, ptr %70, align 8
  %82 = add i64 %0, 256
  %83 = add i64 %82, %74
  %84 = add i64 %83, 4096
  %85 = mul i64 %84, 4
  %86 = getelementptr i8, ptr addrspace(6) %79, i64 %85
  %87 = load <2 x float>, ptr addrspace(6) %86, align 8
  %88 = getelementptr i8, ptr %70, i32 8
  store <2 x float> %87, ptr %88, align 8
  %89 = add i64 %0, 512
  %90 = add i64 %89, %74
  %91 = add i64 %90, 4096
  %92 = mul i64 %91, 4
  %93 = getelementptr i8, ptr addrspace(6) %79, i64 %92
  %94 = load <2 x float>, ptr addrspace(6) %93, align 8
  %95 = getelementptr i8, ptr %70, i32 16
  store <2 x float> %94, ptr %95, align 8
  %96 = add i64 %0, 768
  %97 = add i64 %96, %74
  %98 = add i64 %97, 4096
  %99 = mul i64 %98, 4
  %100 = getelementptr i8, ptr addrspace(6) %79, i64 %99
  %101 = load <2 x float>, ptr addrspace(6) %100, align 8
  %102 = getelementptr i8, ptr %70, i32 24
  store <2 x float> %101, ptr %102, align 8
  %103 = add i64 %0, 1024
  %104 = add i64 %103, %74
  %105 = add i64 %104, 4096
  %106 = mul i64 %105, 4
  %107 = getelementptr i8, ptr addrspace(6) %79, i64 %106
  %108 = load <2 x float>, ptr addrspace(6) %107, align 8
  %109 = getelementptr i8, ptr %70, i32 32
  store <2 x float> %108, ptr %109, align 8
  %110 = add i64 %0, 1280
  %111 = add i64 %110, %74
  %112 = add i64 %111, 4096
  %113 = mul i64 %112, 4
  %114 = getelementptr i8, ptr addrspace(6) %79, i64 %113
  %115 = load <2 x float>, ptr addrspace(6) %114, align 8
  %116 = getelementptr i8, ptr %70, i32 40
  store <2 x float> %115, ptr %116, align 8
  %117 = add i64 %0, 1536
  %118 = add i64 %117, %74
  %119 = add i64 %118, 4096
  %120 = mul i64 %119, 4
  %121 = getelementptr i8, ptr addrspace(6) %79, i64 %120
  %122 = load <2 x float>, ptr addrspace(6) %121, align 8
  %123 = getelementptr i8, ptr %70, i32 48
  store <2 x float> %122, ptr %123, align 8
  %124 = add i64 %0, 1792
  %125 = add i64 %124, %74
  %126 = add i64 %125, 4096
  %127 = mul i64 %126, 4
  %128 = getelementptr i8, ptr addrspace(6) %79, i64 %127
  %129 = load <2 x float>, ptr addrspace(6) %128, align 8
  %130 = getelementptr i8, ptr %70, i32 56
  store <2 x float> %129, ptr %130, align 8
  %131 = add i64 %0, 2048
  %132 = add i64 %131, %74
  %133 = add i64 %132, 4096
  %134 = mul i64 %133, 4
  %135 = getelementptr i8, ptr addrspace(6) %79, i64 %134
  %136 = load <2 x float>, ptr addrspace(6) %135, align 8
  %137 = getelementptr i8, ptr %70, i32 64
  store <2 x float> %136, ptr %137, align 8
  %138 = add i64 %0, 2304
  %139 = add i64 %138, %74
  %140 = add i64 %139, 4096
  %141 = mul i64 %140, 4
  %142 = getelementptr i8, ptr addrspace(6) %79, i64 %141
  %143 = load <2 x float>, ptr addrspace(6) %142, align 8
  %144 = getelementptr i8, ptr %70, i32 72
  store <2 x float> %143, ptr %144, align 8
  %145 = add i64 %0, 2560
  %146 = add i64 %145, %74
  %147 = add i64 %146, 4096
  %148 = mul i64 %147, 4
  %149 = getelementptr i8, ptr addrspace(6) %79, i64 %148
  %150 = load <2 x float>, ptr addrspace(6) %149, align 8
  %151 = getelementptr i8, ptr %70, i32 80
  store <2 x float> %150, ptr %151, align 8
  %152 = add i64 %0, 2816
  %153 = add i64 %152, %74
  %154 = add i64 %153, 4096
  %155 = mul i64 %154, 4
  %156 = getelementptr i8, ptr addrspace(6) %79, i64 %155
  %157 = load <2 x float>, ptr addrspace(6) %156, align 8
  %158 = getelementptr i8, ptr %70, i32 88
  store <2 x float> %157, ptr %158, align 8
  %159 = add i64 %0, 3072
  %160 = add i64 %159, %74
  %161 = add i64 %160, 4096
  %162 = mul i64 %161, 4
  %163 = getelementptr i8, ptr addrspace(6) %79, i64 %162
  %164 = load <2 x float>, ptr addrspace(6) %163, align 8
  %165 = getelementptr i8, ptr %70, i32 96
  store <2 x float> %164, ptr %165, align 8
  %166 = add i64 %0, 3328
  %167 = add i64 %166, %74
  %168 = add i64 %167, 4096
  %169 = mul i64 %168, 4
  %170 = getelementptr i8, ptr addrspace(6) %79, i64 %169
  %171 = load <2 x float>, ptr addrspace(6) %170, align 8
  %172 = getelementptr i8, ptr %70, i32 104
  store <2 x float> %171, ptr %172, align 8
  %173 = add i64 %0, 3584
  %174 = add i64 %173, %74
  %175 = add i64 %174, 4096
  %176 = mul i64 %175, 4
  %177 = getelementptr i8, ptr addrspace(6) %79, i64 %176
  %178 = load <2 x float>, ptr addrspace(6) %177, align 8
  %179 = getelementptr i8, ptr %70, i32 112
  store <2 x float> %178, ptr %179, align 8
  %180 = add i64 %0, 3840
  %181 = add i64 %180, %74
  %182 = add i64 %181, 4096
  %183 = mul i64 %182, 4
  %184 = getelementptr i8, ptr addrspace(6) %79, i64 %183
  %185 = load <2 x float>, ptr addrspace(6) %184, align 8
  %186 = getelementptr i8, ptr %70, i32 120
  store <2 x float> %185, ptr %186, align 8
  store float 0.000000e+00, ptr %71, align 4
  %187 = load float, ptr %71, align 4
  %188 = load float, ptr %70, align 4
  %189 = fmul float %188, %188
  %190 = fadd float %187, %189
  store float %190, ptr %71, align 4
  %191 = load float, ptr %71, align 4
  %192 = getelementptr i8, ptr %70, i32 4
  %193 = load float, ptr %192, align 4
  %194 = fmul float %193, %193
  %195 = fadd float %191, %194
  store float %195, ptr %71, align 4
  %196 = load float, ptr %71, align 4
  %197 = load float, ptr %88, align 4
  %198 = fmul float %197, %197
  %199 = fadd float %196, %198
  store float %199, ptr %71, align 4
  %200 = load float, ptr %71, align 4
  %201 = getelementptr i8, ptr %70, i32 12
  %202 = load float, ptr %201, align 4
  %203 = fmul float %202, %202
  %204 = fadd float %200, %203
  store float %204, ptr %71, align 4
  %205 = load float, ptr %71, align 4
  %206 = load float, ptr %95, align 4
  %207 = fmul float %206, %206
  %208 = fadd float %205, %207
  store float %208, ptr %71, align 4
  %209 = load float, ptr %71, align 4
  %210 = getelementptr i8, ptr %70, i32 20
  %211 = load float, ptr %210, align 4
  %212 = fmul float %211, %211
  %213 = fadd float %209, %212
  store float %213, ptr %71, align 4
  %214 = load float, ptr %71, align 4
  %215 = load float, ptr %102, align 4
  %216 = fmul float %215, %215
  %217 = fadd float %214, %216
  store float %217, ptr %71, align 4
  %218 = load float, ptr %71, align 4
  %219 = getelementptr i8, ptr %70, i32 28
  %220 = load float, ptr %219, align 4
  %221 = fmul float %220, %220
  %222 = fadd float %218, %221
  store float %222, ptr %71, align 4
  %223 = load float, ptr %71, align 4
  %224 = load float, ptr %109, align 4
  %225 = fmul float %224, %224
  %226 = fadd float %223, %225
  store float %226, ptr %71, align 4
  %227 = load float, ptr %71, align 4
  %228 = getelementptr i8, ptr %70, i32 36
  %229 = load float, ptr %228, align 4
  %230 = fmul float %229, %229
  %231 = fadd float %227, %230
  store float %231, ptr %71, align 4
  %232 = load float, ptr %71, align 4
  %233 = load float, ptr %116, align 4
  %234 = fmul float %233, %233
  %235 = fadd float %232, %234
  store float %235, ptr %71, align 4
  %236 = load float, ptr %71, align 4
  %237 = getelementptr i8, ptr %70, i32 44
  %238 = load float, ptr %237, align 4
  %239 = fmul float %238, %238
  %240 = fadd float %236, %239
  store float %240, ptr %71, align 4
  %241 = load float, ptr %71, align 4
  %242 = load float, ptr %123, align 4
  %243 = fmul float %242, %242
  %244 = fadd float %241, %243
  store float %244, ptr %71, align 4
  %245 = load float, ptr %71, align 4
  %246 = getelementptr i8, ptr %70, i32 52
  %247 = load float, ptr %246, align 4
  %248 = fmul float %247, %247
  %249 = fadd float %245, %248
  store float %249, ptr %71, align 4
  %250 = load float, ptr %71, align 4
  %251 = load float, ptr %130, align 4
  %252 = fmul float %251, %251
  %253 = fadd float %250, %252
  store float %253, ptr %71, align 4
  %254 = load float, ptr %71, align 4
  %255 = getelementptr i8, ptr %70, i32 60
  %256 = load float, ptr %255, align 4
  %257 = fmul float %256, %256
  %258 = fadd float %254, %257
  store float %258, ptr %71, align 4
  %259 = load float, ptr %71, align 4
  %260 = load float, ptr %137, align 4
  %261 = fmul float %260, %260
  %262 = fadd float %259, %261
  store float %262, ptr %71, align 4
  %263 = load float, ptr %71, align 4
  %264 = getelementptr i8, ptr %70, i32 68
  %265 = load float, ptr %264, align 4
  %266 = fmul float %265, %265
  %267 = fadd float %263, %266
  store float %267, ptr %71, align 4
  %268 = load float, ptr %71, align 4
  %269 = load float, ptr %144, align 4
  %270 = fmul float %269, %269
  %271 = fadd float %268, %270
  store float %271, ptr %71, align 4
  %272 = load float, ptr %71, align 4
  %273 = getelementptr i8, ptr %70, i32 76
  %274 = load float, ptr %273, align 4
  %275 = fmul float %274, %274
  %276 = fadd float %272, %275
  store float %276, ptr %71, align 4
  %277 = load float, ptr %71, align 4
  %278 = load float, ptr %151, align 4
  %279 = fmul float %278, %278
  %280 = fadd float %277, %279
  store float %280, ptr %71, align 4
  %281 = load float, ptr %71, align 4
  %282 = getelementptr i8, ptr %70, i32 84
  %283 = load float, ptr %282, align 4
  %284 = fmul float %283, %283
  %285 = fadd float %281, %284
  store float %285, ptr %71, align 4
  %286 = load float, ptr %71, align 4
  %287 = load float, ptr %158, align 4
  %288 = fmul float %287, %287
  %289 = fadd float %286, %288
  store float %289, ptr %71, align 4
  %290 = load float, ptr %71, align 4
  %291 = getelementptr i8, ptr %70, i32 92
  %292 = load float, ptr %291, align 4
  %293 = fmul float %292, %292
  %294 = fadd float %290, %293
  store float %294, ptr %71, align 4
  %295 = load float, ptr %71, align 4
  %296 = load float, ptr %165, align 4
  %297 = fmul float %296, %296
  %298 = fadd float %295, %297
  store float %298, ptr %71, align 4
  %299 = load float, ptr %71, align 4
  %300 = getelementptr i8, ptr %70, i32 100
  %301 = load float, ptr %300, align 4
  %302 = fmul float %301, %301
  %303 = fadd float %299, %302
  store float %303, ptr %71, align 4
  %304 = load float, ptr %71, align 4
  %305 = load float, ptr %172, align 4
  %306 = fmul float %305, %305
  %307 = fadd float %304, %306
  store float %307, ptr %71, align 4
  %308 = load float, ptr %71, align 4
  %309 = getelementptr i8, ptr %70, i32 108
  %310 = load float, ptr %309, align 4
  %311 = fmul float %310, %310
  %312 = fadd float %308, %311
  store float %312, ptr %71, align 4
  %313 = load float, ptr %71, align 4
  %314 = load float, ptr %179, align 4
  %315 = fmul float %314, %314
  %316 = fadd float %313, %315
  store float %316, ptr %71, align 4
  %317 = load float, ptr %71, align 4
  %318 = getelementptr i8, ptr %70, i32 116
  %319 = load float, ptr %318, align 4
  %320 = fmul float %319, %319
  %321 = fadd float %317, %320
  store float %321, ptr %71, align 4
  %322 = load float, ptr %71, align 4
  %323 = load float, ptr %186, align 4
  %324 = fmul float %323, %323
  %325 = fadd float %322, %324
  store float %325, ptr %71, align 4
  %326 = load float, ptr %71, align 4
  %327 = getelementptr i8, ptr %70, i32 124
  %328 = load float, ptr %327, align 4
  %329 = fmul float %328, %328
  %330 = fadd float %326, %329
  store float %330, ptr %71, align 4
  %331 = load float, ptr %71, align 4
  %332 = getelementptr float, ptr addrspace(6) %1, i64 20480
  %333 = sdiv i32 %72, 32
  %334 = mul i32 %333, 32
  %335 = icmp ne i32 %72, %334
  %336 = icmp slt i32 %72, 0
  %337 = icmp ne i1 %336, false
  %338 = and i1 %335, %337
  %339 = add i32 %333, -1
  %340 = select i1 %338, i32 %339, i32 %333
  %341 = call i32 @llvm.hivm.get.laneID()
  %342 = call float @llvm.hivm.redux.add.f32(float %331)
  %343 = icmp slt i32 %341, 1
  br i1 %343, label %344, label %348

344:                                              ; preds = %4
  %345 = add i32 %340, %341
  %346 = sext i32 %345 to i64
  %347 = getelementptr float, ptr addrspace(6) %332, i64 %346
  store float %342, ptr addrspace(6) %347, align 4
  br label %348

348:                                              ; preds = %344, %4
  call void @llvm.hivm.sync.workitems()
  %349 = icmp slt i32 %72, 32
  br i1 %349, label %350, label %357

350:                                              ; preds = %348
  %351 = icmp slt i32 %341, 4
  %352 = sext i32 %341 to i64
  %353 = getelementptr float, ptr addrspace(6) %332, i64 %352
  %354 = load float, ptr addrspace(6) %353, align 4
  %355 = select i1 %351, float %354, float 0.000000e+00
  %356 = call float @llvm.hivm.redux.add.f32(float %355)
  br label %358

357:                                              ; preds = %348
  br label %358

358:                                              ; preds = %350, %357
  %359 = phi float [ 0.000000e+00, %357 ], [ %356, %350 ]
  br label %360

360:                                              ; preds = %358
  %361 = icmp slt i32 %72, 1
  br i1 %361, label %362, label %365

362:                                              ; preds = %360
  %363 = sext i32 %72 to i64
  %364 = getelementptr float, ptr addrspace(6) %332, i64 %363
  store float %359, ptr addrspace(6) %364, align 4
  br label %365

365:                                              ; preds = %362, %360
  call void @llvm.hivm.sync.workitems()
  %366 = getelementptr float, ptr addrspace(6) %332, i64 0
  %367 = load float, ptr addrspace(6) %366, align 4
  call void @llvm.hivm.sync.workitems()
  store float %367, ptr %71, align 4
  %368 = load float, ptr %71, align 4
  %369 = fdiv float %368, 4.096000e+03
  %370 = fadd float %369, %2
  %371 = call float @llvm.sqrt.f32(float %370)
  %372 = fdiv float 1.000000e+00, %371
  %373 = mul i64 %3, 8
  %374 = getelementptr float, ptr addrspace(6) %1, i64 %373
  store float %372, ptr addrspace(6) %374, align 4
  %375 = load <2 x float>, ptr %70, align 8
  %376 = insertelement <2 x float> undef, float %372, i32 0
  %377 = insertelement <2 x float> %376, float %372, i32 1
  %378 = fmul <2 x float> %375, %377
  %379 = insertelement <2 x float> poison, float %7, i32 0
  %380 = insertelement <2 x float> %379, float %9, i32 1
  %381 = fmul <2 x float> %378, %380
  %382 = add i64 %75, 12288
  %383 = mul i64 %382, 4
  %384 = getelementptr i8, ptr addrspace(6) %79, i64 %383
  store <2 x float> %381, ptr addrspace(6) %384, align 8
  %385 = load <2 x float>, ptr %88, align 8
  %386 = fmul <2 x float> %385, %377
  %387 = insertelement <2 x float> poison, float %11, i32 0
  %388 = insertelement <2 x float> %387, float %13, i32 1
  %389 = fmul <2 x float> %386, %388
  %390 = add i64 %83, 12288
  %391 = mul i64 %390, 4
  %392 = getelementptr i8, ptr addrspace(6) %79, i64 %391
  store <2 x float> %389, ptr addrspace(6) %392, align 8
  %393 = load <2 x float>, ptr %95, align 8
  %394 = fmul <2 x float> %393, %377
  %395 = insertelement <2 x float> poison, float %15, i32 0
  %396 = insertelement <2 x float> %395, float %17, i32 1
  %397 = fmul <2 x float> %394, %396
  %398 = add i64 %90, 12288
  %399 = mul i64 %398, 4
  %400 = getelementptr i8, ptr addrspace(6) %79, i64 %399
  store <2 x float> %397, ptr addrspace(6) %400, align 8
  %401 = load <2 x float>, ptr %102, align 8
  %402 = fmul <2 x float> %401, %377
  %403 = insertelement <2 x float> poison, float %19, i32 0
  %404 = insertelement <2 x float> %403, float %21, i32 1
  %405 = fmul <2 x float> %402, %404
  %406 = add i64 %97, 12288
  %407 = mul i64 %406, 4
  %408 = getelementptr i8, ptr addrspace(6) %79, i64 %407
  store <2 x float> %405, ptr addrspace(6) %408, align 8
  %409 = load <2 x float>, ptr %109, align 8
  %410 = fmul <2 x float> %409, %377
  %411 = insertelement <2 x float> poison, float %23, i32 0
  %412 = insertelement <2 x float> %411, float %25, i32 1
  %413 = fmul <2 x float> %410, %412
  %414 = add i64 %104, 12288
  %415 = mul i64 %414, 4
  %416 = getelementptr i8, ptr addrspace(6) %79, i64 %415
  store <2 x float> %413, ptr addrspace(6) %416, align 8
  %417 = load <2 x float>, ptr %116, align 8
  %418 = fmul <2 x float> %417, %377
  %419 = insertelement <2 x float> poison, float %27, i32 0
  %420 = insertelement <2 x float> %419, float %29, i32 1
  %421 = fmul <2 x float> %418, %420
  %422 = add i64 %111, 12288
  %423 = mul i64 %422, 4
  %424 = getelementptr i8, ptr addrspace(6) %79, i64 %423
  store <2 x float> %421, ptr addrspace(6) %424, align 8
  %425 = load <2 x float>, ptr %123, align 8
  %426 = fmul <2 x float> %425, %377
  %427 = insertelement <2 x float> poison, float %31, i32 0
  %428 = insertelement <2 x float> %427, float %33, i32 1
  %429 = fmul <2 x float> %426, %428
  %430 = add i64 %118, 12288
  %431 = mul i64 %430, 4
  %432 = getelementptr i8, ptr addrspace(6) %79, i64 %431
  store <2 x float> %429, ptr addrspace(6) %432, align 8
  %433 = load <2 x float>, ptr %130, align 8
  %434 = fmul <2 x float> %433, %377
  %435 = insertelement <2 x float> poison, float %35, i32 0
  %436 = insertelement <2 x float> %435, float %37, i32 1
  %437 = fmul <2 x float> %434, %436
  %438 = add i64 %125, 12288
  %439 = mul i64 %438, 4
  %440 = getelementptr i8, ptr addrspace(6) %79, i64 %439
  store <2 x float> %437, ptr addrspace(6) %440, align 8
  %441 = load <2 x float>, ptr %137, align 8
  %442 = fmul <2 x float> %441, %377
  %443 = insertelement <2 x float> poison, float %39, i32 0
  %444 = insertelement <2 x float> %443, float %41, i32 1
  %445 = fmul <2 x float> %442, %444
  %446 = add i64 %132, 12288
  %447 = mul i64 %446, 4
  %448 = getelementptr i8, ptr addrspace(6) %79, i64 %447
  store <2 x float> %445, ptr addrspace(6) %448, align 8
  %449 = load <2 x float>, ptr %144, align 8
  %450 = fmul <2 x float> %449, %377
  %451 = insertelement <2 x float> poison, float %43, i32 0
  %452 = insertelement <2 x float> %451, float %45, i32 1
  %453 = fmul <2 x float> %450, %452
  %454 = add i64 %139, 12288
  %455 = mul i64 %454, 4
  %456 = getelementptr i8, ptr addrspace(6) %79, i64 %455
  store <2 x float> %453, ptr addrspace(6) %456, align 8
  %457 = load <2 x float>, ptr %151, align 8
  %458 = fmul <2 x float> %457, %377
  %459 = insertelement <2 x float> poison, float %47, i32 0
  %460 = insertelement <2 x float> %459, float %49, i32 1
  %461 = fmul <2 x float> %458, %460
  %462 = add i64 %146, 12288
  %463 = mul i64 %462, 4
  %464 = getelementptr i8, ptr addrspace(6) %79, i64 %463
  store <2 x float> %461, ptr addrspace(6) %464, align 8
  %465 = load <2 x float>, ptr %158, align 8
  %466 = fmul <2 x float> %465, %377
  %467 = insertelement <2 x float> poison, float %51, i32 0
  %468 = insertelement <2 x float> %467, float %53, i32 1
  %469 = fmul <2 x float> %466, %468
  %470 = add i64 %153, 12288
  %471 = mul i64 %470, 4
  %472 = getelementptr i8, ptr addrspace(6) %79, i64 %471
  store <2 x float> %469, ptr addrspace(6) %472, align 8
  %473 = load <2 x float>, ptr %165, align 8
  %474 = fmul <2 x float> %473, %377
  %475 = insertelement <2 x float> poison, float %55, i32 0
  %476 = insertelement <2 x float> %475, float %57, i32 1
  %477 = fmul <2 x float> %474, %476
  %478 = add i64 %160, 12288
  %479 = mul i64 %478, 4
  %480 = getelementptr i8, ptr addrspace(6) %79, i64 %479
  store <2 x float> %477, ptr addrspace(6) %480, align 8
  %481 = load <2 x float>, ptr %172, align 8
  %482 = fmul <2 x float> %481, %377
  %483 = insertelement <2 x float> poison, float %59, i32 0
  %484 = insertelement <2 x float> %483, float %61, i32 1
  %485 = fmul <2 x float> %482, %484
  %486 = add i64 %167, 12288
  %487 = mul i64 %486, 4
  %488 = getelementptr i8, ptr addrspace(6) %79, i64 %487
  store <2 x float> %485, ptr addrspace(6) %488, align 8
  %489 = load <2 x float>, ptr %179, align 8
  %490 = fmul <2 x float> %489, %377
  %491 = insertelement <2 x float> poison, float %63, i32 0
  %492 = insertelement <2 x float> %491, float %65, i32 1
  %493 = fmul <2 x float> %490, %492
  %494 = add i64 %174, 12288
  %495 = mul i64 %494, 4
  %496 = getelementptr i8, ptr addrspace(6) %79, i64 %495
  store <2 x float> %493, ptr addrspace(6) %496, align 8
  %497 = load <2 x float>, ptr %186, align 8
  %498 = fmul <2 x float> %497, %377
  %499 = insertelement <2 x float> poison, float %67, i32 0
  %500 = insertelement <2 x float> %499, float %69, i32 1
  %501 = fmul <2 x float> %498, %500
  %502 = add i64 %181, 12288
  %503 = mul i64 %502, 4
  %504 = getelementptr i8, ptr addrspace(6) %79, i64 %503
  store <2 x float> %501, ptr addrspace(6) %504, align 8
  %505 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31},={TPER32},={TPER33},={TPER34},={TPER35},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"(i32 %6, i32 %8, i32 %10, i32 %12, i32 %14, i32 %16, i32 %18, i32 %20, i32 %22, i32 %24, i32 %26, i32 %28, i32 %30, i32 %32, i32 %34, i32 %36, i32 %38, i32 %40, i32 %42, i32 %44, i32 %46, i32 %48, i32 %50, i32 %52, i32 %54, i32 %56, i32 %58, i32 %60, i32 %62, i32 %64, i32 %66, i32 %68)
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #2 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #3 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @rmsnorm_d4096_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @rmsnorm_d4096_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = !{ptr @rmsnorm_d4096_kernel_mix_aiv, !"kernel_with_simt", i32 1}
!4 = distinct !{null, !"simt-max-threads", i32 128}
!5 = distinct !{null, !"simt-max-registers", i32 128}
!6 = distinct !{null, !"simt-max-threads", i32 128}
!7 = distinct !{null, !"simt-max-registers", i32 128}
!8 = !{!"simt-max-threads", i32 128}
!9 = !{!"simt-max-registers", i32 128}
