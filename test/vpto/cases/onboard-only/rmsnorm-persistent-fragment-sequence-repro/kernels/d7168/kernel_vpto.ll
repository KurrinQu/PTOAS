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

define void @rmsnorm_d7168_kernel_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, float %4) #2 {
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) null, ptr addrspace(1) %1, i64 962072674320, i64 31525197391622144)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.store.vfsimt.info(i64 4295033088)
  call simt_entry void @rmsnorm_d7168_kernel_simt_0(ptr addrspace(6) null)
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %27, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 64
  br i1 %8, label %9, label %28

9:                                                ; preds = %6
  %10 = and i64 %7, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %10)
  %11 = mul i64 %7, 458752
  %12 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %13 = mul i64 %12, 7168
  %14 = add i64 %11, %13
  %15 = getelementptr float, ptr addrspace(1) %2, i64 %14
  %16 = mul i64 %10, 8192
  %17 = add i64 %16, 7168
  %18 = getelementptr float, ptr addrspace(6) null, i64 %17
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) %18, ptr addrspace(1) %15, i64 962072674320, i64 31525197391622144)
  call void @llvm.hivm.SET.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.store.vfsimt.info(i64 4295033088)
  call simt_entry void @rmsnorm_d7168_kernel_simt_1(i64 %16, ptr addrspace(6) null, i64 %17, float %4, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 5, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 5, i64 %10)
  %19 = add i64 %16, 23552
  %20 = getelementptr float, ptr addrspace(6) null, i64 %19
  %21 = getelementptr float, ptr addrspace(1) %3, i64 %14
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %21, ptr addrspace(6) %20, i64 962072674320, i64 31525197391622144)
  %22 = mul i64 %10, 8
  %23 = getelementptr float, ptr addrspace(6) null, i64 %22
  %24 = mul i64 %7, 64
  %25 = add i64 %24, %12
  %26 = getelementptr float, ptr addrspace(1) %0, i64 %25
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %26, ptr addrspace(6) %23, i64 134217744, i64 4398046511108)
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
define linkonce_odr simt_entry void @rmsnorm_d7168_kernel_simt_0(ptr addrspace(6) %0) #3 !annotation !8 !annotation !9 {
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
  %12 = add i32 %3, 512
  %13 = sext i32 %12 to i64
  %14 = mul i64 %13, 4
  %15 = getelementptr i8, ptr addrspace(6) %7, i64 %14
  %16 = load <2 x float>, ptr addrspace(6) %15, align 8
  %17 = extractelement <2 x float> %16, i32 0
  %18 = extractelement <2 x float> %16, i32 1
  %19 = add i32 %3, 1024
  %20 = sext i32 %19 to i64
  %21 = mul i64 %20, 4
  %22 = getelementptr i8, ptr addrspace(6) %7, i64 %21
  %23 = load <2 x float>, ptr addrspace(6) %22, align 8
  %24 = extractelement <2 x float> %23, i32 0
  %25 = extractelement <2 x float> %23, i32 1
  %26 = add i32 %3, 1536
  %27 = sext i32 %26 to i64
  %28 = mul i64 %27, 4
  %29 = getelementptr i8, ptr addrspace(6) %7, i64 %28
  %30 = load <2 x float>, ptr addrspace(6) %29, align 8
  %31 = extractelement <2 x float> %30, i32 0
  %32 = extractelement <2 x float> %30, i32 1
  %33 = add i32 %3, 2048
  %34 = sext i32 %33 to i64
  %35 = mul i64 %34, 4
  %36 = getelementptr i8, ptr addrspace(6) %7, i64 %35
  %37 = load <2 x float>, ptr addrspace(6) %36, align 8
  %38 = extractelement <2 x float> %37, i32 0
  %39 = extractelement <2 x float> %37, i32 1
  %40 = add i32 %3, 2560
  %41 = sext i32 %40 to i64
  %42 = mul i64 %41, 4
  %43 = getelementptr i8, ptr addrspace(6) %7, i64 %42
  %44 = load <2 x float>, ptr addrspace(6) %43, align 8
  %45 = extractelement <2 x float> %44, i32 0
  %46 = extractelement <2 x float> %44, i32 1
  %47 = add i32 %3, 3072
  %48 = sext i32 %47 to i64
  %49 = mul i64 %48, 4
  %50 = getelementptr i8, ptr addrspace(6) %7, i64 %49
  %51 = load <2 x float>, ptr addrspace(6) %50, align 8
  %52 = extractelement <2 x float> %51, i32 0
  %53 = extractelement <2 x float> %51, i32 1
  %54 = add i32 %3, 3584
  %55 = sext i32 %54 to i64
  %56 = mul i64 %55, 4
  %57 = getelementptr i8, ptr addrspace(6) %7, i64 %56
  %58 = load <2 x float>, ptr addrspace(6) %57, align 8
  %59 = extractelement <2 x float> %58, i32 0
  %60 = extractelement <2 x float> %58, i32 1
  %61 = add i32 %3, 4096
  %62 = sext i32 %61 to i64
  %63 = mul i64 %62, 4
  %64 = getelementptr i8, ptr addrspace(6) %7, i64 %63
  %65 = load <2 x float>, ptr addrspace(6) %64, align 8
  %66 = extractelement <2 x float> %65, i32 0
  %67 = extractelement <2 x float> %65, i32 1
  %68 = add i32 %3, 4608
  %69 = sext i32 %68 to i64
  %70 = mul i64 %69, 4
  %71 = getelementptr i8, ptr addrspace(6) %7, i64 %70
  %72 = load <2 x float>, ptr addrspace(6) %71, align 8
  %73 = extractelement <2 x float> %72, i32 0
  %74 = extractelement <2 x float> %72, i32 1
  %75 = add i32 %3, 5120
  %76 = sext i32 %75 to i64
  %77 = mul i64 %76, 4
  %78 = getelementptr i8, ptr addrspace(6) %7, i64 %77
  %79 = load <2 x float>, ptr addrspace(6) %78, align 8
  %80 = extractelement <2 x float> %79, i32 0
  %81 = extractelement <2 x float> %79, i32 1
  %82 = add i32 %3, 5632
  %83 = sext i32 %82 to i64
  %84 = mul i64 %83, 4
  %85 = getelementptr i8, ptr addrspace(6) %7, i64 %84
  %86 = load <2 x float>, ptr addrspace(6) %85, align 8
  %87 = extractelement <2 x float> %86, i32 0
  %88 = extractelement <2 x float> %86, i32 1
  %89 = add i32 %3, 6144
  %90 = sext i32 %89 to i64
  %91 = mul i64 %90, 4
  %92 = getelementptr i8, ptr addrspace(6) %7, i64 %91
  %93 = load <2 x float>, ptr addrspace(6) %92, align 8
  %94 = extractelement <2 x float> %93, i32 0
  %95 = extractelement <2 x float> %93, i32 1
  %96 = add i32 %3, 6656
  %97 = sext i32 %96 to i64
  %98 = mul i64 %97, 4
  %99 = getelementptr i8, ptr addrspace(6) %7, i64 %98
  %100 = load <2 x float>, ptr addrspace(6) %99, align 8
  %101 = extractelement <2 x float> %100, i32 0
  %102 = extractelement <2 x float> %100, i32 1
  %103 = bitcast float %10 to i32
  %104 = bitcast float %11 to i32
  %105 = bitcast float %17 to i32
  %106 = bitcast float %18 to i32
  %107 = bitcast float %24 to i32
  %108 = bitcast float %25 to i32
  %109 = bitcast float %31 to i32
  %110 = bitcast float %32 to i32
  %111 = bitcast float %38 to i32
  %112 = bitcast float %39 to i32
  %113 = bitcast float %45 to i32
  %114 = bitcast float %46 to i32
  %115 = bitcast float %52 to i32
  %116 = bitcast float %53 to i32
  %117 = bitcast float %59 to i32
  %118 = bitcast float %60 to i32
  %119 = bitcast float %66 to i32
  %120 = bitcast float %67 to i32
  %121 = bitcast float %73 to i32
  %122 = bitcast float %74 to i32
  %123 = bitcast float %80 to i32
  %124 = bitcast float %81 to i32
  %125 = bitcast float %87 to i32
  %126 = bitcast float %88 to i32
  %127 = bitcast float %94 to i32
  %128 = bitcast float %95 to i32
  %129 = bitcast float %101 to i32
  %130 = bitcast float %102 to i32
  %131 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"(i32 %103, i32 %104, i32 %105, i32 %106, i32 %107, i32 %108, i32 %109, i32 %110, i32 %111, i32 %112, i32 %113, i32 %114, i32 %115, i32 %116, i32 %117, i32 %118, i32 %119, i32 %120, i32 %121, i32 %122, i32 %123, i32 %124, i32 %125, i32 %126, i32 %127, i32 %128, i32 %129, i32 %130)
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @rmsnorm_d7168_kernel_simt_1(i64 %0, ptr addrspace(6) %1, i64 %2, float %3, i64 %4) #3 !annotation !8 !annotation !9 {
  %6 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31}"()
  %7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 0
  %8 = bitcast i32 %7 to float
  %9 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 1
  %10 = bitcast i32 %9 to float
  %11 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 2
  %12 = bitcast i32 %11 to float
  %13 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 3
  %14 = bitcast i32 %13 to float
  %15 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 4
  %16 = bitcast i32 %15 to float
  %17 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 5
  %18 = bitcast i32 %17 to float
  %19 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 6
  %20 = bitcast i32 %19 to float
  %21 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 7
  %22 = bitcast i32 %21 to float
  %23 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 8
  %24 = bitcast i32 %23 to float
  %25 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 9
  %26 = bitcast i32 %25 to float
  %27 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 10
  %28 = bitcast i32 %27 to float
  %29 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 11
  %30 = bitcast i32 %29 to float
  %31 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 12
  %32 = bitcast i32 %31 to float
  %33 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 13
  %34 = bitcast i32 %33 to float
  %35 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 14
  %36 = bitcast i32 %35 to float
  %37 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 15
  %38 = bitcast i32 %37 to float
  %39 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 16
  %40 = bitcast i32 %39 to float
  %41 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 17
  %42 = bitcast i32 %41 to float
  %43 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 18
  %44 = bitcast i32 %43 to float
  %45 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 19
  %46 = bitcast i32 %45 to float
  %47 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 20
  %48 = bitcast i32 %47 to float
  %49 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 21
  %50 = bitcast i32 %49 to float
  %51 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 22
  %52 = bitcast i32 %51 to float
  %53 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 23
  %54 = bitcast i32 %53 to float
  %55 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 24
  %56 = bitcast i32 %55 to float
  %57 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 25
  %58 = bitcast i32 %57 to float
  %59 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 26
  %60 = bitcast i32 %59 to float
  %61 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 27
  %62 = bitcast i32 %61 to float
  %63 = alloca float, i32 32, align 4
  %64 = alloca float, align 4
  %65 = call i32 @llvm.hivm.get.TID.X()
  %66 = mul i32 %65, 2
  %67 = sext i32 %66 to i64
  %68 = add i64 %0, %67
  %69 = add i64 %68, 7168
  %70 = mul i64 %69, 4
  %71 = ptrtoint ptr addrspace(6) %1 to i64
  %72 = inttoptr i64 %71 to ptr addrspace(6)
  %73 = getelementptr i8, ptr addrspace(6) %72, i64 %70
  %74 = load <2 x float>, ptr addrspace(6) %73, align 8
  store <2 x float> %74, ptr %63, align 8
  %75 = add i64 %0, 512
  %76 = add i64 %75, %67
  %77 = add i64 %76, 7168
  %78 = mul i64 %77, 4
  %79 = getelementptr i8, ptr addrspace(6) %72, i64 %78
  %80 = load <2 x float>, ptr addrspace(6) %79, align 8
  %81 = getelementptr i8, ptr %63, i32 8
  store <2 x float> %80, ptr %81, align 8
  %82 = add i64 %0, 1024
  %83 = add i64 %82, %67
  %84 = add i64 %83, 7168
  %85 = mul i64 %84, 4
  %86 = getelementptr i8, ptr addrspace(6) %72, i64 %85
  %87 = load <2 x float>, ptr addrspace(6) %86, align 8
  %88 = getelementptr i8, ptr %63, i32 16
  store <2 x float> %87, ptr %88, align 8
  %89 = add i64 %0, 1536
  %90 = add i64 %89, %67
  %91 = add i64 %90, 7168
  %92 = mul i64 %91, 4
  %93 = getelementptr i8, ptr addrspace(6) %72, i64 %92
  %94 = load <2 x float>, ptr addrspace(6) %93, align 8
  %95 = getelementptr i8, ptr %63, i32 24
  store <2 x float> %94, ptr %95, align 8
  %96 = add i64 %0, 2048
  %97 = add i64 %96, %67
  %98 = add i64 %97, 7168
  %99 = mul i64 %98, 4
  %100 = getelementptr i8, ptr addrspace(6) %72, i64 %99
  %101 = load <2 x float>, ptr addrspace(6) %100, align 8
  %102 = getelementptr i8, ptr %63, i32 32
  store <2 x float> %101, ptr %102, align 8
  %103 = add i64 %0, 2560
  %104 = add i64 %103, %67
  %105 = add i64 %104, 7168
  %106 = mul i64 %105, 4
  %107 = getelementptr i8, ptr addrspace(6) %72, i64 %106
  %108 = load <2 x float>, ptr addrspace(6) %107, align 8
  %109 = getelementptr i8, ptr %63, i32 40
  store <2 x float> %108, ptr %109, align 8
  %110 = add i64 %0, 3072
  %111 = add i64 %110, %67
  %112 = add i64 %111, 7168
  %113 = mul i64 %112, 4
  %114 = getelementptr i8, ptr addrspace(6) %72, i64 %113
  %115 = load <2 x float>, ptr addrspace(6) %114, align 8
  %116 = getelementptr i8, ptr %63, i32 48
  store <2 x float> %115, ptr %116, align 8
  %117 = add i64 %0, 3584
  %118 = add i64 %117, %67
  %119 = add i64 %118, 7168
  %120 = mul i64 %119, 4
  %121 = getelementptr i8, ptr addrspace(6) %72, i64 %120
  %122 = load <2 x float>, ptr addrspace(6) %121, align 8
  %123 = getelementptr i8, ptr %63, i32 56
  store <2 x float> %122, ptr %123, align 8
  %124 = add i64 %0, 4096
  %125 = add i64 %124, %67
  %126 = add i64 %125, 7168
  %127 = mul i64 %126, 4
  %128 = getelementptr i8, ptr addrspace(6) %72, i64 %127
  %129 = load <2 x float>, ptr addrspace(6) %128, align 8
  %130 = getelementptr i8, ptr %63, i32 64
  store <2 x float> %129, ptr %130, align 8
  %131 = add i64 %0, 4608
  %132 = add i64 %131, %67
  %133 = add i64 %132, 7168
  %134 = mul i64 %133, 4
  %135 = getelementptr i8, ptr addrspace(6) %72, i64 %134
  %136 = load <2 x float>, ptr addrspace(6) %135, align 8
  %137 = getelementptr i8, ptr %63, i32 72
  store <2 x float> %136, ptr %137, align 8
  %138 = add i64 %0, 5120
  %139 = add i64 %138, %67
  %140 = add i64 %139, 7168
  %141 = mul i64 %140, 4
  %142 = getelementptr i8, ptr addrspace(6) %72, i64 %141
  %143 = load <2 x float>, ptr addrspace(6) %142, align 8
  %144 = getelementptr i8, ptr %63, i32 80
  store <2 x float> %143, ptr %144, align 8
  %145 = add i64 %0, 5632
  %146 = add i64 %145, %67
  %147 = add i64 %146, 7168
  %148 = mul i64 %147, 4
  %149 = getelementptr i8, ptr addrspace(6) %72, i64 %148
  %150 = load <2 x float>, ptr addrspace(6) %149, align 8
  %151 = getelementptr i8, ptr %63, i32 88
  store <2 x float> %150, ptr %151, align 8
  %152 = add i64 %0, 6144
  %153 = add i64 %152, %67
  %154 = add i64 %153, 7168
  %155 = mul i64 %154, 4
  %156 = getelementptr i8, ptr addrspace(6) %72, i64 %155
  %157 = load <2 x float>, ptr addrspace(6) %156, align 8
  %158 = getelementptr i8, ptr %63, i32 96
  store <2 x float> %157, ptr %158, align 8
  %159 = add i64 %0, 6656
  %160 = add i64 %159, %67
  %161 = add i64 %160, 7168
  %162 = mul i64 %161, 4
  %163 = getelementptr i8, ptr addrspace(6) %72, i64 %162
  %164 = load <2 x float>, ptr addrspace(6) %163, align 8
  %165 = getelementptr i8, ptr %63, i32 104
  store <2 x float> %164, ptr %165, align 8
  %166 = add i64 %2, %67
  %167 = add i64 %166, 7168
  %168 = mul i64 %167, 4
  %169 = getelementptr i8, ptr addrspace(6) %72, i64 %168
  %170 = load <2 x float>, ptr addrspace(6) %169, align 8
  %171 = getelementptr i8, ptr %63, i32 112
  store <2 x float> %170, ptr %171, align 8
  %172 = add i64 %0, 7680
  %173 = add i64 %172, %67
  %174 = add i64 %173, 7168
  %175 = mul i64 %174, 4
  %176 = getelementptr i8, ptr addrspace(6) %72, i64 %175
  %177 = load <2 x float>, ptr addrspace(6) %176, align 8
  %178 = getelementptr i8, ptr %63, i32 120
  store <2 x float> %177, ptr %178, align 8
  store float 0.000000e+00, ptr %64, align 4
  %179 = load float, ptr %64, align 4
  %180 = load float, ptr %63, align 4
  %181 = fmul float %180, %180
  %182 = fadd float %179, %181
  store float %182, ptr %64, align 4
  %183 = load float, ptr %64, align 4
  %184 = getelementptr i8, ptr %63, i32 4
  %185 = load float, ptr %184, align 4
  %186 = fmul float %185, %185
  %187 = fadd float %183, %186
  store float %187, ptr %64, align 4
  %188 = load float, ptr %64, align 4
  %189 = load float, ptr %81, align 4
  %190 = fmul float %189, %189
  %191 = fadd float %188, %190
  store float %191, ptr %64, align 4
  %192 = load float, ptr %64, align 4
  %193 = getelementptr i8, ptr %63, i32 12
  %194 = load float, ptr %193, align 4
  %195 = fmul float %194, %194
  %196 = fadd float %192, %195
  store float %196, ptr %64, align 4
  %197 = load float, ptr %64, align 4
  %198 = load float, ptr %88, align 4
  %199 = fmul float %198, %198
  %200 = fadd float %197, %199
  store float %200, ptr %64, align 4
  %201 = load float, ptr %64, align 4
  %202 = getelementptr i8, ptr %63, i32 20
  %203 = load float, ptr %202, align 4
  %204 = fmul float %203, %203
  %205 = fadd float %201, %204
  store float %205, ptr %64, align 4
  %206 = load float, ptr %64, align 4
  %207 = load float, ptr %95, align 4
  %208 = fmul float %207, %207
  %209 = fadd float %206, %208
  store float %209, ptr %64, align 4
  %210 = load float, ptr %64, align 4
  %211 = getelementptr i8, ptr %63, i32 28
  %212 = load float, ptr %211, align 4
  %213 = fmul float %212, %212
  %214 = fadd float %210, %213
  store float %214, ptr %64, align 4
  %215 = load float, ptr %64, align 4
  %216 = load float, ptr %102, align 4
  %217 = fmul float %216, %216
  %218 = fadd float %215, %217
  store float %218, ptr %64, align 4
  %219 = load float, ptr %64, align 4
  %220 = getelementptr i8, ptr %63, i32 36
  %221 = load float, ptr %220, align 4
  %222 = fmul float %221, %221
  %223 = fadd float %219, %222
  store float %223, ptr %64, align 4
  %224 = load float, ptr %64, align 4
  %225 = load float, ptr %109, align 4
  %226 = fmul float %225, %225
  %227 = fadd float %224, %226
  store float %227, ptr %64, align 4
  %228 = load float, ptr %64, align 4
  %229 = getelementptr i8, ptr %63, i32 44
  %230 = load float, ptr %229, align 4
  %231 = fmul float %230, %230
  %232 = fadd float %228, %231
  store float %232, ptr %64, align 4
  %233 = load float, ptr %64, align 4
  %234 = load float, ptr %116, align 4
  %235 = fmul float %234, %234
  %236 = fadd float %233, %235
  store float %236, ptr %64, align 4
  %237 = load float, ptr %64, align 4
  %238 = getelementptr i8, ptr %63, i32 52
  %239 = load float, ptr %238, align 4
  %240 = fmul float %239, %239
  %241 = fadd float %237, %240
  store float %241, ptr %64, align 4
  %242 = load float, ptr %64, align 4
  %243 = load float, ptr %123, align 4
  %244 = fmul float %243, %243
  %245 = fadd float %242, %244
  store float %245, ptr %64, align 4
  %246 = load float, ptr %64, align 4
  %247 = getelementptr i8, ptr %63, i32 60
  %248 = load float, ptr %247, align 4
  %249 = fmul float %248, %248
  %250 = fadd float %246, %249
  store float %250, ptr %64, align 4
  %251 = load float, ptr %64, align 4
  %252 = load float, ptr %130, align 4
  %253 = fmul float %252, %252
  %254 = fadd float %251, %253
  store float %254, ptr %64, align 4
  %255 = load float, ptr %64, align 4
  %256 = getelementptr i8, ptr %63, i32 68
  %257 = load float, ptr %256, align 4
  %258 = fmul float %257, %257
  %259 = fadd float %255, %258
  store float %259, ptr %64, align 4
  %260 = load float, ptr %64, align 4
  %261 = load float, ptr %137, align 4
  %262 = fmul float %261, %261
  %263 = fadd float %260, %262
  store float %263, ptr %64, align 4
  %264 = load float, ptr %64, align 4
  %265 = getelementptr i8, ptr %63, i32 76
  %266 = load float, ptr %265, align 4
  %267 = fmul float %266, %266
  %268 = fadd float %264, %267
  store float %268, ptr %64, align 4
  %269 = load float, ptr %64, align 4
  %270 = load float, ptr %144, align 4
  %271 = fmul float %270, %270
  %272 = fadd float %269, %271
  store float %272, ptr %64, align 4
  %273 = load float, ptr %64, align 4
  %274 = getelementptr i8, ptr %63, i32 84
  %275 = load float, ptr %274, align 4
  %276 = fmul float %275, %275
  %277 = fadd float %273, %276
  store float %277, ptr %64, align 4
  %278 = load float, ptr %64, align 4
  %279 = load float, ptr %151, align 4
  %280 = fmul float %279, %279
  %281 = fadd float %278, %280
  store float %281, ptr %64, align 4
  %282 = load float, ptr %64, align 4
  %283 = getelementptr i8, ptr %63, i32 92
  %284 = load float, ptr %283, align 4
  %285 = fmul float %284, %284
  %286 = fadd float %282, %285
  store float %286, ptr %64, align 4
  %287 = load float, ptr %64, align 4
  %288 = load float, ptr %158, align 4
  %289 = fmul float %288, %288
  %290 = fadd float %287, %289
  store float %290, ptr %64, align 4
  %291 = load float, ptr %64, align 4
  %292 = getelementptr i8, ptr %63, i32 100
  %293 = load float, ptr %292, align 4
  %294 = fmul float %293, %293
  %295 = fadd float %291, %294
  store float %295, ptr %64, align 4
  %296 = load float, ptr %64, align 4
  %297 = load float, ptr %165, align 4
  %298 = fmul float %297, %297
  %299 = fadd float %296, %298
  store float %299, ptr %64, align 4
  %300 = load float, ptr %64, align 4
  %301 = getelementptr i8, ptr %63, i32 108
  %302 = load float, ptr %301, align 4
  %303 = fmul float %302, %302
  %304 = fadd float %300, %303
  store float %304, ptr %64, align 4
  %305 = load float, ptr %64, align 4
  %306 = getelementptr float, ptr addrspace(6) %1, i64 39936
  %307 = sdiv i32 %65, 32
  %308 = mul i32 %307, 32
  %309 = icmp ne i32 %65, %308
  %310 = icmp slt i32 %65, 0
  %311 = icmp ne i1 %310, false
  %312 = and i1 %309, %311
  %313 = add i32 %307, -1
  %314 = select i1 %312, i32 %313, i32 %307
  %315 = call i32 @llvm.hivm.get.laneID()
  %316 = call float @llvm.hivm.redux.add.f32(float %305)
  %317 = icmp slt i32 %315, 1
  br i1 %317, label %318, label %322

318:                                              ; preds = %5
  %319 = add i32 %314, %315
  %320 = sext i32 %319 to i64
  %321 = getelementptr float, ptr addrspace(6) %306, i64 %320
  store float %316, ptr addrspace(6) %321, align 4
  br label %322

322:                                              ; preds = %318, %5
  call void @llvm.hivm.sync.workitems()
  %323 = icmp slt i32 %65, 32
  br i1 %323, label %324, label %331

324:                                              ; preds = %322
  %325 = icmp slt i32 %315, 8
  %326 = sext i32 %315 to i64
  %327 = getelementptr float, ptr addrspace(6) %306, i64 %326
  %328 = load float, ptr addrspace(6) %327, align 4
  %329 = select i1 %325, float %328, float 0.000000e+00
  %330 = call float @llvm.hivm.redux.add.f32(float %329)
  br label %332

331:                                              ; preds = %322
  br label %332

332:                                              ; preds = %324, %331
  %333 = phi float [ 0.000000e+00, %331 ], [ %330, %324 ]
  br label %334

334:                                              ; preds = %332
  %335 = icmp slt i32 %65, 1
  br i1 %335, label %336, label %339

336:                                              ; preds = %334
  %337 = sext i32 %65 to i64
  %338 = getelementptr float, ptr addrspace(6) %306, i64 %337
  store float %333, ptr addrspace(6) %338, align 4
  br label %339

339:                                              ; preds = %336, %334
  call void @llvm.hivm.sync.workitems()
  %340 = getelementptr float, ptr addrspace(6) %306, i64 0
  %341 = load float, ptr addrspace(6) %340, align 4
  call void @llvm.hivm.sync.workitems()
  store float %341, ptr %64, align 4
  %342 = load float, ptr %64, align 4
  %343 = fdiv float %342, 7.168000e+03
  %344 = fadd float %343, %3
  %345 = call float @llvm.sqrt.f32(float %344)
  %346 = fdiv float 1.000000e+00, %345
  %347 = mul i64 %4, 8
  %348 = getelementptr float, ptr addrspace(6) %1, i64 %347
  store float %346, ptr addrspace(6) %348, align 4
  %349 = load <2 x float>, ptr %63, align 8
  %350 = insertelement <2 x float> undef, float %346, i32 0
  %351 = insertelement <2 x float> %350, float %346, i32 1
  %352 = fmul <2 x float> %349, %351
  %353 = insertelement <2 x float> poison, float %8, i32 0
  %354 = insertelement <2 x float> %353, float %10, i32 1
  %355 = fmul <2 x float> %352, %354
  %356 = add i64 %68, 23552
  %357 = mul i64 %356, 4
  %358 = getelementptr i8, ptr addrspace(6) %72, i64 %357
  store <2 x float> %355, ptr addrspace(6) %358, align 8
  %359 = load <2 x float>, ptr %81, align 8
  %360 = fmul <2 x float> %359, %351
  %361 = insertelement <2 x float> poison, float %12, i32 0
  %362 = insertelement <2 x float> %361, float %14, i32 1
  %363 = fmul <2 x float> %360, %362
  %364 = add i64 %76, 23552
  %365 = mul i64 %364, 4
  %366 = getelementptr i8, ptr addrspace(6) %72, i64 %365
  store <2 x float> %363, ptr addrspace(6) %366, align 8
  %367 = load <2 x float>, ptr %88, align 8
  %368 = fmul <2 x float> %367, %351
  %369 = insertelement <2 x float> poison, float %16, i32 0
  %370 = insertelement <2 x float> %369, float %18, i32 1
  %371 = fmul <2 x float> %368, %370
  %372 = add i64 %83, 23552
  %373 = mul i64 %372, 4
  %374 = getelementptr i8, ptr addrspace(6) %72, i64 %373
  store <2 x float> %371, ptr addrspace(6) %374, align 8
  %375 = load <2 x float>, ptr %95, align 8
  %376 = fmul <2 x float> %375, %351
  %377 = insertelement <2 x float> poison, float %20, i32 0
  %378 = insertelement <2 x float> %377, float %22, i32 1
  %379 = fmul <2 x float> %376, %378
  %380 = add i64 %90, 23552
  %381 = mul i64 %380, 4
  %382 = getelementptr i8, ptr addrspace(6) %72, i64 %381
  store <2 x float> %379, ptr addrspace(6) %382, align 8
  %383 = load <2 x float>, ptr %102, align 8
  %384 = fmul <2 x float> %383, %351
  %385 = insertelement <2 x float> poison, float %24, i32 0
  %386 = insertelement <2 x float> %385, float %26, i32 1
  %387 = fmul <2 x float> %384, %386
  %388 = add i64 %97, 23552
  %389 = mul i64 %388, 4
  %390 = getelementptr i8, ptr addrspace(6) %72, i64 %389
  store <2 x float> %387, ptr addrspace(6) %390, align 8
  %391 = load <2 x float>, ptr %109, align 8
  %392 = fmul <2 x float> %391, %351
  %393 = insertelement <2 x float> poison, float %28, i32 0
  %394 = insertelement <2 x float> %393, float %30, i32 1
  %395 = fmul <2 x float> %392, %394
  %396 = add i64 %104, 23552
  %397 = mul i64 %396, 4
  %398 = getelementptr i8, ptr addrspace(6) %72, i64 %397
  store <2 x float> %395, ptr addrspace(6) %398, align 8
  %399 = load <2 x float>, ptr %116, align 8
  %400 = fmul <2 x float> %399, %351
  %401 = insertelement <2 x float> poison, float %32, i32 0
  %402 = insertelement <2 x float> %401, float %34, i32 1
  %403 = fmul <2 x float> %400, %402
  %404 = add i64 %111, 23552
  %405 = mul i64 %404, 4
  %406 = getelementptr i8, ptr addrspace(6) %72, i64 %405
  store <2 x float> %403, ptr addrspace(6) %406, align 8
  %407 = load <2 x float>, ptr %123, align 8
  %408 = fmul <2 x float> %407, %351
  %409 = insertelement <2 x float> poison, float %36, i32 0
  %410 = insertelement <2 x float> %409, float %38, i32 1
  %411 = fmul <2 x float> %408, %410
  %412 = add i64 %118, 23552
  %413 = mul i64 %412, 4
  %414 = getelementptr i8, ptr addrspace(6) %72, i64 %413
  store <2 x float> %411, ptr addrspace(6) %414, align 8
  %415 = load <2 x float>, ptr %130, align 8
  %416 = fmul <2 x float> %415, %351
  %417 = insertelement <2 x float> poison, float %40, i32 0
  %418 = insertelement <2 x float> %417, float %42, i32 1
  %419 = fmul <2 x float> %416, %418
  %420 = add i64 %125, 23552
  %421 = mul i64 %420, 4
  %422 = getelementptr i8, ptr addrspace(6) %72, i64 %421
  store <2 x float> %419, ptr addrspace(6) %422, align 8
  %423 = load <2 x float>, ptr %137, align 8
  %424 = fmul <2 x float> %423, %351
  %425 = insertelement <2 x float> poison, float %44, i32 0
  %426 = insertelement <2 x float> %425, float %46, i32 1
  %427 = fmul <2 x float> %424, %426
  %428 = add i64 %132, 23552
  %429 = mul i64 %428, 4
  %430 = getelementptr i8, ptr addrspace(6) %72, i64 %429
  store <2 x float> %427, ptr addrspace(6) %430, align 8
  %431 = load <2 x float>, ptr %144, align 8
  %432 = fmul <2 x float> %431, %351
  %433 = insertelement <2 x float> poison, float %48, i32 0
  %434 = insertelement <2 x float> %433, float %50, i32 1
  %435 = fmul <2 x float> %432, %434
  %436 = add i64 %139, 23552
  %437 = mul i64 %436, 4
  %438 = getelementptr i8, ptr addrspace(6) %72, i64 %437
  store <2 x float> %435, ptr addrspace(6) %438, align 8
  %439 = load <2 x float>, ptr %151, align 8
  %440 = fmul <2 x float> %439, %351
  %441 = insertelement <2 x float> poison, float %52, i32 0
  %442 = insertelement <2 x float> %441, float %54, i32 1
  %443 = fmul <2 x float> %440, %442
  %444 = add i64 %146, 23552
  %445 = mul i64 %444, 4
  %446 = getelementptr i8, ptr addrspace(6) %72, i64 %445
  store <2 x float> %443, ptr addrspace(6) %446, align 8
  %447 = load <2 x float>, ptr %158, align 8
  %448 = fmul <2 x float> %447, %351
  %449 = insertelement <2 x float> poison, float %56, i32 0
  %450 = insertelement <2 x float> %449, float %58, i32 1
  %451 = fmul <2 x float> %448, %450
  %452 = add i64 %153, 23552
  %453 = mul i64 %452, 4
  %454 = getelementptr i8, ptr addrspace(6) %72, i64 %453
  store <2 x float> %451, ptr addrspace(6) %454, align 8
  %455 = load <2 x float>, ptr %165, align 8
  %456 = fmul <2 x float> %455, %351
  %457 = insertelement <2 x float> poison, float %60, i32 0
  %458 = insertelement <2 x float> %457, float %62, i32 1
  %459 = fmul <2 x float> %456, %458
  %460 = add i64 %160, 23552
  %461 = mul i64 %460, 4
  %462 = getelementptr i8, ptr addrspace(6) %72, i64 %461
  store <2 x float> %459, ptr addrspace(6) %462, align 8
  %463 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},={TPER24},={TPER25},={TPER26},={TPER27},={TPER28},={TPER29},={TPER30},={TPER31},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"(i32 %7, i32 %9, i32 %11, i32 %13, i32 %15, i32 %17, i32 %19, i32 %21, i32 %23, i32 %25, i32 %27, i32 %29, i32 %31, i32 %33, i32 %35, i32 %37, i32 %39, i32 %41, i32 %43, i32 %45, i32 %47, i32 %49, i32 %51, i32 %53, i32 %55, i32 %57, i32 %59, i32 %61)
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #2 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #3 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @rmsnorm_d7168_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @rmsnorm_d7168_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = !{ptr @rmsnorm_d7168_kernel_mix_aiv, !"kernel_with_simt", i32 1}
!4 = distinct !{null, !"simt-max-threads", i32 256}
!5 = distinct !{null, !"simt-max-registers", i32 128}
!6 = distinct !{null, !"simt-max-threads", i32 256}
!7 = distinct !{null, !"simt-max-registers", i32 128}
!8 = !{!"simt-max-threads", i32 256}
!9 = !{!"simt-max-registers", i32 128}
