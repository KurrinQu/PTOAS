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

define void @rmsnorm_d5120_kernel_mix_aiv(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3, float %4) #2 {
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 5, i64 1, i64 1)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 0)
  call void @llvm.hivm.SET.FLAG.IMM(i64 1, i64 4, i64 1)
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) null, ptr addrspace(1) %1, i64 687194767376, i64 22517998136872960)
  call void @llvm.hivm.SET.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.WAIT.FLAG.IMM(i64 4, i64 1, i64 2)
  call void @llvm.hivm.store.vfsimt.info(i64 4295033088)
  call simt_entry void @rmsnorm_d5120_kernel_simt_0(ptr addrspace(6) null)
  br label %6

6:                                                ; preds = %9, %5
  %7 = phi i64 [ %27, %9 ], [ 0, %5 ]
  %8 = icmp slt i64 %7, 64
  br i1 %8, label %9, label %28

9:                                                ; preds = %6
  %10 = and i64 %7, 1
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 4, i64 %10)
  %11 = mul i64 %7, 327680
  %12 = call i64 @llvm.hivm.GET.BLOCK.IDX()
  %13 = mul i64 %12, 5120
  %14 = add i64 %11, %13
  %15 = getelementptr float, ptr addrspace(1) %2, i64 %14
  %16 = mul i64 %10, 8192
  %17 = add i64 %16, 5120
  %18 = getelementptr float, ptr addrspace(6) null, i64 %17
  call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV(ptr addrspace(6) %18, ptr addrspace(1) %15, i64 687194767376, i64 22517998136872960)
  call void @llvm.hivm.SET.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 5, i64 1, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 4, i64 1, i64 %10)
  call void @llvm.hivm.store.vfsimt.info(i64 4295033088)
  call simt_entry void @rmsnorm_d5120_kernel_simt_1(i64 %16, ptr addrspace(6) null, i64 %17, float %4, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 5, i64 %10)
  call void @llvm.hivm.SET.FLAG.REG(i64 1, i64 4, i64 %10)
  call void @llvm.hivm.WAIT.FLAG.REG(i64 1, i64 5, i64 %10)
  %19 = mul i64 %10, 8
  %20 = getelementptr float, ptr addrspace(6) null, i64 %19
  %21 = mul i64 %7, 64
  %22 = add i64 %21, %12
  %23 = getelementptr float, ptr addrspace(1) %0, i64 %22
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %23, ptr addrspace(6) %20, i64 134217744, i64 4398046511108)
  %24 = add i64 %16, 21504
  %25 = getelementptr float, ptr addrspace(6) null, i64 %24
  %26 = getelementptr float, ptr addrspace(1) %3, i64 %14
  call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV(ptr addrspace(1) %26, ptr addrspace(6) %25, i64 687194767376, i64 22517998136872960)
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
define linkonce_odr simt_entry void @rmsnorm_d5120_kernel_simt_0(ptr addrspace(6) %0) #3 !annotation !8 !annotation !9 {
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
  %75 = bitcast float %10 to i32
  %76 = bitcast float %11 to i32
  %77 = bitcast float %17 to i32
  %78 = bitcast float %18 to i32
  %79 = bitcast float %24 to i32
  %80 = bitcast float %25 to i32
  %81 = bitcast float %31 to i32
  %82 = bitcast float %32 to i32
  %83 = bitcast float %38 to i32
  %84 = bitcast float %39 to i32
  %85 = bitcast float %45 to i32
  %86 = bitcast float %46 to i32
  %87 = bitcast float %52 to i32
  %88 = bitcast float %53 to i32
  %89 = bitcast float %59 to i32
  %90 = bitcast float %60 to i32
  %91 = bitcast float %66 to i32
  %92 = bitcast float %67 to i32
  %93 = bitcast float %73 to i32
  %94 = bitcast float %74 to i32
  %95 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"(i32 %75, i32 %76, i32 %77, i32 %78, i32 %79, i32 %80, i32 %81, i32 %82, i32 %83, i32 %84, i32 %85, i32 %86, i32 %87, i32 %88, i32 %89, i32 %90, i32 %91, i32 %92, i32 %93, i32 %94)
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @rmsnorm_d5120_kernel_simt_1(i64 %0, ptr addrspace(6) %1, i64 %2, float %3, i64 %4) #3 !annotation !8 !annotation !9 {
  %6 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23}"()
  %7 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 0
  %8 = bitcast i32 %7 to float
  %9 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 1
  %10 = bitcast i32 %9 to float
  %11 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 2
  %12 = bitcast i32 %11 to float
  %13 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 3
  %14 = bitcast i32 %13 to float
  %15 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 4
  %16 = bitcast i32 %15 to float
  %17 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 5
  %18 = bitcast i32 %17 to float
  %19 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 6
  %20 = bitcast i32 %19 to float
  %21 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 7
  %22 = bitcast i32 %21 to float
  %23 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 8
  %24 = bitcast i32 %23 to float
  %25 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 9
  %26 = bitcast i32 %25 to float
  %27 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 10
  %28 = bitcast i32 %27 to float
  %29 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 11
  %30 = bitcast i32 %29 to float
  %31 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 12
  %32 = bitcast i32 %31 to float
  %33 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 13
  %34 = bitcast i32 %33 to float
  %35 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 14
  %36 = bitcast i32 %35 to float
  %37 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 15
  %38 = bitcast i32 %37 to float
  %39 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 16
  %40 = bitcast i32 %39 to float
  %41 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 17
  %42 = bitcast i32 %41 to float
  %43 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 18
  %44 = bitcast i32 %43 to float
  %45 = extractvalue { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } %6, 19
  %46 = bitcast i32 %45 to float
  %47 = alloca float, i32 32, align 4
  %48 = alloca float, align 4
  %49 = call i32 @llvm.hivm.get.TID.X()
  %50 = mul i32 %49, 2
  %51 = sext i32 %50 to i64
  %52 = add i64 %0, %51
  %53 = add i64 %52, 5120
  %54 = mul i64 %53, 4
  %55 = ptrtoint ptr addrspace(6) %1 to i64
  %56 = inttoptr i64 %55 to ptr addrspace(6)
  %57 = getelementptr i8, ptr addrspace(6) %56, i64 %54
  %58 = load <2 x float>, ptr addrspace(6) %57, align 8
  store <2 x float> %58, ptr %47, align 8
  %59 = add i64 %0, 512
  %60 = add i64 %59, %51
  %61 = add i64 %60, 5120
  %62 = mul i64 %61, 4
  %63 = getelementptr i8, ptr addrspace(6) %56, i64 %62
  %64 = load <2 x float>, ptr addrspace(6) %63, align 8
  %65 = getelementptr i8, ptr %47, i32 8
  store <2 x float> %64, ptr %65, align 8
  %66 = add i64 %0, 1024
  %67 = add i64 %66, %51
  %68 = add i64 %67, 5120
  %69 = mul i64 %68, 4
  %70 = getelementptr i8, ptr addrspace(6) %56, i64 %69
  %71 = load <2 x float>, ptr addrspace(6) %70, align 8
  %72 = getelementptr i8, ptr %47, i32 16
  store <2 x float> %71, ptr %72, align 8
  %73 = add i64 %0, 1536
  %74 = add i64 %73, %51
  %75 = add i64 %74, 5120
  %76 = mul i64 %75, 4
  %77 = getelementptr i8, ptr addrspace(6) %56, i64 %76
  %78 = load <2 x float>, ptr addrspace(6) %77, align 8
  %79 = getelementptr i8, ptr %47, i32 24
  store <2 x float> %78, ptr %79, align 8
  %80 = add i64 %0, 2048
  %81 = add i64 %80, %51
  %82 = add i64 %81, 5120
  %83 = mul i64 %82, 4
  %84 = getelementptr i8, ptr addrspace(6) %56, i64 %83
  %85 = load <2 x float>, ptr addrspace(6) %84, align 8
  %86 = getelementptr i8, ptr %47, i32 32
  store <2 x float> %85, ptr %86, align 8
  %87 = add i64 %0, 2560
  %88 = add i64 %87, %51
  %89 = add i64 %88, 5120
  %90 = mul i64 %89, 4
  %91 = getelementptr i8, ptr addrspace(6) %56, i64 %90
  %92 = load <2 x float>, ptr addrspace(6) %91, align 8
  %93 = getelementptr i8, ptr %47, i32 40
  store <2 x float> %92, ptr %93, align 8
  %94 = add i64 %0, 3072
  %95 = add i64 %94, %51
  %96 = add i64 %95, 5120
  %97 = mul i64 %96, 4
  %98 = getelementptr i8, ptr addrspace(6) %56, i64 %97
  %99 = load <2 x float>, ptr addrspace(6) %98, align 8
  %100 = getelementptr i8, ptr %47, i32 48
  store <2 x float> %99, ptr %100, align 8
  %101 = add i64 %0, 3584
  %102 = add i64 %101, %51
  %103 = add i64 %102, 5120
  %104 = mul i64 %103, 4
  %105 = getelementptr i8, ptr addrspace(6) %56, i64 %104
  %106 = load <2 x float>, ptr addrspace(6) %105, align 8
  %107 = getelementptr i8, ptr %47, i32 56
  store <2 x float> %106, ptr %107, align 8
  %108 = add i64 %0, 4096
  %109 = add i64 %108, %51
  %110 = add i64 %109, 5120
  %111 = mul i64 %110, 4
  %112 = getelementptr i8, ptr addrspace(6) %56, i64 %111
  %113 = load <2 x float>, ptr addrspace(6) %112, align 8
  %114 = getelementptr i8, ptr %47, i32 64
  store <2 x float> %113, ptr %114, align 8
  %115 = add i64 %0, 4608
  %116 = add i64 %115, %51
  %117 = add i64 %116, 5120
  %118 = mul i64 %117, 4
  %119 = getelementptr i8, ptr addrspace(6) %56, i64 %118
  %120 = load <2 x float>, ptr addrspace(6) %119, align 8
  %121 = getelementptr i8, ptr %47, i32 72
  store <2 x float> %120, ptr %121, align 8
  %122 = add i64 %2, %51
  %123 = add i64 %122, 5120
  %124 = mul i64 %123, 4
  %125 = getelementptr i8, ptr addrspace(6) %56, i64 %124
  %126 = load <2 x float>, ptr addrspace(6) %125, align 8
  %127 = getelementptr i8, ptr %47, i32 80
  store <2 x float> %126, ptr %127, align 8
  %128 = add i64 %0, 5632
  %129 = add i64 %128, %51
  %130 = add i64 %129, 5120
  %131 = mul i64 %130, 4
  %132 = getelementptr i8, ptr addrspace(6) %56, i64 %131
  %133 = load <2 x float>, ptr addrspace(6) %132, align 8
  %134 = getelementptr i8, ptr %47, i32 88
  store <2 x float> %133, ptr %134, align 8
  %135 = add i64 %0, 6144
  %136 = add i64 %135, %51
  %137 = add i64 %136, 5120
  %138 = mul i64 %137, 4
  %139 = getelementptr i8, ptr addrspace(6) %56, i64 %138
  %140 = load <2 x float>, ptr addrspace(6) %139, align 8
  %141 = getelementptr i8, ptr %47, i32 96
  store <2 x float> %140, ptr %141, align 8
  %142 = add i64 %0, 6656
  %143 = add i64 %142, %51
  %144 = add i64 %143, 5120
  %145 = mul i64 %144, 4
  %146 = getelementptr i8, ptr addrspace(6) %56, i64 %145
  %147 = load <2 x float>, ptr addrspace(6) %146, align 8
  %148 = getelementptr i8, ptr %47, i32 104
  store <2 x float> %147, ptr %148, align 8
  %149 = add i64 %0, 7168
  %150 = add i64 %149, %51
  %151 = add i64 %150, 5120
  %152 = mul i64 %151, 4
  %153 = getelementptr i8, ptr addrspace(6) %56, i64 %152
  %154 = load <2 x float>, ptr addrspace(6) %153, align 8
  %155 = getelementptr i8, ptr %47, i32 112
  store <2 x float> %154, ptr %155, align 8
  %156 = add i64 %0, 7680
  %157 = add i64 %156, %51
  %158 = add i64 %157, 5120
  %159 = mul i64 %158, 4
  %160 = getelementptr i8, ptr addrspace(6) %56, i64 %159
  %161 = load <2 x float>, ptr addrspace(6) %160, align 8
  %162 = getelementptr i8, ptr %47, i32 120
  store <2 x float> %161, ptr %162, align 8
  store float 0.000000e+00, ptr %48, align 4
  %163 = load float, ptr %48, align 4
  %164 = load float, ptr %47, align 4
  %165 = fmul float %164, %164
  %166 = fadd float %163, %165
  store float %166, ptr %48, align 4
  %167 = load float, ptr %48, align 4
  %168 = getelementptr i8, ptr %47, i32 4
  %169 = load float, ptr %168, align 4
  %170 = fmul float %169, %169
  %171 = fadd float %167, %170
  store float %171, ptr %48, align 4
  %172 = load float, ptr %48, align 4
  %173 = load float, ptr %65, align 4
  %174 = fmul float %173, %173
  %175 = fadd float %172, %174
  store float %175, ptr %48, align 4
  %176 = load float, ptr %48, align 4
  %177 = getelementptr i8, ptr %47, i32 12
  %178 = load float, ptr %177, align 4
  %179 = fmul float %178, %178
  %180 = fadd float %176, %179
  store float %180, ptr %48, align 4
  %181 = load float, ptr %48, align 4
  %182 = load float, ptr %72, align 4
  %183 = fmul float %182, %182
  %184 = fadd float %181, %183
  store float %184, ptr %48, align 4
  %185 = load float, ptr %48, align 4
  %186 = getelementptr i8, ptr %47, i32 20
  %187 = load float, ptr %186, align 4
  %188 = fmul float %187, %187
  %189 = fadd float %185, %188
  store float %189, ptr %48, align 4
  %190 = load float, ptr %48, align 4
  %191 = load float, ptr %79, align 4
  %192 = fmul float %191, %191
  %193 = fadd float %190, %192
  store float %193, ptr %48, align 4
  %194 = load float, ptr %48, align 4
  %195 = getelementptr i8, ptr %47, i32 28
  %196 = load float, ptr %195, align 4
  %197 = fmul float %196, %196
  %198 = fadd float %194, %197
  store float %198, ptr %48, align 4
  %199 = load float, ptr %48, align 4
  %200 = load float, ptr %86, align 4
  %201 = fmul float %200, %200
  %202 = fadd float %199, %201
  store float %202, ptr %48, align 4
  %203 = load float, ptr %48, align 4
  %204 = getelementptr i8, ptr %47, i32 36
  %205 = load float, ptr %204, align 4
  %206 = fmul float %205, %205
  %207 = fadd float %203, %206
  store float %207, ptr %48, align 4
  %208 = load float, ptr %48, align 4
  %209 = load float, ptr %93, align 4
  %210 = fmul float %209, %209
  %211 = fadd float %208, %210
  store float %211, ptr %48, align 4
  %212 = load float, ptr %48, align 4
  %213 = getelementptr i8, ptr %47, i32 44
  %214 = load float, ptr %213, align 4
  %215 = fmul float %214, %214
  %216 = fadd float %212, %215
  store float %216, ptr %48, align 4
  %217 = load float, ptr %48, align 4
  %218 = load float, ptr %100, align 4
  %219 = fmul float %218, %218
  %220 = fadd float %217, %219
  store float %220, ptr %48, align 4
  %221 = load float, ptr %48, align 4
  %222 = getelementptr i8, ptr %47, i32 52
  %223 = load float, ptr %222, align 4
  %224 = fmul float %223, %223
  %225 = fadd float %221, %224
  store float %225, ptr %48, align 4
  %226 = load float, ptr %48, align 4
  %227 = load float, ptr %107, align 4
  %228 = fmul float %227, %227
  %229 = fadd float %226, %228
  store float %229, ptr %48, align 4
  %230 = load float, ptr %48, align 4
  %231 = getelementptr i8, ptr %47, i32 60
  %232 = load float, ptr %231, align 4
  %233 = fmul float %232, %232
  %234 = fadd float %230, %233
  store float %234, ptr %48, align 4
  %235 = load float, ptr %48, align 4
  %236 = load float, ptr %114, align 4
  %237 = fmul float %236, %236
  %238 = fadd float %235, %237
  store float %238, ptr %48, align 4
  %239 = load float, ptr %48, align 4
  %240 = getelementptr i8, ptr %47, i32 68
  %241 = load float, ptr %240, align 4
  %242 = fmul float %241, %241
  %243 = fadd float %239, %242
  store float %243, ptr %48, align 4
  %244 = load float, ptr %48, align 4
  %245 = load float, ptr %121, align 4
  %246 = fmul float %245, %245
  %247 = fadd float %244, %246
  store float %247, ptr %48, align 4
  %248 = load float, ptr %48, align 4
  %249 = getelementptr i8, ptr %47, i32 76
  %250 = load float, ptr %249, align 4
  %251 = fmul float %250, %250
  %252 = fadd float %248, %251
  store float %252, ptr %48, align 4
  %253 = load float, ptr %48, align 4
  %254 = getelementptr float, ptr addrspace(6) %1, i64 37888
  %255 = sdiv i32 %49, 32
  %256 = mul i32 %255, 32
  %257 = icmp ne i32 %49, %256
  %258 = icmp slt i32 %49, 0
  %259 = icmp ne i1 %258, false
  %260 = and i1 %257, %259
  %261 = add i32 %255, -1
  %262 = select i1 %260, i32 %261, i32 %255
  %263 = call i32 @llvm.hivm.get.laneID()
  %264 = call float @llvm.hivm.redux.add.f32(float %253)
  %265 = icmp slt i32 %263, 1
  br i1 %265, label %266, label %270

266:                                              ; preds = %5
  %267 = add i32 %262, %263
  %268 = sext i32 %267 to i64
  %269 = getelementptr float, ptr addrspace(6) %254, i64 %268
  store float %264, ptr addrspace(6) %269, align 4
  br label %270

270:                                              ; preds = %266, %5
  call void @llvm.hivm.sync.workitems()
  %271 = icmp slt i32 %49, 32
  br i1 %271, label %272, label %279

272:                                              ; preds = %270
  %273 = icmp slt i32 %263, 8
  %274 = sext i32 %263 to i64
  %275 = getelementptr float, ptr addrspace(6) %254, i64 %274
  %276 = load float, ptr addrspace(6) %275, align 4
  %277 = select i1 %273, float %276, float 0.000000e+00
  %278 = call float @llvm.hivm.redux.add.f32(float %277)
  br label %280

279:                                              ; preds = %270
  br label %280

280:                                              ; preds = %272, %279
  %281 = phi float [ 0.000000e+00, %279 ], [ %278, %272 ]
  br label %282

282:                                              ; preds = %280
  %283 = icmp slt i32 %49, 1
  br i1 %283, label %284, label %287

284:                                              ; preds = %282
  %285 = sext i32 %49 to i64
  %286 = getelementptr float, ptr addrspace(6) %254, i64 %285
  store float %281, ptr addrspace(6) %286, align 4
  br label %287

287:                                              ; preds = %284, %282
  call void @llvm.hivm.sync.workitems()
  %288 = getelementptr float, ptr addrspace(6) %254, i64 0
  %289 = load float, ptr addrspace(6) %288, align 4
  call void @llvm.hivm.sync.workitems()
  store float %289, ptr %48, align 4
  %290 = load float, ptr %48, align 4
  %291 = fdiv float %290, 5.120000e+03
  %292 = fadd float %291, %3
  %293 = call float @llvm.sqrt.f32(float %292)
  %294 = fdiv float 1.000000e+00, %293
  %295 = mul i64 %4, 8
  %296 = getelementptr float, ptr addrspace(6) %1, i64 %295
  store float %294, ptr addrspace(6) %296, align 4
  %297 = load <2 x float>, ptr %47, align 8
  %298 = insertelement <2 x float> undef, float %294, i32 0
  %299 = insertelement <2 x float> %298, float %294, i32 1
  %300 = fmul <2 x float> %297, %299
  %301 = insertelement <2 x float> poison, float %8, i32 0
  %302 = insertelement <2 x float> %301, float %10, i32 1
  %303 = fmul <2 x float> %300, %302
  %304 = add i64 %52, 21504
  %305 = mul i64 %304, 4
  %306 = getelementptr i8, ptr addrspace(6) %56, i64 %305
  store <2 x float> %303, ptr addrspace(6) %306, align 8
  %307 = load <2 x float>, ptr %65, align 8
  %308 = fmul <2 x float> %307, %299
  %309 = insertelement <2 x float> poison, float %12, i32 0
  %310 = insertelement <2 x float> %309, float %14, i32 1
  %311 = fmul <2 x float> %308, %310
  %312 = add i64 %60, 21504
  %313 = mul i64 %312, 4
  %314 = getelementptr i8, ptr addrspace(6) %56, i64 %313
  store <2 x float> %311, ptr addrspace(6) %314, align 8
  %315 = load <2 x float>, ptr %72, align 8
  %316 = fmul <2 x float> %315, %299
  %317 = insertelement <2 x float> poison, float %16, i32 0
  %318 = insertelement <2 x float> %317, float %18, i32 1
  %319 = fmul <2 x float> %316, %318
  %320 = add i64 %67, 21504
  %321 = mul i64 %320, 4
  %322 = getelementptr i8, ptr addrspace(6) %56, i64 %321
  store <2 x float> %319, ptr addrspace(6) %322, align 8
  %323 = load <2 x float>, ptr %79, align 8
  %324 = fmul <2 x float> %323, %299
  %325 = insertelement <2 x float> poison, float %20, i32 0
  %326 = insertelement <2 x float> %325, float %22, i32 1
  %327 = fmul <2 x float> %324, %326
  %328 = add i64 %74, 21504
  %329 = mul i64 %328, 4
  %330 = getelementptr i8, ptr addrspace(6) %56, i64 %329
  store <2 x float> %327, ptr addrspace(6) %330, align 8
  %331 = load <2 x float>, ptr %86, align 8
  %332 = fmul <2 x float> %331, %299
  %333 = insertelement <2 x float> poison, float %24, i32 0
  %334 = insertelement <2 x float> %333, float %26, i32 1
  %335 = fmul <2 x float> %332, %334
  %336 = add i64 %81, 21504
  %337 = mul i64 %336, 4
  %338 = getelementptr i8, ptr addrspace(6) %56, i64 %337
  store <2 x float> %335, ptr addrspace(6) %338, align 8
  %339 = load <2 x float>, ptr %93, align 8
  %340 = fmul <2 x float> %339, %299
  %341 = insertelement <2 x float> poison, float %28, i32 0
  %342 = insertelement <2 x float> %341, float %30, i32 1
  %343 = fmul <2 x float> %340, %342
  %344 = add i64 %88, 21504
  %345 = mul i64 %344, 4
  %346 = getelementptr i8, ptr addrspace(6) %56, i64 %345
  store <2 x float> %343, ptr addrspace(6) %346, align 8
  %347 = load <2 x float>, ptr %100, align 8
  %348 = fmul <2 x float> %347, %299
  %349 = insertelement <2 x float> poison, float %32, i32 0
  %350 = insertelement <2 x float> %349, float %34, i32 1
  %351 = fmul <2 x float> %348, %350
  %352 = add i64 %95, 21504
  %353 = mul i64 %352, 4
  %354 = getelementptr i8, ptr addrspace(6) %56, i64 %353
  store <2 x float> %351, ptr addrspace(6) %354, align 8
  %355 = load <2 x float>, ptr %107, align 8
  %356 = fmul <2 x float> %355, %299
  %357 = insertelement <2 x float> poison, float %36, i32 0
  %358 = insertelement <2 x float> %357, float %38, i32 1
  %359 = fmul <2 x float> %356, %358
  %360 = add i64 %102, 21504
  %361 = mul i64 %360, 4
  %362 = getelementptr i8, ptr addrspace(6) %56, i64 %361
  store <2 x float> %359, ptr addrspace(6) %362, align 8
  %363 = load <2 x float>, ptr %114, align 8
  %364 = fmul <2 x float> %363, %299
  %365 = insertelement <2 x float> poison, float %40, i32 0
  %366 = insertelement <2 x float> %365, float %42, i32 1
  %367 = fmul <2 x float> %364, %366
  %368 = add i64 %109, 21504
  %369 = mul i64 %368, 4
  %370 = getelementptr i8, ptr addrspace(6) %56, i64 %369
  store <2 x float> %367, ptr addrspace(6) %370, align 8
  %371 = load <2 x float>, ptr %121, align 8
  %372 = fmul <2 x float> %371, %299
  %373 = insertelement <2 x float> poison, float %44, i32 0
  %374 = insertelement <2 x float> %373, float %46, i32 1
  %375 = fmul <2 x float> %372, %374
  %376 = add i64 %116, 21504
  %377 = mul i64 %376, 4
  %378 = getelementptr i8, ptr addrspace(6) %56, i64 %377
  store <2 x float> %375, ptr addrspace(6) %378, align 8
  %379 = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "", "={TPER4},={TPER5},={TPER6},={TPER7},={TPER8},={TPER9},={TPER10},={TPER11},={TPER12},={TPER13},={TPER14},={TPER15},={TPER16},={TPER17},={TPER18},={TPER19},={TPER20},={TPER21},={TPER22},={TPER23},0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"(i32 %7, i32 %9, i32 %11, i32 %13, i32 %15, i32 %17, i32 %19, i32 %21, i32 %23, i32 %25, i32 %27, i32 %29, i32 %31, i32 %33, i32 %35, i32 %37, i32 %39, i32 %41, i32 %43, i32 %45)
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #2 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #3 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @rmsnorm_d5120_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @rmsnorm_d5120_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = !{ptr @rmsnorm_d5120_kernel_mix_aiv, !"kernel_with_simt", i32 1}
!4 = distinct !{null, !"simt-max-threads", i32 256}
!5 = distinct !{null, !"simt-max-registers", i32 128}
!6 = distinct !{null, !"simt-max-threads", i32 256}
!7 = distinct !{null, !"simt-max-registers", i32 128}
!8 = !{!"simt-max-threads", i32 256}
!9 = !{!"simt-max-registers", i32 128}
