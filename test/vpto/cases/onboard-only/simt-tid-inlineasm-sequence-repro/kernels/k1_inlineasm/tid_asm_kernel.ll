; ModuleID = 'tid-asm-simt-sequence-repro'
; kernel1：SIMT 内用空 inline asm（keep/resume 形态，fixed TPER 约束）传递 tid 并写 GM。
; 空 inline asm 触发 BiSheng function-size = -1 -> VF_SIMT code-size = 0xffff。
; scalar 侧循环 dispatch 64 次（足够的 dispatch 次数是污染触发的必要条件）。
source_filename = "tid_asm_simt_repro"

; Unknown intrinsic
declare void @llvm.hivm.store.vfsimt.info(i64) #0

; Unknown intrinsic
declare i32 @llvm.hivm.get.TID.X() #0

define void @tid_asm_kernel_mix_aiv(ptr addrspace(1) %0) #2 {
entry:
  br label %loop

loop:
  %t = phi i64 [ 0, %entry ], [ %tn, %body ]
  %cond = icmp slt i64 %t, 64
  br i1 %cond, label %body, label %exit

body:
  call void @llvm.hivm.store.vfsimt.info(i64 4295032960)
  call simt_entry void @tid_asm_kernel_simt_0(ptr addrspace(1) %0)
  %tn = add i64 %t, 1
  br label %loop

exit:
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @tid_asm_kernel_simt_0(ptr addrspace(1) %0) #3 !annotation !8 !annotation !9 {
entry:
  %tid = call i32 @llvm.hivm.get.TID.X()
  ; keep: 空 asm 模板，不生成设备指令；tid 绑定到固定物理寄存器 TPER4
  %k = call i32 asm sideeffect "", "={TPER4},0"(i32 %tid)
  ; resume: 经空 inline asm 从 TPER4 取回 tid
  %tid2 = call i32 asm sideeffect "", "={TPER4}"()
  %p = getelementptr i32, ptr addrspace(1) %0, i32 %tid2
  store i32 %tid2, ptr addrspace(1) %p, align 4
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #2 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #3 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @tid_asm_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @tid_asm_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = !{ptr @tid_asm_kernel_mix_aiv, !"kernel_with_simt", i32 1}
!4 = distinct !{null, !"simt-max-threads", i32 128}
!5 = distinct !{null, !"simt-max-registers", i32 128}
!6 = distinct !{null, !"simt-max-threads", i32 128}
!7 = distinct !{null, !"simt-max-registers", i32 128}
!8 = !{!"simt-max-threads", i32 128}
!9 = !{!"simt-max-registers", i32 128}
