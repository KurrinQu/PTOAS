; ModuleID = 'tid-plain-simt-sequence-repro'
; kernel2：SIMT 内直接读 tid（llvm.hivm.get.TID.X）并写 GM，不含 inline asm。
; 作为 kernel1 的受害者：kernel1 的 VF_SIMT 声明 0xffff 范围后，本 kernel 的首次
; SIMT dispatch 取指异常。
source_filename = "tid_plain_simt_repro"

; Unknown intrinsic
declare void @llvm.hivm.store.vfsimt.info(i64) #0

; Unknown intrinsic
declare i32 @llvm.hivm.get.TID.X() #0

define void @tid_plain_kernel_mix_aiv(ptr addrspace(1) %0) #2 {
  call void @llvm.hivm.store.vfsimt.info(i64 4295032960)
  call simt_entry void @tid_plain_kernel_simt_0(ptr addrspace(1) %0)
  ret void
}

; Function Attrs: noinline
define linkonce_odr simt_entry void @tid_plain_kernel_simt_0(ptr addrspace(1) %0) #3 !annotation !8 !annotation !9 {
  %tid = call i32 @llvm.hivm.get.TID.X()
  %p = getelementptr i32, ptr addrspace(1) %0, i32 %tid
  store i32 %tid, ptr addrspace(1) %p, align 4
  ret void
}

attributes #0 = { "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #2 = { "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }
attributes #3 = { noinline "target-cpu"="dav-c310-vec" "target-features"="+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,+MOVX8,+MSTX,+SPR7bits,+SyncV,+dav-c310-vec" }

!llvm.module.flags = !{!0}
!hivm.annotations = !{!1, !2, !3, !4, !5, !6, !7}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{ptr @tid_plain_kernel_mix_aiv, !"kernel", i32 1}
!2 = !{ptr @tid_plain_kernel_mix_aiv, !"kernel_with_simd", i32 1}
!3 = !{ptr @tid_plain_kernel_mix_aiv, !"kernel_with_simt", i32 1}
!4 = distinct !{null, !"simt-max-threads", i32 128}
!5 = distinct !{null, !"simt-max-registers", i32 128}
!6 = distinct !{null, !"simt-max-threads", i32 128}
!7 = distinct !{null, !"simt-max-registers", i32 128}
!8 = !{!"simt-max-threads", i32 128}
!9 = !{!"simt-max-registers", i32 128}
