from ptodsl import pto, scalar

@pto.simt(name="simt_vf_2", max_threads=128, max_regs=64)
def simt_vf_2(buf_dyn_shmem: pto.ptr(pto.i8, "ub"), t: pto.i32, eps: pto.f32):
    simtvf_tx = pto.get_tid_x()
    simtvf_ty = pto.get_tid_y()
    simtvf_tz = pto.get_tid_z()
    x_frag = pto.alloc_buffer((32,), pto.f32)
    sum_sq = pto.alloc_buffer((1,), pto.f32)
    for i in pto.static_range(0, 16):
      load_vec = scalar.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((t & 1) * 4096) + (i * 256)) + (simtvf_tx * 2)) + 4224, contiguous=2)
      scalar.store(load_vec, x_frag, i * 2)
    scalar.store(float.fromhex('0x0p+0'), sum_sq, 0)
    for i_1 in pto.static_range(0, 32):
      load_val = scalar.load(sum_sq, 0)
      load_val_1 = scalar.load(x_frag, i_1)
      mul_val = load_val_1 * load_val_1
      add_val = load_val + mul_val
      scalar.store(add_val, sum_sq, 0)
    load_val_2 = scalar.load(sum_sq, 0)
    value_val = pto.simt_allreduce_sum(load_val_2, threads=128, scale=1, thread_offset=0, scratch=pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 4096))
    scalar.store(value_val, sum_sq, 0)
    load_val_3 = scalar.load(sum_sq, 0)
    div_val = load_val_3 / float.fromhex('0x1p+12')
    add_val_1 = div_val + eps
    var = add_val_1
    sqrt_val = pto.sqrt(var)
    rsqrt_val = 1.0 / sqrt_val
    rstd_val = rsqrt_val
    scalar.store(rsqrt_val, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((t & 1) * 8) + 20608)
    for i_2 in pto.static_range(0, 16):
      load_vec_1 = scalar.load(x_frag, i_2 * 2, contiguous=2)
      broadcast_vec = pto.vec(pto.f32, 2, init=rsqrt_val)
      mul_vec = load_vec_1 * broadcast_vec
      load_vec_2 = scalar.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (i_2 * 256) + (simtvf_tx * 2), contiguous=2)
      mul_vec_1 = mul_vec * load_vec_2
      scalar.store(mul_vec_1, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((t & 1) * 4096) + (i_2 * 256)) + (simtvf_tx * 2)) + 12416)

@pto.jit(name="main_kernel", kernel_kind="vector", target="a5", mode="explicit", dyn_shared_memory_buf=82496)
def main_kernel(RSTD: pto.ptr(pto.f32, "gm"), W: pto.ptr(pto.f32, "gm"), X: pto.ptr(pto.f32, "gm"), Y: pto.ptr(pto.f32, "gm"), eps: pto.f32):
  bx = pto.get_block_idx()
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i8, "ub"))
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.mte_gm_ub(W, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 0, 16384, nburst=(1, 16384, 16384))
  pto.set_flag("MTE2", "V", event_id=2)
  for t in range(0, 67):
    if 3 <= t:
      pto.wait_flag("V", "MTE3", event_id=(t + 1) & 1)
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (((t + 1) & 1) * 8) + 20608), pto.addptr(RSTD, ((t * 64) + bx) - 192), 4, nburst=(1, 4, 4))
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (((t + 1) & 1) * 4096) + 12416), pto.addptr(Y, ((t * 262144) + (bx * 4096)) - 786432), 16384, nburst=(1, 16384, 16384))
      pto.set_flag("MTE3", "V", event_id=(t + 1) & 1)
    if (t < 66) & (2 <= t):
      if t == 2:
        pto.wait_flag("MTE2", "V", event_id=2)
      pto.wait_flag("MTE3", "V", event_id=t & 1)
      pto.wait_flag("MTE2", "V", event_id=t & 1)
      simt_vf_2[128, 1, 1](buf_dyn_shmem, t, eps)
      pto.set_flag("V", "MTE3", event_id=t & 1)
      pto.set_flag("V", "MTE2", event_id=t & 1)
    if t < 64:
      pto.wait_flag("V", "MTE2", event_id=t & 1)
      pto.mte_gm_ub(pto.addptr(X, (t * 262144) + (bx * 4096)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((t & 1) * 4096) + 4224), 0, 16384, nburst=(1, 16384, 16384))
      pto.set_flag("MTE2", "V", event_id=t & 1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)