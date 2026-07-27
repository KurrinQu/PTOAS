from ptodsl import pto, scalar
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value

@pto.jit(name="rmsnorm_d5120_kernel", kernel_kind="vector", target="a5", mode="explicit")
def rmsnorm_d5120_kernel(RSTD: pto.ptr(pto.f32, "gm"), W: pto.ptr(pto.f32, "gm"), X: pto.ptr(pto.f32, "gm"), Y: pto.ptr(pto.f32, "gm"), eps: pto.f32):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.i8, "ub"))
  w_frag = [None] * 32
  w_frag_1 = pto.alloc_buffer((32,), pto.f32)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.mte_gm_ub(W, pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 0, 20480, nburst=(1, 20480, 20480))
  pto.set_flag("MTE2", "V", event_id=2)
  pto.wait_flag("MTE2", "V", event_id=2)
  with pto.simt(256, 1, 1):
    simtvf_tx = pto.get_tid_x()
    simtvf_ty = pto.get_tid_y()
    simtvf_tz = pto.get_tid_z()
    for i in pto.static_range(0, 16):
      if i < 10:
        scalar.store(scalar.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (i * 512) + (simtvf_tx * 2), contiguous=2), w_frag_1, i * 2)
  with pto.for_(0, 64, step=1) as t:
    pto.wait_flag("V", "MTE2", event_id=t & 1)
    pto.mte_gm_ub(pto.addptr(X, (t * 327680) + (pto.get_block_idx() * 5120)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((t & 1) * 8192) + 5120), 0, 20480, nburst=(1, 20480, 20480))
    pto.set_flag("MTE2", "V", event_id=t & 1)
    pto.wait_flag("MTE3", "V", event_id=t & 1)
    pto.wait_flag("MTE2", "V", event_id=t & 1)
    with pto.simt(256, 1, 1):
      x_frag = pto.alloc_buffer((32,), pto.f32)
      sum_sq = pto.alloc_buffer((1,), pto.f32)
      simtvf_tx_1 = pto.get_tid_x()
      simtvf_ty_1 = pto.get_tid_y()
      simtvf_tz_1 = pto.get_tid_z()
      for i_1 in pto.static_range(0, 16):
        scalar.store(scalar.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((t & 1) * 8192) + (i_1 * 512)) + (simtvf_tx_1 * 2)) + 5120, contiguous=2), x_frag, i_1 * 2)
      scalar.store(float.fromhex('0x0p+0'), sum_sq, 0)
      for i_2 in pto.static_range(0, 32):
        if i_2 < 20:
          scalar.store(scalar.load(sum_sq, 0) + (scalar.load(x_frag, i_2) * scalar.load(x_frag, i_2)), sum_sq, 0)
      scalar.store(pto.simt_allreduce_sum(scalar.load(sum_sq, 0), threads=256, scale=1, thread_offset=0, scratch=pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 37888)), sum_sq, 0)
      var = (scalar.load(sum_sq, 0) / float.fromhex('0x1.4p+12')) + eps
      rstd_val = float.fromhex('0x1p+0') / pto.sqrt(var)
      scalar.store(float.fromhex('0x1p+0') / pto.sqrt(var), pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (t & 1) * 8)
      for i_3 in pto.static_range(0, 16):
        if i_3 < 10:
          scalar.store((scalar.load(x_frag, i_3 * 2, contiguous=2) * pto.Vec(pto.f32, 2, init=(float.fromhex('0x1p+0') / pto.sqrt(var)))) * scalar.load(w_frag_1, i_3 * 2, contiguous=2), pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((t & 1) * 8192) + (i_3 * 512)) + (simtvf_tx_1 * 2)) + 21504)
    pto.set_flag("V", "MTE3", event_id=t & 1)
    pto.set_flag("V", "MTE2", event_id=t & 1)
    pto.wait_flag("V", "MTE3", event_id=t & 1)
    pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (t & 1) * 8), pto.addptr(RSTD, (t * 64) + pto.get_block_idx()), 4, nburst=(1, 4, 4))
    pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((t & 1) * 8192) + 21504), pto.addptr(Y, (t * 327680) + (pto.get_block_idx() * 5120)), 20480, nburst=(1, 20480, 20480))
    pto.set_flag("MTE3", "V", event_id=t & 1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
