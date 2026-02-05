#!/usr/bin/env python3
# Emits a fixed MLIR module exercising high-level sync ops
IR = r"""
module {
  func.func @run_sync_high() {
    pto.record_event[#pto.sync_op_type<TLOAD>, #pto.sync_op_type<TMATMUL>, #pto.event<EVENT_ID0>]
    pto.wait_event[#pto.sync_op_type<TLOAD>, #pto.sync_op_type<TMATMUL>, #pto.event<EVENT_ID0>]
    return
  }
}
"""
print(IR)
