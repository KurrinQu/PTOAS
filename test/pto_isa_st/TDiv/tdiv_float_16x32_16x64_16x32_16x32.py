from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="tdiv_float_16x32_16x64_16x32_16x32_kernel",
        op_name="tdiv",
        dtype_token="f32",
        dst_shape=(16, 32),
        src0_shape=(16, 64),
        src1_shape=(16, 32),
        valid_shape=(16, 32),
    )


if __name__ == "__main__":
    print(build())
