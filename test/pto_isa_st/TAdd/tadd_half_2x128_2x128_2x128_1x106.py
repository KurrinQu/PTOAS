from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="tadd_half_2x128_2x128_2x128_1x106_kernel",
        op_name="tadd",
        dtype_token="f16",
        dst_shape=(2, 128),
        src0_shape=(2, 128),
        src1_shape=(2, 128),
        valid_shape=(1, 106),
    )


if __name__ == "__main__":
    print(build())
