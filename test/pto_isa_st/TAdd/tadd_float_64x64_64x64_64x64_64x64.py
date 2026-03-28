from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="tadd_float_64x64_64x64_64x64_64x64_kernel",
        op_name="tadd",
        dtype_token="f32",
        dst_shape=(64, 64),
        src0_shape=(64, 64),
        src1_shape=(64, 64),
        valid_shape=(64, 64),
    )


if __name__ == "__main__":
    print(build())
