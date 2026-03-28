from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="tmax_int16_32x128_32x128_32x256_32x128_kernel",
        op_name="tmax",
        dtype_token="i16",
        dst_shape=(32, 128),
        src0_shape=(32, 128),
        src1_shape=(32, 256),
        valid_shape=(32, 128),
    )


if __name__ == "__main__":
    print(build())
