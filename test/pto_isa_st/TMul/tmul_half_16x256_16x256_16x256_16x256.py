from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="tmul_half_16x256_16x256_16x256_16x256_kernel",
        op_name="tmul",
        dtype_token="f16",
        dst_shape=(16, 256),
        src0_shape=(16, 256),
        src1_shape=(16, 256),
        valid_shape=(16, 256),
    )


if __name__ == "__main__":
    print(build())
