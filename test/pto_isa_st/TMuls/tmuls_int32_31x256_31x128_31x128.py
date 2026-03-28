from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tmuls_int32_31x256_31x128_31x128_kernel",
        op_name="tmuls",
        dtype_token="i32",
        dst_shape=(31, 256),
        src_shape=(31, 128),
        valid_shape=(31, 128),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
