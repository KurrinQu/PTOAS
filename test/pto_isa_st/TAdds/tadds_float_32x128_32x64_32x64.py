from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tadds_float_32x128_32x64_32x64_kernel",
        op_name="tadds",
        dtype_token="f32",
        dst_shape=(32, 128),
        src_shape=(32, 64),
        valid_shape=(32, 64),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
