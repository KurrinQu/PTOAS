from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tmins_float_64x64_32x32_32x32_kernel",
        op_name="tmins",
        dtype_token="f32",
        dst_shape=(64, 64),
        src_shape=(32, 32),
        valid_shape=(32, 32),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
