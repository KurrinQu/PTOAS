from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tmaxs_uint16_32x128_32x128_32x128_kernel",
        op_name="tmaxs",
        dtype_token="u16",
        dst_shape=(32, 128),
        src_shape=(32, 128),
        valid_shape=(32, 128),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
