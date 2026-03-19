from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tdivs_float_256x32_256x16_256x16_kernel",
        op_name="tdivs",
        dtype_token="f32",
        dst_shape=(256, 32),
        src_shape=(256, 16),
        valid_shape=(256, 16),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
