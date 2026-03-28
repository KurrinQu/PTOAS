from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tmins_float_1x3600_2x4096_1x3600_padmax_kernel",
        op_name="tmins",
        dtype_token="f32",
        dst_shape=(1, 3600),
        src_shape=(2, 4096),
        valid_shape=(1, 3600),
        pad_value="max",
    )


if __name__ == "__main__":
    print(build())
