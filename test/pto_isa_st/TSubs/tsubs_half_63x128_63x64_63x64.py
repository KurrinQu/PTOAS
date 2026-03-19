from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sample_utils.pto_isa_st_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="tsubs_half_63x128_63x64_63x64_kernel",
        op_name="tsubs",
        dtype_token="f16",
        dst_shape=(63, 128),
        src_shape=(63, 64),
        valid_shape=(63, 64),
        pad_value="null",
    )


if __name__ == "__main__":
    print(build())
