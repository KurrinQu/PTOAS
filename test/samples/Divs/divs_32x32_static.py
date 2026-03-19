from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sample_utils.tiled_elementwise_cases import build_scalar_case


def build():
    return build_scalar_case(
        kernel_name="divs_kernel_32x32_static",
        op_name="divs",
        rows=32,
        cols=32,
        dynamic_shape=False,
    )


if __name__ == "__main__":
    print(build())
