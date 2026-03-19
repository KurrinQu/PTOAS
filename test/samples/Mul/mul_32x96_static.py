from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sample_utils.tiled_elementwise_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="mul_kernel_32x96_static",
        op_name="mul",
        rows=32,
        cols=96,
        dynamic_shape=False,
    )


if __name__ == "__main__":
    print(build())
