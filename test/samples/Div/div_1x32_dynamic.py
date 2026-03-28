from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sample_utils.tiled_elementwise_cases import build_binary_case


def build():
    return build_binary_case(
        kernel_name="div_kernel_1x32_dynamic",
        op_name="div",
        rows=1,
        cols=32,
        dynamic_shape=True,
    )


if __name__ == "__main__":
    print(build())
