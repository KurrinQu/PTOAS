from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sample_utils.tile_fusion_cases import build_scalar_chain_case


def build():
    return build_scalar_chain_case(kernel_name="tile_fusion_scalar_kernel_32x96_static")


if __name__ == "__main__":
    print(build())
