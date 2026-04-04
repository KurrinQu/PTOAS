"""Flattened TileLang DSL advanced-mode version of A5 `TADD_IMPL`.

This example mirrors the user-facing `TADD_IMPL -> TAdd -> BinaryInstr ->
TBinOps_2D_NoPostUpdate` flow from `pto/npu/a5/TAdd.hpp`, but spells the final
2D row-major vector body directly in Python:

- top-level interface uses `dst, src0, src1` Tile parameters like `TADD`
- `advanced=True` enables implicit `pto.vecscope` inference
- `pto.vlds(tile[row, col:])` / `pto.vsts(vec, tile[row, col:], mask)` use
  tile indexing sugar instead of manual offset arithmetic
"""

import sys
from pathlib import Path


def _import_tilelang_dsl():
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "tilelang-dsl" / "python",
        repo_root / "build" / "python",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
    import tilelang_dsl as pto

    return pto


pto = _import_tilelang_dsl()


@pto.vkernel(
    op="tadd",
    dtypes=[(pto.f32, pto.f32, pto.f32)],
    advanced=True,
    name="tilelang_advanced_tadd_demo",
)
def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    # Flattened equivalent of the TAddCheck/TADD_IMPL parameter plumbing.
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return None


def build_specialized_kernel():
    return kernel.specialize(
        dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
    )


def main(argv) -> int:
    specialized = build_specialized_kernel()

    if len(argv) > 2:
        print(f"usage: {Path(argv[0]).name} [output.mlir]", file=sys.stderr)
        return 2

    if len(argv) == 2:
        output_path = Path(argv[1])
        specialized.emit(output_path)
        print(f"wrote MLIR to {output_path}")
        return 0

    print(specialized.mlir_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
