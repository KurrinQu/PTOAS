import unittest

import tilelang_dsl as pto
from tilelang_dsl import (
    f4e1m2x2,
    f4e2m1x2,
    f8e4m3,
    f8e5m2,
    hif8,
    is_low_precision_dtype,
    is_packed_dtype,
)
from tilelang_dsl.semantic import _classify_vcvt_elem_kind
from tilelang_dsl.support_matrix import SUPPORTED_TOPLEVEL_PTO_CALLS
from tilelang_dsl.types import is_float_dtype


LOW_PRECISION_DTYPES = (
    (f8e4m3, "f8E4M3FN"),
    (f8e5m2, "f8E5M2"),
    (hif8, "!pto.hif8"),
    (f4e1m2x2, "!pto.f4E1M2x2"),
    (f4e2m1x2, "!pto.f4E2M1x2"),
)


def _tile_shape(dtype: pto.ScalarType) -> tuple[int, int]:
    return (8, pto.get_lanes(dtype))


class LowPrecisionDTypeTests(unittest.TestCase):
    def test_runtime_helpers_classify_low_precision_dtypes(self) -> None:
        for dtype, _ in LOW_PRECISION_DTYPES:
            with self.subTest(dtype=dtype.name):
                self.assertEqual(pto.bytewidth(dtype), 1)
                self.assertEqual(pto.get_lanes(dtype), 256)
                self.assertEqual(pto.elements_per_vreg(dtype), 256)
                self.assertTrue(is_low_precision_dtype(dtype))
                self.assertFalse(is_float_dtype(dtype))

        self.assertTrue(is_packed_dtype(f4e1m2x2))
        self.assertTrue(is_packed_dtype(f4e2m1x2))
        self.assertFalse(is_packed_dtype(f8e4m3))
        self.assertFalse(is_packed_dtype(hif8))

    def test_package_exports_and_support_matrix_include_low_precision_dtypes(self) -> None:
        self.assertIs(pto.f8e4m3, f8e4m3)
        self.assertIs(pto.f8e5m2, f8e5m2)
        self.assertIs(pto.hif8, hif8)
        self.assertIs(pto.f4e1m2x2, f4e1m2x2)
        self.assertIs(pto.f4e2m1x2, f4e2m1x2)
        self.assertTrue(hasattr(pto, "is_low_precision_dtype"))
        self.assertTrue(hasattr(pto, "is_packed_dtype"))

        for dtype, _ in LOW_PRECISION_DTYPES:
            with self.subTest(dtype=dtype.name):
                self.assertIn(dtype.name, SUPPORTED_TOPLEVEL_PTO_CALLS)

    def test_pad_value_max_and_min_reject_low_precision_dtypes(self) -> None:
        for pad_value in (pto.PadValue.MAX, pto.PadValue.MIN):
            for dtype, _ in LOW_PRECISION_DTYPES:
                with self.subTest(pad_value=pad_value.name, dtype=dtype.name):
                    with self.assertRaises(TypeError):
                        pad_value.eval(dtype)

    def test_pad_value_zero_null_and_custom_f32_continue_to_work(self) -> None:
        custom = pto.PadValue.custom_f32(0x3F800000)
        self.assertIsNone(pto.PadValue.NULL.eval(hif8))
        self.assertEqual(pto.PadValue.ZERO.eval(hif8), 0)
        self.assertAlmostEqual(custom.eval(hif8), 1.0)

    def test_tile_specialization_lowers_all_low_precision_dtype_names(self) -> None:
        for dtype, mlir_name in LOW_PRECISION_DTYPES:
            with self.subTest(dtype=dtype.name):
                @pto.vkernel(
                    op=f"low_precision_tile_buf_{dtype.name}_unique",
                    dtypes=[(dtype,)],
                )
                def kernel(tile: pto.Tile):
                    return None

                text = kernel.specialize(
                    tile=pto.TileSpecialization(
                        shape=_tile_shape(dtype),
                        memory_space=pto.MemorySpace.UB,
                    )
                ).mlir_text()
                self.assertIn(f"dtype={mlir_name}", text)

    def test_pointer_and_scalar_lowering_use_low_precision_mlir_type_names(self) -> None:
        @pto.vkernel(
            op="low_precision_ptr_hif8_unique",
            dtypes=[(pto.hif8,)],
            advanced=True,
        )
        def kernel(src: pto.ptr(pto.hif8, pto.MemorySpace.UB)):
            _ = pto.load_scalar(src, 0)
            return None

        text = kernel.mlir_text()
        self.assertIn("!pto.ptr<!pto.hif8, ub>", text)
        self.assertIn("-> !pto.hif8", text)

    def test_tensor_view_and_partition_view_lower_low_precision_dtype_names(self) -> None:
        @pto.vkernel(op="low_precision_partition_view_hif8_unique", dtypes=[(pto.hif8,)])
        def kernel(inp: pto.TensorView):
            part = inp[0:8, 0:16]
            rows, cols = part.shape
            if rows != 0 and cols != 0:
                rows = cols
            return None

        text = kernel.mlir_text()
        self.assertIn("!pto.tensor_view<?x?x?x?x?x!pto.hif8>", text)
        self.assertIn("-> !pto.partition_tensor_view<8x16x!pto.hif8>", text)

    def test_low_precision_vreg_and_make_mask_lower_to_b8_vectors(self) -> None:
        @pto.vkernel(
            op="low_precision_vreg_f8e4m3_unique",
            dtypes=[(pto.f8e4m3,)],
            advanced=True,
        )
        def kernel(tile: pto.Tile):
            dtype = tile.element_type
            mask = pto.make_mask(dtype, pto.PAT.ALL)
            vec: pto.vreg(dtype) = pto.vlds(tile, 0)
            pto.vsts(vec, tile, 0, mask)
            return None

        text = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=_tile_shape(pto.f8e4m3),
                memory_space=pto.MemorySpace.UB,
            )
        ).mlir_text()
        self.assertIn("!pto.vreg<256xf8E4M3FN>", text)
        self.assertIn("!pto.mask<b8>", text)

    def test_low_precision_vcvt_kind_classification_and_lowering(self) -> None:
        expected_kinds = {
            pto.f8e4m3: "f8e4m3",
            pto.f8e5m2: "f8e5m2",
            pto.hif8: "h8",
            pto.f4e1m2x2: "f4e1m2x2",
            pto.f4e2m1x2: "f4e2m1x2",
        }
        for dtype, kind in expected_kinds.items():
            with self.subTest(dtype=dtype.name):
                self.assertEqual(_classify_vcvt_elem_kind(dtype), kind)

        @pto.vkernel(
            op="low_precision_vcvt_f32_to_f8e4m3_unique",
            dtypes=[(pto.f8e4m3, pto.f32)],
            advanced=True,
        )
        def kernel(dst: pto.Tile, src: pto.Tile):
            src_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            dst_mask = pto.make_mask(pto.f8e4m3, pto.PAT.ALL)
            vec = pto.vlds(src, 0)
            out = pto.vcvt(vec, pto.f8e4m3, src_mask)
            pto.vsts(out, dst, 0, dst_mask)
            return None

        text = kernel.specialize(
            dst=pto.TileSpecialization(
                shape=_tile_shape(pto.f8e4m3),
                memory_space=pto.MemorySpace.UB,
            ),
            src=pto.TileSpecialization(
                shape=(8, pto.get_lanes(pto.f32)),
                memory_space=pto.MemorySpace.UB,
            ),
        ).mlir_text()
        self.assertIn("pto.vcvt", text)
        self.assertIn("!pto.vreg<256xf8E4M3FN>", text)

    def test_vector_arithmetic_ops_continue_to_reject_low_precision_dtypes(self) -> None:
        def build_vadd_kernel(dtype: pto.ScalarType):
            @pto.vkernel(
                op=f"low_precision_vadd_{dtype.name}_unique",
                dtypes=[(dtype, dtype)],
                advanced=True,
            )
            def kernel(dst: pto.Tile, src: pto.Tile):
                dtype = dst.element_type
                mask = pto.make_mask(dtype, pto.PAT.ALL)
                lhs = pto.vlds(dst, 0)
                rhs = pto.vlds(src, 0)
                out = pto.vadd(lhs, rhs, mask)
                pto.vsts(out, dst, 0, mask)
                return None

            return kernel

        def build_vexp_kernel(dtype: pto.ScalarType):
            @pto.vkernel(
                op=f"low_precision_vexp_{dtype.name}_unique",
                dtypes=[(dtype,)],
                advanced=True,
            )
            def kernel(dst: pto.Tile):
                dtype = dst.element_type
                mask = pto.make_mask(dtype, pto.PAT.ALL)
                vec = pto.vlds(dst, 0)
                out = pto.vexp(vec, mask)
                pto.vsts(out, dst, 0, mask)
                return None

            return kernel

        def build_vmov_kernel(dtype: pto.ScalarType):
            @pto.vkernel(
                op=f"low_precision_vmov_{dtype.name}_unique",
                dtypes=[(dtype,)],
                advanced=True,
            )
            def kernel(dst: pto.Tile):
                dtype = dst.element_type
                mask = pto.make_mask(dtype, pto.PAT.ALL)
                vec = pto.vlds(dst, 0)
                out = pto.vmov(vec, mask)
                pto.vsts(out, dst, 0, mask)
                return None

            return kernel

        for dtype, _ in LOW_PRECISION_DTYPES:
            tile = pto.TileSpecialization(
                shape=_tile_shape(dtype),
                memory_space=pto.MemorySpace.UB,
            )

            for builder, op_name in (
                (build_vadd_kernel, "vadd"),
                (build_vexp_kernel, "vexp"),
                (build_vmov_kernel, "vmov"),
            ):
                with self.subTest(dtype=dtype.name, op=op_name):
                    kernel = builder(dtype)
                    with self.assertRaises(TypeError):
                        if op_name == "vadd":
                            kernel.specialize(dst=tile, src=tile).mlir_text()
                        else:
                            kernel.specialize(dst=tile).mlir_text()
