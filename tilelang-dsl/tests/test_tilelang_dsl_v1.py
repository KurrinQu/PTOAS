import tempfile
import unittest
from unittest import mock
from importlib import util
from pathlib import Path

import tilelang_dsl as pto
import tilelang_dsl.kernel as kernel_impl
from tilelang_dsl.support_matrix import (
    ADVANCED_EXPLICIT_VECSCOPE_SURFACES,
    ADVANCED_LOW_LEVEL_DMA_SURFACES,
    ADVANCED_RAW_POINTER_SURFACES,
    ADVANCED_TILE_HELPER_SURFACES,
    ADVANCED_TIER,
    AUTHORING_TIER_SURFACE_GROUPS,
    BASIC_TIER,
    BASIC_TILE_INDEXING_SURFACES,
    get_feature_tier,
    get_surface_group_tier,
)
from tilelang_dsl.frontend_ast import (
    FrontendAssignStmt,
    FrontendCallExpr,
    FrontendExprStmt,
    FrontendForStmt,
    FrontendStrictVecscopeStmt,
    build_frontend_kernel_node,
)
from tilelang_dsl.lowering import AuthoringModule, lower_semantic_kernel
from tilelang_dsl.semantic import (
    SemanticAssignStmt,
    SemanticBinaryExpr,
    SemanticCallExpr,
    SemanticDmaConfigStmt,
    SemanticForStmt,
    SemanticIfStmt,
    SemanticIndexType,
    SemanticLowLevelCopyStmt,
    SemanticMaskType,
    SemanticPipeBarrierStmt,
    SemanticPtrType,
    SemanticScalarType,
    SemanticSetFlagStmt,
    SemanticStrictVecscopeStmt,
    SemanticSymbolExpr,
    SemanticTensorViewType,
    SemanticTileType,
    SemanticVecscopeStmt,
    SemanticVectorStoreStmt,
    SemanticWaitFlagStmt,
    analyze_frontend_kernel,
)


class TileLangDSLPackageTests(unittest.TestCase):
    def test_package_exports_surface(self) -> None:
        self.assertIsNotNone(pto.__file__)
        self.assertTrue(hasattr(pto, "vkernel"))
        self.assertTrue(hasattr(pto, "KernelRegistry"))
        self.assertTrue(hasattr(pto, "select_kernel"))
        self.assertTrue(hasattr(pto, "TensorView"))
        self.assertTrue(hasattr(pto, "Tile"))
        self.assertTrue(hasattr(pto, "TileSpecialization"))
        self.assertTrue(hasattr(pto, "PointerType"))
        self.assertTrue(hasattr(pto, "ptr"))
        self.assertTrue(hasattr(pto, "bytewidth"))
        self.assertTrue(hasattr(pto, "get_lanes"))
        self.assertTrue(hasattr(pto, "PAT"))
        self.assertTrue(hasattr(pto, "PadMode"))
        self.assertTrue(hasattr(pto, "PIPE"))
        self.assertTrue(hasattr(pto, "EVENT"))
        self.assertEqual(pto.PadMode.PadNull.value, "PadNull")
        self.assertEqual(pto.PadMode.PadFirstElem.value, "PadFirstElem")
        self.assertEqual(pto.PadMode.PadValue.value, "PadValue")


class TileLangDSLSupportMatrixTests(unittest.TestCase):
    def test_stable_starter_surface_groups_map_to_stable_tier(self) -> None:
        self.assertEqual(get_surface_group_tier("TensorView"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("Tile"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("base_vector_ops"), BASIC_TIER)
        self.assertEqual(get_surface_group_tier("tile_indexing_sugar"), BASIC_TIER)

        self.assertIn("TensorView", AUTHORING_TIER_SURFACE_GROUPS["TensorView"])
        self.assertIn("Tile", AUTHORING_TIER_SURFACE_GROUPS["Tile"])
        self.assertNotIn("dma_load/store", AUTHORING_TIER_SURFACE_GROUPS)
        self.assertIn("pto.vlds", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vsts", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vadd", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("pto.vmuls", AUTHORING_TIER_SURFACE_GROUPS["base_vector_ops"])
        self.assertIn("tile[start:]", BASIC_TILE_INDEXING_SURFACES)
        self.assertIn("tile[row, col:]", BASIC_TILE_INDEXING_SURFACES)

        self.assertEqual(get_feature_tier("TensorView"), BASIC_TIER)
        self.assertEqual(get_feature_tier("Tile"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vlds"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vsts"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vadd"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.vmuls"), BASIC_TIER)
        self.assertEqual(get_feature_tier("PadMode"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.bytewidth"), BASIC_TIER)
        self.assertEqual(get_feature_tier("pto.get_lanes"), BASIC_TIER)
        self.assertEqual(get_feature_tier("tile[start:]"), BASIC_TIER)
        self.assertEqual(get_feature_tier("tile[row, col:]"), BASIC_TIER)

    def test_non_stable_surface_groups_keep_advanced_boundaries(self) -> None:
        self.assertEqual(get_surface_group_tier("strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("raw_pointer_family"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("low_level_dma_family"), ADVANCED_TIER)
        self.assertEqual(get_surface_group_tier("tile_helper_family"), ADVANCED_TIER)

        self.assertIn("pto.strict_vecscope", ADVANCED_EXPLICIT_VECSCOPE_SURFACES)
        self.assertIn("pto.ptr", ADVANCED_RAW_POINTER_SURFACES)
        self.assertIn("pto.castptr", ADVANCED_RAW_POINTER_SURFACES)
        self.assertIn("pto.copy_ubuf_to_ubuf", ADVANCED_LOW_LEVEL_DMA_SURFACES)
        self.assertIn("pto.tile_with_strides", ADVANCED_TILE_HELPER_SURFACES)

        self.assertEqual(get_feature_tier("strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.strict_vecscope"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.ptr"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.castptr"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.copy_ubuf_to_ubuf"), ADVANCED_TIER)
        self.assertEqual(get_feature_tier("pto.tile_with_strides"), ADVANCED_TIER)

    def test_unsupported_features_do_not_report_legacy_tiers(self) -> None:
        with self.assertRaises(KeyError):
            get_surface_group_tier("dma_load/store")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_load")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_store")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.dma_copy")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.vreduce")
        with self.assertRaises(KeyError):
            get_feature_tier("pto.mask_b32")


class TileLangDSLMatcherEntryTests(unittest.TestCase):
    def test_select_kernel_returns_descriptor_from_default_registry(self) -> None:
        @pto.vkernel(op="matcher_entry_default_registry_unique", dtypes=[(pto.f32, pto.i32)])
        def kernel(inp: pto.TensorView, scale: pto.i32):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_entry_default_registry_unique",
            (pto.f32, pto.i32),
        )

        self.assertIs(selected, kernel)

    def test_select_kernel_uses_explicit_registry_without_falling_back(self) -> None:
        @pto.vkernel(op="matcher_entry_registry_isolation_unique", dtypes=[(pto.f32,)])
        def default_kernel(inp: pto.TensorView):
            return None

        empty_registry = pto.KernelRegistry()
        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_entry_registry_isolation_unique",
                (pto.f32,),
                registry=empty_registry,
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

        isolated_registry = pto.KernelRegistry()
        isolated_registry.register(default_kernel)
        selected = pto.select_kernel(
            "a5",
            "matcher_entry_registry_isolation_unique",
            (pto.f32,),
            registry=isolated_registry,
        )

        self.assertIs(selected, default_kernel)
        self.assertEqual(len(isolated_registry.descriptors), 1)

    def test_select_kernel_binds_concrete_signature_from_multi_signature_descriptor(self) -> None:
        @pto.vkernel(
            op="matcher_multi_signature_unique",
            dtypes=[
                (pto.f16, pto.f16),
                (pto.f32, pto.f32),
            ],
        )
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_multi_signature_unique",
            (pto.f32, pto.f32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f32)],
        )
        specialized = selected.specialize(
            tile=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB)
        )
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16, v_row=8, v_col=16",
            specialized.mlir_text(),
        )

    def test_select_kernel_binds_omitted_dtypes_via_anytype_defaults(self) -> None:
        @pto.vkernel(op="matcher_default_dtypes_unique")
        def kernel(inp: pto.Tile, out: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_default_dtypes_unique",
            (pto.f16, pto.f16),
        )

        self.assertEqual(selected.dtype_signature, (pto.f16, pto.f16))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in selected.parameters],
            [("inp", "tile", pto.f16), ("out", "tile", pto.f16)],
        )
        specialized = selected.specialize(
            inp=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
            out=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        self.assertIn(
            "!pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=16, v_row=8, v_col=16",
            specialized.mlir_text(),
        )

    def test_select_kernel_default_dtypes_preserve_scalar_annotations(self) -> None:
        @pto.vkernel(op="matcher_default_dtypes_scalar_guard_unique")
        def kernel(inp: pto.TensorView, scale: pto.i32):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_default_dtypes_scalar_guard_unique",
            (pto.f32, pto.i32),
        )
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.i32))

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_default_dtypes_scalar_guard_unique",
                (pto.f32, pto.f16),
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

    def test_select_kernel_matches_wildcards_deterministically(self) -> None:
        @pto.vkernel(
            op="matcher_wildcard_unique",
            dtypes=[
                (pto.AnyInt, pto.AnyType),
                (pto.AnyFloat, pto.AnyType),
            ],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_wildcard_unique",
            (pto.f32, pto.i32),
        )

        self.assertEqual(selected.dtype_signature, (pto.f32, pto.i32))
        self.assertEqual(selected.parameters[0].dtype, pto.f32)
        self.assertEqual(selected.parameters[1].dtype, pto.i32)

    def test_select_kernel_enforces_typevar_consistency_per_signature(self) -> None:
        @pto.vkernel(
            op="matcher_typevar_unique",
            dtypes=[(pto.TypeVar("T"), pto.TypeVar("T"))],
        )
        def kernel(lhs: pto.TensorView, rhs: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_typevar_unique",
            (pto.f32, pto.f32),
        )
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_typevar_unique",
                (pto.f32, pto.i32),
            )
        self.assertIn("found no registered kernel", str(ctx.exception))

    def test_polymorphic_descriptor_requires_select_kernel_before_materialization(self) -> None:
        @pto.vkernel(
            op="matcher_materialization_gate_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(ValueError) as ctx:
            kernel.mlir_text()
        self.assertIn("requires pto.select_kernel(...)", str(ctx.exception))

    def test_select_kernel_evaluates_constraints_before_priority(self) -> None:
        def requires_large_batch(batch=0):
            return batch >= 1024

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[requires_large_batch],
            priority=100,
        )
        def high_priority_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_constraint_priority_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[],
            priority=10,
        )
        def fallback_kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 128},
        )
        self.assertIs(selected.py_fn, fallback_kernel.py_fn)
        self.assertEqual(selected.priority, 10)

        selected = pto.select_kernel(
            "a5",
            "matcher_constraint_priority_unique",
            (pto.f32, pto.f32),
            context_attrs={"batch": 4096},
        )
        self.assertIs(selected.py_fn, high_priority_kernel.py_fn)
        self.assertEqual(selected.priority, 100)

    def test_select_kernel_raises_tie_error_for_equal_highest_priority(self) -> None:
        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def lhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            op="matcher_priority_tie_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            priority=50,
        )
        def rhs(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_priority_tie_unique",
                (pto.f32, pto.f32),
            )
        self.assertIn("multiple highest-priority kernels", str(ctx.exception))
        self.assertIn("lhs(priority=50", str(ctx.exception))
        self.assertIn("rhs(priority=50", str(ctx.exception))

    def test_select_kernel_reports_no_candidate_after_constraint_evaluation(self) -> None:
        @pto.vkernel(
            op="matcher_constraint_empty_unique",
            dtypes=[(pto.AnyFloat, pto.AnyFloat)],
            constraints=[lambda enabled=False: enabled],
            priority=1,
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_constraint_empty_unique",
                (pto.f32, pto.f32),
                context_attrs={"enabled": False},
            )
        self.assertIn("after constraint evaluation", str(ctx.exception))

    def test_materialization_constraints_can_see_specializations_and_selected_context_attrs(self) -> None:
        @pto.vkernel(
            op="matcher_materialization_constraint_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda src: src.rank == 5,
                lambda dst, expected_rows=None: dst.shape[0] == expected_rows,
                lambda src, dst: dst.valid_shape[1] <= src.shape[4],
            ],
        )
        def kernel(src: pto.TensorView, dst: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_materialization_constraint_unique",
            (pto.f32, pto.f32),
            context_attrs={"expected_rows": 8, "src_shape": (2, 2, 1, 1, 16), "src_strides": (32, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB, valid_shape=(4, 16)),
        )
        text = selected.mlir_text()
        self.assertIn("!pto.tensor_view<?x?x?x?x?xf32>", text)
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16", text)

        rejected = pto.select_kernel(
            "a5",
            "matcher_materialization_constraint_unique",
            (pto.f32, pto.f32),
            context_attrs={"expected_rows": 8, "src_shape": (2, 2, 1, 1, 8), "src_strides": (16, 8, 8, 8, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB, valid_shape=(4, 16)),
        )
        with self.assertRaises(LookupError) as ctx:
            rejected.mlir_text()
        self.assertIn("constraint evaluation rejected", str(ctx.exception))

    def test_constraints_support_parameter_style_shape_and_stride_access(self) -> None:
        @pto.vkernel(
            op="matcher_parameter_style_constraints_unique",
            dtypes=[(pto.f32, pto.f32)],
            constraints=[
                lambda src, dst: src.rank == 5,
                lambda src: src.strides[4] == 1,
                lambda src, dst: src.shape[0] <= dst.shape[0],
            ],
        )
        def kernel(src: pto.TensorView, dst: pto.Tile):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_parameter_style_constraints_unique",
            (pto.f32, pto.f32),
            context_attrs={"src_shape": (4, 1, 1, 1, 16), "src_strides": (16, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=16", selected.mlir_text())

        rejected = pto.select_kernel(
            "a5",
            "matcher_parameter_style_constraints_unique",
            (pto.f32, pto.f32),
            context_attrs={"src_shape": (16, 1, 1, 1, 16), "src_strides": (16, 16, 16, 16, 1)},
        ).specialize(
            dst=pto.TileSpecialization(shape=(8, 16), memory_space=pto.MemorySpace.UB),
        )
        with self.assertRaises(LookupError):
            rejected.mlir_text()

    def test_select_kernel_binds_selected_op_for_multi_op_descriptor(self) -> None:
        @pto.vkernel(
            ops=["matcher_multi_op_bind_add_unique", "matcher_multi_op_bind_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_multi_op_bind_sub_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, kernel.py_fn)
        self.assertEqual(selected.match_ops, ("matcher_multi_op_bind_add_unique", "matcher_multi_op_bind_sub_unique"))
        self.assertEqual(selected.selected_op, "matcher_multi_op_bind_sub_unique")
        self.assertEqual(selected.op, "matcher_multi_op_bind_sub_unique")
        self.assertEqual(selected.dtype_signature, (pto.f32, pto.f32))

    def test_select_kernel_hits_same_multi_op_descriptor_for_multiple_query_ops(self) -> None:
        @pto.vkernel(
            ops=[
                "matcher_multi_hit_add_unique",
                "matcher_multi_hit_mul_unique",
                "matcher_multi_hit_div_unique",
            ],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        add_selected = pto.select_kernel(
            "a5",
            "matcher_multi_hit_add_unique",
            (pto.f32, pto.f32),
        )
        mul_selected = pto.select_kernel(
            "a5",
            "matcher_multi_hit_mul_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(add_selected.py_fn, kernel.py_fn)
        self.assertIs(mul_selected.py_fn, kernel.py_fn)
        self.assertEqual(add_selected.match_ops, kernel.match_ops)
        self.assertEqual(mul_selected.match_ops, kernel.match_ops)
        self.assertEqual(add_selected.selected_op, "matcher_multi_hit_add_unique")
        self.assertEqual(mul_selected.selected_op, "matcher_multi_hit_mul_unique")
        self.assertEqual(add_selected.op, "matcher_multi_hit_add_unique")
        self.assertEqual(mul_selected.op, "matcher_multi_hit_mul_unique")

    def test_select_kernel_prefers_higher_priority_single_op_over_multi_op(self) -> None:
        @pto.vkernel(
            op="matcher_single_beats_multi_priority_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=12,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=[
                "matcher_single_beats_multi_priority_unique",
                "matcher_single_beats_multi_priority_alt_unique",
            ],
            dtypes=[(pto.f32, pto.f32)],
            priority=4,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_single_beats_multi_priority_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, single.py_fn)
        self.assertEqual(selected.selected_op, "matcher_single_beats_multi_priority_unique")
        self.assertEqual(selected.priority, 12)

    def test_select_kernel_prefers_priority_over_single_op_specificity(self) -> None:
        @pto.vkernel(
            op="matcher_single_vs_multi_priority_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=5,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=["matcher_single_vs_multi_priority_unique", "matcher_single_vs_multi_priority_alt_unique"],
            dtypes=[(pto.f32, pto.f32)],
            priority=9,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "matcher_single_vs_multi_priority_unique",
            (pto.f32, pto.f32),
        )

        self.assertIs(selected.py_fn, multi.py_fn)
        self.assertEqual(selected.selected_op, "matcher_single_vs_multi_priority_unique")
        self.assertEqual(selected.priority, 9)

    def test_select_kernel_raises_tie_error_when_single_and_multi_op_candidates_tie(self) -> None:
        @pto.vkernel(
            op="matcher_single_multi_tie_unique",
            dtypes=[(pto.f32, pto.f32)],
            priority=17,
        )
        def single(inp: pto.TensorView, out: pto.TensorView):
            return None

        @pto.vkernel(
            ops=["matcher_single_multi_tie_unique", "matcher_single_multi_tie_alt_unique"],
            dtypes=[(pto.f32, pto.f32)],
            priority=17,
        )
        def multi(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(LookupError) as ctx:
            pto.select_kernel(
                "a5",
                "matcher_single_multi_tie_unique",
                (pto.f32, pto.f32),
            )

        self.assertIn("multiple highest-priority kernels", str(ctx.exception))
        self.assertIn("single(priority=17", str(ctx.exception))
        self.assertIn("multi(priority=17", str(ctx.exception))


class TileLangDSLDescriptorTests(unittest.TestCase):
    def test_descriptor_metadata_and_parameter_binding(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], verify=False)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        self.assertEqual(kernel.target, "a5")
        self.assertEqual(kernel.op, "eltwise")
        self.assertEqual(kernel.name, "kernel")
        self.assertFalse(kernel.verify_enabled)
        self.assertFalse(kernel.advanced_enabled)
        self.assertEqual(kernel.metadata["verify"], False)
        self.assertEqual(kernel.metadata["advanced"], False)
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f16, pto.i32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("tile", "tile", pto.f16), ("scale", "scalar", pto.i32)],
        )
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)
        self.assertEqual(kernel.parameters[1].element_dtype, pto.f16)
        self.assertIsNone(kernel.parameters[2].element_dtype)

    def test_descriptor_accepts_multi_op_matcher_metadata(self) -> None:
        @pto.vkernel(ops=["tadd", "tsub"], dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        self.assertEqual(kernel.match_ops, ("tadd", "tsub"))
        self.assertIsNone(kernel.selected_op)
        self.assertIsNone(kernel.metadata["op"])
        self.assertEqual(kernel.metadata["match_ops"], ("tadd", "tsub"))
        self.assertIsNone(kernel.metadata["selected_op"])
        self.assertEqual(kernel.dtype_signature, (pto.f32, pto.f32))
        self.assertEqual(
            [(param.name, param.kind, param.dtype) for param in kernel.parameters],
            [("inp", "tensorview", pto.f32), ("out", "tensorview", pto.f32)],
        )
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.op
        self.assertIn("bind a concrete op", str(ctx.exception))

    def test_descriptor_defaults_dtypes_for_beginner_tile_kernels(self) -> None:
        @pto.vkernel(op="default_dtypes_unique")
        def kernel(inp: pto.Tile, out: pto.Tile):
            return None

        self.assertEqual(kernel.match_ops, ("default_dtypes_unique",))
        self.assertEqual(kernel.dtypes, ((pto.AnyType, pto.AnyType),))
        self.assertEqual(kernel.metadata["dtypes"], ((pto.AnyType, pto.AnyType),))
        with self.assertRaises(ValueError) as ctx:
            _ = kernel.dtype_signature
        self.assertIn("choose a concrete dtype signature", str(ctx.exception))

    def test_descriptor_accepts_templates_metadata(self) -> None:
        @pto.vkernel(
            ops=["tadd", "tsub", "tmul"],
            dtypes=[(pto.f32, pto.f32)],
            templates={
                "core": {
                    "tadd": "vadd",
                    "tsub": "vsub",
                },
                "post": {
                    "tmul": "vrelu",
                },
            },
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        self.assertEqual(
            kernel.templates,
            {
                "core": {
                    "tadd": "vadd",
                    "tsub": "vsub",
                },
                "post": {
                    "tmul": "vrelu",
                },
            },
        )
        self.assertEqual(kernel.metadata["templates"], kernel.templates)

    def test_descriptor_rejects_op_and_ops_together(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(op="tadd", ops=["tsub"], dtypes=[(pto.f32,)])
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("either op= or ops=", str(ctx.exception))

    def test_descriptor_requires_one_of_op_or_ops(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(dtypes=[(pto.f32,)])
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("exactly one of op= or ops=", str(ctx.exception))

    def test_descriptor_rejects_template_slot_with_non_string_name(self) -> None:
        with self.assertRaises(TypeError) as ctx:
            @pto.vkernel(
                ops=["tadd"],
                dtypes=[(pto.f32,)],
                templates={1: {"tadd": "vadd"}},
            )
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("template slot names must be non-empty strings", str(ctx.exception))

    def test_descriptor_rejects_template_op_outside_matcher_set(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(
                ops=["tadd", "tsub"],
                dtypes=[(pto.f32, pto.f32)],
                templates={"core": {"tmul": "vmul"}},
            )
            def kernel(inp: pto.TensorView, out: pto.TensorView):
                return None

        self.assertIn("outside descriptor matcher set", str(ctx.exception))

    def test_descriptor_rejects_template_mapping_to_unknown_pto_op(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            @pto.vkernel(
                ops=["tadd"],
                dtypes=[(pto.f32,)],
                templates={"core": {"tadd": "vunknown"}},
            )
            def kernel(inp: pto.TensorView):
                return None

        self.assertIn("maps to unsupported pto op", str(ctx.exception))

    def test_pointer_parameter_annotation_binds_as_ptr_kind(self) -> None:
        @pto.vkernel(op="ptr_surface", dtypes=[(pto.f32, pto.i64)], advanced=True)
        def kernel(src: pto.ptr(pto.f32, pto.MemorySpace.UB), addr: pto.i64):
            return None

        self.assertEqual(kernel.parameters[0].kind, "ptr")
        self.assertEqual(kernel.parameters[0].dtype, pto.f32)
        self.assertEqual(kernel.parameters[0].annotation, pto.ptr(pto.f32, pto.MemorySpace.UB))
        self.assertEqual(kernel.parameters[0].element_dtype, pto.f32)

    def test_specialization_enables_materialization_apis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
                config=pto.TileConfig.from_mapping({"layout": "row_major"}),
            )
        )

        self.assertIn("tile", specialized.specializations_by_name)
        text = specialized.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.specialize tile shape=(16, 32) memory_space=ub", text)
        self.assertIn('module attributes {pto.target_arch = "a5"} {', text)
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance } {",
            text,
        )
        module = specialized.mlir_module()
        self.assertEqual(type(module).__name__, "MaterializedMLIRModule")
        mocked_result = kernel_impl.VerificationResult(
            status="passed",
            available=True,
            passed=True,
            message="ok",
            command=("ptoas",),
            returncode=0,
        )
        with mock.patch("tilelang_dsl.kernel._run_ptoas_verifier", return_value=mocked_result):
            self.assertTrue(module.verify())
            self.assertTrue(specialized.verify())
            self.assertEqual(module.verify().status, "passed")
            self.assertEqual(specialized.verify().status, "passed")

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            specialized.emit(out)
            self.assertEqual(out.read_text(encoding="utf-8"), text)

    def test_multi_op_descriptor_requires_select_kernel_before_materialization_apis(self) -> None:
        @pto.vkernel(
            ops=["multi_op_gate_add_unique", "multi_op_gate_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        with self.assertRaises(ValueError) as text_ctx:
            kernel.mlir_text()
        self.assertIn("mlir_text() requires pto.select_kernel(...) to bind a concrete op", str(text_ctx.exception))

        with self.assertRaises(ValueError) as module_ctx:
            kernel.mlir_module()
        self.assertIn(
            "mlir_module() requires pto.select_kernel(...) to bind a concrete op",
            str(module_ctx.exception),
        )

        with self.assertRaises(ValueError) as verify_ctx:
            kernel.verify()
        self.assertIn("verify() requires pto.select_kernel(...) to bind a concrete op", str(verify_ctx.exception))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "kernel.mlir"
            with self.assertRaises(ValueError) as emit_ctx:
                kernel.emit(out)
        self.assertIn("emit() requires pto.select_kernel(...) to bind a concrete op", str(emit_ctx.exception))

    def test_selected_multi_op_descriptor_can_materialize_normally(self) -> None:
        @pto.vkernel(
            ops=["multi_op_materialize_add_unique", "multi_op_materialize_sub_unique"],
            dtypes=[(pto.f32, pto.f32)],
        )
        def kernel(inp: pto.TensorView, out: pto.TensorView):
            return None

        selected = pto.select_kernel(
            "a5",
            "multi_op_materialize_sub_unique",
            (pto.f32, pto.f32),
        )

        text = selected.mlir_text()
        self.assertIn("// tilelang.target = a5", text)
        self.assertIn("// tilelang.op = multi_op_materialize_sub_unique", text)
        self.assertIn(
            'func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tensor_view<?x?x?x?x?xf32>) attributes { pto.tilelang.instance } {',
            text,
        )

    def test_verify_reports_structured_unavailable_when_ptoas_is_missing(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 32),
                memory_space=pto.MemorySpace.UB,
            )
        )

        result = specialized.verify(ptoas_bin="/definitely-missing/ptoas")
        self.assertFalse(result)
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.available)
        self.assertFalse(result.passed)
        self.assertIn("verifier unavailable", result.message)

    def test_descriptor_materialization_flows_through_pipeline(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(frontend_kernel.name, "kernel")
        self.assertEqual(
            [(param.name, param.kind) for param in frontend_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile"), ("scale", "scalar")],
        )
        self.assertEqual(frontend_kernel.tile_specializations[0].shape, (8, 16))

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertEqual(semantic_kernel.symbol_name, "kernel")
        self.assertEqual(semantic_kernel.tile_bindings[0].memory_space, "ub")

        authoring_module = lower_semantic_kernel(semantic_kernel)
        self.assertIsInstance(authoring_module, AuthoringModule)
        self.assertEqual(authoring_module.render(), specialized.mlir_text())
        self.assertIn("return", authoring_module.render())

    def test_frontend_rejects_hidden_dma_load_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_load_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(inp: pto.TensorView, tile: pto.Tile):
                pto.dma_load(inp[0:16, 0:16], tile)
                return None

        self.assertIn("unsupported op surface `pto.dma_load`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_hidden_dma_store_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_store_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(out: pto.TensorView, tile: pto.Tile):
                pto.dma_store(tile, out[0:16, 0:16])
                return None

        self.assertIn("unsupported op surface `pto.dma_store`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_hidden_dma_copy_surface(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_copy_hidden", dtypes=[(pto.f32, pto.f32)])
            def kernel(src: pto.Tile, dst: pto.Tile):
                pto.dma_copy(src, dst)
                return None

        self.assertIn("unsupported op surface `pto.dma_copy`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rejects_keyword_arguments_on_public_surfaces(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="dma_kw_wrong_surface", dtypes=[(pto.f32, pto.f32)])
            def kernel(inp: pto.TensorView, tile: pto.Tile):
                pto.vlds(tile, offset=0)
                return None

        self.assertIn("no public call surface currently accepts them", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_frontend_rewrites_template_slot_to_selected_real_op(self) -> None:
        @pto.vkernel(
            ops=["template_slot_add_unique", "template_slot_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_add_unique": "vadd",
                    "template_slot_sub_unique": "vsub",
                }
            },
        )
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                out_tile,
                lhs_tile,
                rhs_tile,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    lhs = pto.vlds(lhs_tile, lane)
                    rhs = pto.vlds(rhs_tile, lane)
                    out = pto.tpl("core", lhs, rhs, mask)
                    pto.vsts(out, out_tile, lane, mask)
            return None

        add_selected = pto.select_kernel(
            "a5",
            "template_slot_add_unique",
            (pto.f32, pto.f32, pto.f32),
        ).specialize(
            dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
        )
        sub_selected = pto.select_kernel(
            "a5",
            "template_slot_sub_unique",
            (pto.f32, pto.f32, pto.f32),
        ).specialize(
            dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
        )

        add_frontend = build_frontend_kernel_node(add_selected)
        sub_frontend = build_frontend_kernel_node(sub_selected)

        add_vecscope = add_frontend.body[0]
        sub_vecscope = sub_frontend.body[0]
        self.assertIsInstance(add_vecscope, FrontendStrictVecscopeStmt)
        self.assertIsInstance(sub_vecscope, FrontendStrictVecscopeStmt)

        add_loop = add_vecscope.body[0]
        sub_loop = sub_vecscope.body[0]
        self.assertIsInstance(add_loop, FrontendForStmt)
        self.assertIsInstance(sub_loop, FrontendForStmt)

        add_out_assign = add_loop.body[3]
        sub_out_assign = sub_loop.body[3]
        self.assertIsInstance(add_out_assign, FrontendAssignStmt)
        self.assertIsInstance(sub_out_assign, FrontendAssignStmt)
        self.assertIsInstance(add_out_assign.value, FrontendCallExpr)
        self.assertIsInstance(sub_out_assign.value, FrontendCallExpr)
        self.assertEqual(add_out_assign.value.namespace, "pto")
        self.assertEqual(sub_out_assign.value.namespace, "pto")
        self.assertEqual(add_out_assign.value.name, "vadd")
        self.assertEqual(sub_out_assign.value.name, "vsub")

        add_text = add_selected.mlir_text()
        sub_text = sub_selected.mlir_text()
        self.assertIn("pto.vadd", add_text)
        self.assertNotIn("pto.vsub", add_text)
        self.assertIn("pto.vsub", sub_text)
        self.assertNotIn("pto.vadd", sub_text)

    def test_template_slot_shared_kernel_body_expands_for_four_ops(self) -> None:
        @pto.vkernel(
            ops=[
                "template_slot_tadd_unique",
                "template_slot_tsub_unique",
                "template_slot_tmul_unique",
                "template_slot_tdiv_unique",
            ],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_tadd_unique": "vadd",
                    "template_slot_tsub_unique": "vsub",
                    "template_slot_tmul_unique": "vmul",
                    "template_slot_tdiv_unique": "vdiv",
                }
            },
        )
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                out_tile,
                lhs_tile,
                rhs_tile,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    lhs = pto.vlds(lhs_tile, lane)
                    rhs = pto.vlds(rhs_tile, lane)
                    out = pto.tpl("core", lhs, rhs, mask)
                    pto.vsts(out, out_tile, lane, mask)
            return None

        isolated_registry = pto.KernelRegistry((kernel,))
        expected_ops = {
            "template_slot_tadd_unique": "vadd",
            "template_slot_tsub_unique": "vsub",
            "template_slot_tmul_unique": "vmul",
            "template_slot_tdiv_unique": "vdiv",
        }

        for query_op, real_op in expected_ops.items():
            selected = pto.select_kernel(
                "a5",
                query_op,
                (pto.f32, pto.f32, pto.f32),
                registry=isolated_registry,
            ).specialize(
                dst=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
                src0=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
                src1=pto.TileSpecialization(shape=(16, 16), memory_space=pto.MemorySpace.UB),
            )

            frontend_kernel = build_frontend_kernel_node(selected)
            vecscope = frontend_kernel.body[0]
            self.assertIsInstance(vecscope, FrontendStrictVecscopeStmt)
            loop_stmt = vecscope.body[0]
            self.assertIsInstance(loop_stmt, FrontendForStmt)
            out_assign = loop_stmt.body[3]
            self.assertIsInstance(out_assign, FrontendAssignStmt)
            self.assertIsInstance(out_assign.value, FrontendCallExpr)
            self.assertEqual(out_assign.value.name, real_op)

            text = selected.mlir_text()
            self.assertIn(f"pto.{real_op}", text)
            self.assertNotIn("pto.tpl(", text)

    def test_template_slot_rejects_non_literal_slot_name(self) -> None:
        slot_name = "core"

        @pto.vkernel(
            op="template_slot_non_literal_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_non_literal_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl(slot_name, lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("pto.tpl() requires a non-empty string literal slot name", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_rejects_unknown_slot_before_ir_generation(self) -> None:
        @pto.vkernel(
            op="template_slot_unknown_slot_unique",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_unknown_slot_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("missing", lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("unknown template slot 'missing'", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_rejects_missing_selected_op_mapping(self) -> None:
        @pto.vkernel(
            ops=["template_slot_missing_map_add_unique", "template_slot_missing_map_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={"core": {"template_slot_missing_map_add_unique": "vadd"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("core", lhs_tile, rhs_tile, mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "template_slot_missing_map_sub_unique",
            (pto.f32, pto.f32, pto.f32),
        )

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(selected)

        self.assertIn("template slot 'core' does not define an implementation for selected op", str(ctx.exception))
        self.assertIn("template_slot_missing_map_sub_unique", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_requires_selected_op_before_expansion(self) -> None:
        @pto.vkernel(
            ops=["template_slot_unbound_add_unique", "template_slot_unbound_sub_unique"],
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
            templates={
                "core": {
                    "template_slot_unbound_add_unique": "vadd",
                    "template_slot_unbound_sub_unique": "vsub",
                }
            },
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (out_tile, lhs_tile, rhs_tile, lb, ub, step):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    out = pto.tpl("core", lhs_tile, rhs_tile, mask)
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("pto.tpl() requires pto.select_kernel(...) to bind a concrete op before expansion", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_template_slot_respects_resolved_op_surface_rules(self) -> None:
        @pto.vkernel(
            op="template_slot_advanced_surface_unique",
            dtypes=[(pto.i32, pto.i32, pto.i32)],
            templates={"cmp": {"template_slot_advanced_surface_unique": "vcmp"}},
        )
        def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
            mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            out = pto.tpl("cmp", dst, src0, mask, "lt")
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            build_frontend_kernel_node(kernel)

        self.assertIn("surface `pto.vcmp` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_callable_based_runtime_template_dispatch_remains_rejected(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(
                op="template_slot_callable_dispatch_unique",
                dtypes=[(pto.f32, pto.f32, pto.f32)],
                advanced=True,
            )
            def kernel(dst: pto.TensorView, src0: pto.TensorView, src1: pto.TensorView):
                table = {"core": pto.vadd}
                with pto.strict_vecscope(dst, src0, src1, 0, 64, 64) as (
                    out_tile,
                    lhs_tile,
                    rhs_tile,
                    lb,
                    ub,
                    step,
                ):
                    for lane in range(lb, ub, step):
                        mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                        out = table["core"](lhs_tile, rhs_tile, mask)
                return None

        self.assertIn("unsupported call surface in TileLang DSL v1", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_semantic_pipeline_binds_parameter_loop_and_strict_vecscope_types(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            rows = tile.shape[0]
            step = rows
            with pto.strict_vecscope(inp, tile, scale, 0, rows, step) as (
                vin,
                vtmp,
                factor,
                lb,
                ub,
                vec_step,
            ):
                for lane in range(lb, ub, vec_step):
                    current = factor
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        frontend_kernel = build_frontend_kernel_node(specialized)
        self.assertEqual(len(frontend_kernel.body), 4)

        semantic_kernel = analyze_frontend_kernel(frontend_kernel)
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertEqual(semantic_kernel.parameters[0].type.rank, 5)
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticTileType)
        self.assertEqual(semantic_kernel.parameters[1].type.shape, (8, 16))
        self.assertIsInstance(semantic_kernel.parameters[2].type, SemanticScalarType)

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)
        self.assertTrue(rows_assign.targets[0].ssa_name.startswith("%rows_"))

        vecscope_stmt = semantic_kernel.body[2]
        self.assertIsInstance(vecscope_stmt, SemanticStrictVecscopeStmt)
        self.assertEqual(
            [binding.name for binding in vecscope_stmt.block_arguments],
            ["vin", "vtmp", "factor", "lb", "ub", "vec_step"],
        )
        self.assertIsInstance(vecscope_stmt.block_arguments[0].type, SemanticTensorViewType)
        self.assertIsInstance(vecscope_stmt.block_arguments[1].type, SemanticTileType)
        self.assertIsInstance(vecscope_stmt.block_arguments[2].type, SemanticScalarType)
        self.assertIsInstance(vecscope_stmt.block_arguments[3].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[4].type, SemanticIndexType)
        self.assertIsInstance(vecscope_stmt.block_arguments[5].type, SemanticIndexType)
        self.assertTrue(vecscope_stmt.block_arguments[0].ssa_name.startswith("%vin_"))

        loop_stmt = vecscope_stmt.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(loop_stmt.induction_variable.name, "lane")
        self.assertIsInstance(loop_stmt.induction_variable.type, SemanticIndexType)
        self.assertTrue(loop_stmt.induction_variable.ssa_name.startswith("%lane_"))
        self.assertEqual(loop_stmt.loop_carried, ())

        text = specialized.mlir_text()
        self.assertIn("%rows_", text)
        self.assertIn("= arith.constant 8 : index", text)
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %arg2, %c0, %rows_\d+, %rows_\d+\)",
        )
        self.assertIn("^bb0(", text)
        self.assertIn("scf.for %lane_", text)
        self.assertIn("to %ub_6 step %vec_step_7 {", text)

    def test_tensorview_defaults_to_5d_shape_profile(self) -> None:
        @pto.vkernel(op="tensorview_5d_shape_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            d0, d1, d2, d3, d4 = inp.valid_shape
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticTensorViewType)
        self.assertEqual(semantic_kernel.parameters[0].type.rank, 5)
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview")],
        )

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>) "
            "attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 5)

    def test_tensorview_strides_profile_lowers_through_explicit_stride_queries(self) -> None:
        @pto.vkernel(op="tensorview_5d_stride_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            s0, s1, s2, s3, s4 = inp.strides
            for lane in range(0, s4, 1):
                current = lane
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview")],
        )

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>) "
            "attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertEqual(text.count("pto.get_tensor_view_stride"), 5)
        self.assertRegex(text, r"scf\.for %lane_\d+ = %c0 to %s4_\d+ step %c1 \{")

    def test_tensorview_accepts_full_5d_slice_profile(self) -> None:
        @pto.vkernel(op="tensorview_5d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:1, 0:2, 0:3, 0:4, 0:5]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 5)
        self.assertEqual(slice_assign.value.type.extents, (1, 2, 3, 4, 5))
        self.assertEqual(slice_assign.value.type.physical_axes, (0, 1, 2, 3, 4))

    def test_tensorview_3d_slice_profile_right_aligns_into_5d_descriptor(self) -> None:
        @pto.vkernel(op="tensorview_3d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:8, 0:16, 0:32]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 3)
        self.assertEqual(slice_assign.value.type.extents, (8, 16, 32))
        self.assertEqual(slice_assign.value.type.physical_axes, (2, 3, 4))

    def test_tensorview_2d_slice_profile_right_aligns_into_5d_descriptor(self) -> None:
        @pto.vkernel(op="tensorview_2d_slice_profile_unique", dtypes=[(pto.f32,)])
        def kernel(inp: pto.TensorView):
            view = inp[0:16, 0:32]
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        slice_assign = semantic_kernel.body[0]
        self.assertIsInstance(slice_assign, SemanticAssignStmt)
        self.assertEqual(slice_assign.value.type.rank, 2)
        self.assertEqual(slice_assign.value.type.extents, (16, 32))
        self.assertEqual(slice_assign.value.type.physical_axes, (3, 4))

    def test_dynamic_tensorview_shape_profile_supports_runtime_bound_without_high_level_dma(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            rows = inp.shape[0]
            for lane in range(0, rows, 1):
                current = lane
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("inp", "tensorview"), ("tile", "tile")],
        )

        rows_assign = semantic_kernel.body[0]
        self.assertIsInstance(rows_assign, SemanticAssignStmt)
        self.assertIsInstance(rows_assign.targets[0].type, SemanticIndexType)

        loop_stmt = semantic_kernel.body[1]
        self.assertIsInstance(loop_stmt, SemanticForStmt)

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.get_tensor_view_dim", text)

    def test_semantic_recognizes_padmode_symbol(self) -> None:
        @pto.vkernel(op="pad_mode_symbol", dtypes=[(pto.f32, pto.f32)])
        def kernel(inp: pto.TensorView, tile: pto.Tile):
            mode = pto.PadMode.PadFirstElem
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        assign_stmt = semantic_kernel.body[0]
        self.assertIsInstance(assign_stmt, SemanticAssignStmt)
        self.assertIsInstance(assign_stmt.value, SemanticSymbolExpr)
        self.assertEqual(assign_stmt.value.value, pto.PadMode.PadFirstElem)
        self.assertEqual(assign_stmt.value.type.kind, "pad_mode")


    def test_make_mask_vlds_vsts_and_vector_families_lower_inside_strict_vecscope(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(tile: pto.Tile, scale: pto.f32):
            with pto.strict_vecscope(tile, tile, scale, 0, 256, 64) as (
                src,
                dst,
                factor,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    biased = pto.vadds(vec, factor, mask)
                    summed = pto.vadd(biased, vec, mask)
                    activated = pto.vrelu(summed, mask)
                    pto.vsts(activated, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[0]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        mask_assign = loop_stmt.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertIsInstance(mask_assign.value, SemanticCallExpr)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(loop_stmt.body[-1], SemanticVectorStoreStmt)

        text = specialized.mlir_text()
        self.assertRegex(text, r'%mask_\d+ = pto\.pset_b32 "PAT_ALL" : !pto\.mask<b32>')
        self.assertRegex(text, r"%vec_\d+ = pto\.vlds %src_\d+\[%lane_\d+\] : !pto\.ptr<f32, ub> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%biased_\d+ = pto\.vadds %vec_\d+, %factor_\d+, %mask_\d+ : !pto\.vreg<64xf32>, f32, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%summed_\d+ = pto\.vadd %biased_\d+, %vec_\d+, %mask_\d+ : !pto\.vreg<64xf32>, !pto\.vreg<64xf32>, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"%activated_\d+ = pto\.vrelu %summed_\d+, %mask_\d+ : !pto\.vreg<64xf32>, !pto\.mask<b32> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"pto\.vsts %activated_\d+, %dst_\d+\[%lane_\d+\], %mask_\d+ : !pto\.vreg<64xf32>, !pto\.ptr<f32, ub>, !pto\.mask<b32>")

    def test_tail_make_mask_lowers_to_typed_plt_and_updates_remaining(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.i32)], advanced=True)
        def kernel(tile: pto.Tile, remaining: pto.i32):
            with pto.strict_vecscope(tile, tile, remaining, 0, 64, 64) as (src, dst, rem_in, lb, ub, step):
                mask, next_remaining = pto.make_mask(pto.f32, rem_in)
                vec = pto.vlds(src, lb)
                pto.vsts(vec, dst, lb, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope = semantic_kernel.body[0]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        mask_assign = vecscope.body[0]
        self.assertIsInstance(mask_assign, SemanticAssignStmt)
        self.assertEqual(mask_assign.value.name, "make_mask")
        self.assertEqual(len(mask_assign.targets), 2)
        self.assertIsInstance(mask_assign.targets[0].type, SemanticMaskType)
        self.assertIsInstance(mask_assign.targets[1].type, SemanticScalarType)
        self.assertEqual(mask_assign.targets[1].type.dtype, pto.i32)

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%mask_\d+, %next_remaining_\d+ = pto\.plt_b32 %rem_in_\d+ : i32 -> !pto\.mask<b32>, i32",
        )
        self.assertIn(
            "pto.vsts %vec_",
            text,
        )

    def test_nested_index_arithmetic_lowers_before_vector_accesses(self) -> None:
        @pto.vkernel(
            op="eltwise",
            dtypes=[(pto.f32, pto.f32, pto.f32)],
            advanced=True,
        )
        def kernel(
            lhs_tile: pto.Tile,
            rhs_tile: pto.Tile,
            dst_tile: pto.Tile,
        ):
            rows = lhs_tile.shape[0]
            cols = lhs_tile.shape[1]
            row_stride = lhs_tile.shape[1]

            with pto.strict_vecscope(
                lhs_tile,
                rhs_tile,
                dst_tile,
                rows,
                cols,
                row_stride,
                0,
                rows,
                1,
            ) as (lhs, rhs, dst, valid_rows, valid_cols, stride, row_lb, row_ub, row_step):
                for row in range(row_lb, row_ub, row_step):
                    for lane in range(0, valid_cols, 64):
                        offset = row * stride + lane
                        mask, next_remaining = pto.make_mask(pto.f32, valid_cols - lane)
                        summed = pto.vadd(pto.vlds(lhs, offset), pto.vlds(rhs, offset), mask)
                        pto.vsts(summed, dst, offset, mask)
            return None

        specialized = kernel.specialize(
            lhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            rhs_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst_tile=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertRegex(text, r"%tmp_\d+ = arith\.muli %row_\d+, %stride_\d+ : index")
        self.assertRegex(text, r"%offset_\d+ = arith\.addi %tmp_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.subi %valid_cols_\d+, %lane_\d+ : index")
        self.assertRegex(text, r"%tmp_\d+ = arith\.index_cast %tmp_\d+ : index to i32")
        self.assertIn("pto.plt_b32", text)
        self.assertIn("pto.vadd", text)

    def test_stable_mode_infers_vecscope_and_lowers_tile_vector_sugar(self) -> None:
        @pto.vkernel(op="tadd_stable", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            rows, cols = dst.valid_shape
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dtype)):
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, all_mask)
                    pto.vsts(summed, dst[row, col:], all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertIn("pto.vecscope {", text)
        self.assertNotIn("pto.strict_vecscope(", text)
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%row_\d+, %col_\d+\]")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%row_\d+, %col_\d+\], %(?:all_mask|mask)_\d+")

    def test_advanced_mode_infers_vecscope_and_lowers_tile_vector_sugar(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            rows, cols = dst.valid_shape
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dtype)):
                    lhs = pto.vlds(src0[row, col:])
                    rhs = pto.vlds(src1[row, col:])
                    summed = pto.vadd(lhs, rhs, all_mask)
                    pto.vsts(summed, dst[row, col:], all_mask)
            return None

        self.assertTrue(kernel.advanced_enabled)

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)
        vecscope = vecscope_stmts[0]
        self.assertIsInstance(vecscope, SemanticVecscopeStmt)
        outer_loop = next(stmt for stmt in vecscope.body if isinstance(stmt, SemanticForStmt))
        self.assertIsInstance(outer_loop, SemanticForStmt)
        inner_loop = outer_loop.body[0]
        self.assertIsInstance(inner_loop, SemanticForStmt)
        self.assertTrue(inner_loop.body)

        text = specialized.mlir_text()
        self.assertIn("// tilelang.advanced = True", text)
        self.assertIn("pto.vecscope {", text)
        self.assertNotIn("pto.strict_vecscope(", text)
        self.assertRegex(text, r"pto\.vecscope \{\n(?:.|\n)*scf\.for %row_")
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertIn("!pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)
        self.assertRegex(text, r"pto\.tile_buf_addr %arg1 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64")
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%row_\d+, %col_\d+\] : memref<8x64xf32, #pto\.address_space<vec>> -> !pto\.vreg<64xf32>")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%row_\d+, %col_\d+\], %(?:all_mask|mask)_\d+ : !pto\.vreg<64xf32>, memref<8x64xf32, #pto\.address_space<vec>>, !pto\.mask<b32>")
        self.assertNotRegex(text, r"arith\.muli %row_\d+, %c64 : index")
        self.assertNotRegex(text, r"arith\.addi %tmp_\d+, %col_\d+ : index")
        self.assertLess(text.index("pto.tile_buf_addr %arg1"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_buf_addr %arg2"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_buf_addr %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_rows %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_cols %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.vecscope {"), text.index("scf.for %row_"))
        self.assertLess(text.rindex("pto.vecscope {"), text.index("return"))

    def test_element_type_valid_shape_and_get_lanes_surface_lower_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tadd", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("step %c64", text)
        self.assertRegex(text, r"%mask_\d+, %remained_\d+ = pto\.plt_b32 %remained_iter_\d+ : i32 -> !pto\.mask<b32>, i32")
        self.assertIn("pto.vadd", text)
        self.assertIn("pto.vsts", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertRegex(text, r"pto\.vlds %tmp_\d+\[%row_\d+, %col_\d+\]")
        self.assertRegex(text, r"pto\.vsts %summed_\d+, %tmp_\d+\[%row_\d+, %col_\d+\], %mask_\d+")

    def test_bytewidth_surface_lowers_to_constant_index(self) -> None:
        @pto.vkernel(op="bytewidth_query_unique", dtypes=[(pto.f32,)], advanced=True)
        def kernel(dst: pto.Tile):
            elem_bytes = pto.bytewidth(dst.element_type)
            rows, cols = dst.valid_shape
            for col in range(0, cols, elem_bytes):
                current = col
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("= arith.constant 4 : index", text)
        self.assertRegex(text, r"scf\.for %col_\d+ = %c0 to %cols_\d+ step %elem_bytes_\d+")
        self.assertIn("pto.tile_valid_cols %arg0", text)

    def test_scalar_loop_prologue_does_not_force_vecscope_into_inner_loop(self) -> None:
        @pto.vkernel(op="tadd_outer_scope_unique", dtypes=[(pto.f32, pto.f32, pto.f32)])
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
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

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)
        outer_loop = vecscope_stmts[0].body[0]
        self.assertIsInstance(outer_loop, SemanticForStmt)
        self.assertIsInstance(outer_loop.body[0], SemanticAssignStmt)
        self.assertIsInstance(outer_loop.body[1], SemanticForStmt)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertRegex(text, r"pto\.vecscope \{\n\s+scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertNotRegex(text, r"scf\.for %row_\d+ = [^\n]+\{\n\s+pto\.vecscope \{")

    def test_unused_tile_does_not_hoist_tile_buf_addr_or_valid_shape_intrinsics(self) -> None:
        @pto.vkernel(op="tile_usage_scan_unique", dtypes=[(pto.f32, pto.f32, pto.f32)], advanced=True)
        def kernel(dst: pto.Tile, src: pto.Tile, scratch: pto.Tile):
            rows, cols = dst.valid_shape
            mask = pto.make_mask(dst.element_type, pto.PAT.ALL)
            for row in range(0, rows, 1):
                for col in range(0, cols, pto.get_lanes(dst.element_type)):
                    value = pto.vlds(src[row, col:])
                    pto.vsts(value, dst[row, col:], mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            scratch=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertIn("pto.tile_buf_addr %arg0", text)
        self.assertIn("pto.tile_buf_addr %arg1", text)
        self.assertNotIn("pto.tile_buf_addr %arg2", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)

    def test_tile_dynamic_valid_shape_profile_lowers_to_runtime_bounds_in_advanced_mode(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(op="tadd_dynamic_valid_shape_unique", dtypes=[(elem, elem, elem)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "tadd_dynamic_valid_shape_unique",
            (pto.f16, pto.f16, pto.f16),
        )
        specialized = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", "valid_cols"),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_0", "tile_valid_shape"),
                ("__valid_shape_dst_1", "tile_valid_shape"),
            ],
        )
        self.assertEqual(semantic_kernel.tile_bindings[0].valid_shape, (None, None))

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg1: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg2: !pto.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertIn("valid_shape=(?, ?)", text)
        self.assertIn("pto.vecscope {", text)
        self.assertIn("step %c128", text)
        self.assertIn("pto.tile_valid_rows %arg0", text)
        self.assertIn("pto.tile_valid_cols %arg0", text)
        self.assertNotIn("pto.tile_valid_rows %arg1", text)
        self.assertNotIn("pto.tile_valid_cols %arg1", text)
        self.assertNotIn("pto.tile_valid_rows %arg2", text)
        self.assertNotIn("pto.tile_valid_cols %arg2", text)
        self.assertLess(text.index("pto.tile_valid_rows %arg0"), text.index("pto.vecscope {"))
        self.assertLess(text.index("pto.tile_valid_cols %arg0"), text.index("pto.vecscope {"))
        self.assertRegex(text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")
        self.assertRegex(text, r"%tmp_\d+ = arith\.index_cast %valid_cols_\d+ : index to i32")
        self.assertRegex(
            text,
            r"pto\.tile_buf_addr %arg1 : !pto\.tile_buf<loc=vec, dtype=f16, rows=8, cols=128, v_row=8, v_col=128",
        )
        self.assertRegex(
            text,
            r"pto\.vlds %tmp_\d+\[%row_\d+, %col_\d+\] : memref<8x128xf16, #pto\.address_space<vec>> -> !pto\.vreg<128xf16>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %summed_\d+, %tmp_\d+\[%row_\d+, %col_\d+\], %mask_\d+ : !pto\.vreg<128xf16>, memref<8x128xf16, #pto\.address_space<vec>>, !pto\.mask<b16>",
        )

    def test_tile_partial_dynamic_valid_shape_profile_tracks_dynamic_axes_only(self) -> None:
        elem = pto.TypeVar("Elem")

        @pto.vkernel(op="tadd_partial_dynamic_valid_shape_unique", dtypes=[(elem, elem, elem)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
            dtype = dst.element_type
            valid_rows, valid_cols = dst.valid_shape
            remained = valid_cols
            for row in range(0, valid_rows, 1):
                for col in range(0, valid_cols, pto.get_lanes(dtype)):
                    mask, remained = pto.make_mask(dtype, remained)
                    summed = pto.vadd(pto.vlds(src0[row, col:]), pto.vlds(src1[row, col:]), mask)
                    pto.vsts(summed, dst[row, col:], mask)
            return None

        selected = pto.select_kernel(
            "a5",
            "tadd_partial_dynamic_valid_shape_unique",
            (pto.f16, pto.f16, pto.f16),
        )

        rows_dynamic = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=("valid_rows", 128),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )
        rows_dynamic_semantic = analyze_frontend_kernel(build_frontend_kernel_node(rows_dynamic))
        self.assertEqual(
            [(param.name, param.kind) for param in rows_dynamic_semantic.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_0", "tile_valid_shape"),
            ],
        )
        rows_dynamic_text = rows_dynamic.mlir_text()
        self.assertIn("valid_shape=(?, 128)", rows_dynamic_text)
        self.assertIn("pto.tile_valid_rows %arg0", rows_dynamic_text)
        self.assertIn("pto.tile_valid_cols %arg0", rows_dynamic_text)
        self.assertRegex(rows_dynamic_text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(rows_dynamic_text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")

        cols_dynamic = selected.specialize(
            dst=pto.TileSpecialization(
                shape=(8, 128),
                memory_space=pto.MemorySpace.UB,
                valid_shape=(8, "valid_cols"),
            ),
            src0=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 128), memory_space=pto.MemorySpace.UB),
        )
        cols_dynamic_semantic = analyze_frontend_kernel(build_frontend_kernel_node(cols_dynamic))
        self.assertEqual(
            [(param.name, param.kind) for param in cols_dynamic_semantic.parameters],
            [
                ("dst", "tile"),
                ("src0", "tile"),
                ("src1", "tile"),
                ("__valid_shape_dst_1", "tile_valid_shape"),
            ],
        )
        cols_dynamic_text = cols_dynamic.mlir_text()
        self.assertIn("valid_shape=(8, ?)", cols_dynamic_text)
        self.assertIn("pto.tile_valid_rows %arg0", cols_dynamic_text)
        self.assertIn("pto.tile_valid_cols %arg0", cols_dynamic_text)
        self.assertRegex(cols_dynamic_text, r"scf\.for %row_\d+ = %c0 to %valid_rows_\d+ step %c1")
        self.assertRegex(cols_dynamic_text, r"scf\.for %col_\d+ = %c0 to %valid_cols_\d+ step %c128")

    def test_advanced_mode_scalar_boundaries_split_inferred_vecscope_runs(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            dtype = src.element_type
            first_mask = pto.make_mask(dtype, pto.PAT.ALL)
            first = pto.vlds(src[0, 0:])
            pto.vsts(first, dst[0, 0:], first_mask)
            boundary = 1
            second_mask = pto.make_mask(dtype, pto.PAT.ALL)
            second = pto.vlds(src[1, 0:])
            pto.vsts(second, dst[1, 0:], second_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 2)

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 2)
        self.assertLess(text.index("pto.vecscope {"), text.index("%boundary_"))
        self.assertLess(text.index("%boundary_"), text.index("return"))
        self.assertLess(text.index("%boundary_"), text.rindex("pto.vecscope {"))

    def test_advanced_mode_control_flow_infers_vecscope_per_branch(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile, flag: pto.i32):
            dtype = src.element_type
            all_mask = pto.make_mask(dtype, pto.PAT.ALL)
            if flag:
                first = pto.vlds(src[0, 0:])
                pto.vsts(first, dst[0, 0:], all_mask)
            else:
                second = pto.vlds(src[1, 0:])
                pto.vsts(second, dst[1, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual([type(stmt).__name__ for stmt in semantic_kernel.body[:-1]], [
            "SemanticAssignStmt",
            "SemanticAssignStmt",
            "SemanticIfStmt",
        ])
        if_stmt = semantic_kernel.body[2]
        self.assertIsInstance(if_stmt, SemanticIfStmt)
        self.assertEqual(len(if_stmt.then_body), 1)
        self.assertEqual(len(if_stmt.else_body), 1)
        self.assertIsInstance(if_stmt.then_body[0], SemanticVecscopeStmt)
        self.assertIsInstance(if_stmt.else_body[0], SemanticVecscopeStmt)

        text = specialized.mlir_text()
        self.assertIn("scf.if", text)
        self.assertEqual(text.count("pto.vecscope {"), 2)
        self.assertLess(text.index("scf.if"), text.index("pto.vecscope {"))
        self.assertLess(text.index("scf.if"), text.index("return"))

    def test_advanced_mode_keeps_strict_vecscope_as_hard_boundary(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.Tile):
            all_mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            rows = src.shape[0]
            for row in range(0, rows, 1):
                vec = pto.vlds(src[row, 0:])
                pto.vsts(vec, dst[row, 0:], all_mask)
            with pto.strict_vecscope(src, dst, all_mask, 0, 64, 64) as (vin, vout, mask, lb, ub, step):
                for lane in range(lb, ub, step):
                    scoped = pto.vlds(vin, lane)
                    pto.vsts(scoped, vout, lane, mask)
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.vecscope {"), 1)
        self.assertEqual(text.count("pto.strict_vecscope("), 1)

    def test_advanced_mode_lowers_raw_pointer_and_low_level_dma_surface(self) -> None:
        @pto.vkernel(op="ptr_dma", dtypes=[(pto.f32, pto.f32, pto.i64)], advanced=True)
        def kernel(
            src_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            dst_gm: pto.ptr(pto.f32, pto.MemorySpace.GM),
            addr: pto.i64,
        ):
            ub_src = pto.castptr(addr, pto.ptr(pto.f32, pto.MemorySpace.UB))
            ub_dst = pto.addptr(ub_src, 64)
            mask = pto.make_mask(pto.f32, pto.PAT.ALL)
            vec = pto.vlds(ub_src, 0)
            pto.vsts(vec, ub_dst, 0, mask)

            src_bytes = pto.castptr(src_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            dst_bytes = pto.castptr(dst_gm, pto.ptr(pto.i8, pto.MemorySpace.GM))
            src_offset = pto.addptr(src_bytes, 0)
            dst_offset = pto.addptr(dst_bytes, 0)
            typed_src = pto.castptr(src_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))
            typed_dst = pto.castptr(dst_offset, pto.ptr(pto.f32, pto.MemorySpace.GM))

            pto.set_loop2_stride_outtoub(4096, 4096)
            pto.set_loop1_stride_outtoub(4096, 4096)
            pto.set_loop_size_outtoub(1, 1)
            pto.copy_gm_to_ubuf(typed_src, ub_src, 0, 32, 128, 0, 0, False, 0, 128, 128)

            pto.set_loop2_stride_ubtoout(4096, 4096)
            pto.set_loop1_stride_ubtoout(4096, 4096)
            pto.set_loop_size_ubtoout(1, 1)
            pto.copy_ubuf_to_ubuf(ub_src, ub_dst, 0, 32, 128, 128, 128)
            pto.copy_ubuf_to_gm(ub_dst, typed_dst, 0, 32, 128, 0, 128, 128)
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertIsInstance(semantic_kernel.parameters[0].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[0].type.memory_space, "gm")
        self.assertIsInstance(semantic_kernel.parameters[1].type, SemanticPtrType)
        self.assertEqual(semantic_kernel.parameters[1].type.memory_space, "gm")
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = kernel.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: i64) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertRegex(
            text,
            r"%ub_src_\d+ = pto\.castptr %arg2 : i64 -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(
            text,
            r"%ub_dst_\d+ = pto\.addptr %ub_src_\d+, %c64 : !pto\.ptr<f32, ub> -> !pto\.ptr<f32, ub>",
        )
        self.assertIn("pto.vecscope {", text)
        self.assertRegex(
            text,
            r"%vec_\d+ = pto\.vlds %ub_src_\d+\[%c0\] : !pto\.ptr<f32, ub> -> !pto\.vreg<64xf32>",
        )
        self.assertRegex(
            text,
            r"pto\.vsts %vec_\d+, %ub_dst_\d+\[%c0\], %mask_\d+ : !pto\.vreg<64xf32>, !pto\.ptr<f32, ub>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%src_bytes_\d+ = pto\.castptr %arg0 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_bytes_\d+ = pto\.castptr %arg1 : !pto\.ptr<f32, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%src_offset_\d+ = pto\.addptr %src_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"%dst_offset_\d+ = pto\.addptr %dst_bytes_\d+, %c0 : !pto\.ptr<i8, gm> -> !pto\.ptr<i8, gm>",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_outtoub %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %typed_src_\d+, %ub_src_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %false, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.set_loop2_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop1_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.set_loop_size_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64",
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_ubuf %ub_src_\d+, %ub_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )
        self.assertIn(
            ": !pto.ptr<f32, ub>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64",
            text,
        )
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_gm %ub_dst_\d+, %typed_dst_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_as_ptr_method_and_keyword_low_level_dma_surface_lower_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tensorview_tile_as_ptr_dma_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(inp: pto.TensorView, dst: pto.Tile):
            gm_ptr = inp.as_ptr()
            ub_ptr = dst.as_ptr()

            pto.set_loop2_stride_outtoub(src_stride=4096, dst_stride=2048)
            pto.set_loop1_stride_outtoub(src_stride=1024, dst_stride=512)
            pto.set_loop_size_outtoub(loop1=1, loop2=1)
            pto.copy_gm_to_ubuf(
                src=gm_ptr,
                dst=ub_ptr,
                n_burst=1,
                len_burst=64,
                gm_stride=128,
                ub_stride=128,
                enable_ub_pad=False,
            )
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%gm_ptr_\d+ = pto\.castptr %arg0 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> !pto\.ptr<f32, gm>",
        )
        self.assertRegex(
            text,
            r"%ub_ptr_\d+ = pto\.castptr %arg1 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(text, r"pto\.set_loop2_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop1_stride_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop_size_outtoub %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(
            text,
            r"pto\.copy_gm_to_ubuf %gm_ptr_\d+, %ub_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %false, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_copy_ubuf_to_gm_keyword_surface_lowers_in_advanced_mode(self) -> None:
        @pto.vkernel(op="tile_to_tensorview_dma_unique", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.Tile, dst: pto.TensorView):
            ub_ptr = src.as_ptr()
            gm_ptr = dst.as_ptr()

            pto.set_loop2_stride_ubtoout(src_stride=4096, dst_stride=2048)
            pto.set_loop1_stride_ubtoout(src_stride=1024, dst_stride=512)
            pto.set_loop_size_ubtoout(loop1=1, loop2=1)
            pto.copy_ubuf_to_gm(
                src=ub_ptr,
                dst=gm_ptr,
                n_burst=1,
                len_burst=64,
                gm_stride=128,
                ub_stride=128,
            )
            return None

        specialized = kernel.specialize(
            src=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertTrue(any(isinstance(stmt, SemanticDmaConfigStmt) for stmt in semantic_kernel.body))
        self.assertTrue(any(isinstance(stmt, SemanticLowLevelCopyStmt) for stmt in semantic_kernel.body))

        text = specialized.mlir_text()
        self.assertRegex(
            text,
            r"%ub_ptr_\d+ = pto\.castptr %arg0 : !pto\.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto\.ptr<f32, ub>",
        )
        self.assertRegex(
            text,
            r"%gm_ptr_\d+ = pto\.castptr %arg1 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> !pto\.ptr<f32, gm>",
        )
        self.assertRegex(text, r"pto\.set_loop2_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop1_stride_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(text, r"pto\.set_loop_size_ubtoout %tmp_\d+, %tmp_\d+ : i64, i64")
        self.assertRegex(
            text,
            r"pto\.copy_ubuf_to_gm %ub_ptr_\d+, %gm_ptr_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+, %tmp_\d+",
        )

    def test_if_compare_or_condition_lowers_to_cmp_and_bool_ops(self) -> None:
        @pto.vkernel(op="if_compare_or", dtypes=[(pto.f32,)])
        def kernel(src: pto.TensorView):
            loop1 = src.shape[3]
            loop2 = src.shape[4]
            step = 64
            if loop1 != 1 or loop2 != 1:
                step = 128
            else:
                step = 64
            return None

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(kernel))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("src", "tensorview")],
        )
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        condition = semantic_kernel.body[3].condition
        self.assertIsInstance(condition, SemanticBinaryExpr)
        self.assertEqual(condition.op, "or")
        self.assertIsInstance(condition.lhs, SemanticBinaryExpr)
        self.assertEqual(condition.lhs.op, "ne")
        self.assertIsInstance(condition.rhs, SemanticBinaryExpr)
        self.assertEqual(condition.rhs.op, "ne")

        text = kernel.mlir_text()
        self.assertEqual(text.count("arith.cmpi ne"), 2)
        self.assertRegex(text, r"%loop1_\d+ = pto\.get_tensor_view_dim %arg0, %c3 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index")
        self.assertRegex(text, r"%loop2_\d+ = pto\.get_tensor_view_dim %arg0, %c4 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index")
        self.assertRegex(text, r"arith\.cmpi ne, %loop1_\d+, %c1 : index")
        self.assertRegex(text, r"arith\.cmpi ne, %loop2_\d+, %c1 : index")
        self.assertRegex(text, r"arith\.ori %tmp_\d+, %tmp_\d+ : i1")
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")

    def test_shape_and_stride_tuple_unpacking_lower_cleanly(self) -> None:
        @pto.vkernel(op="shape_stride_unpack", dtypes=[(pto.f32, pto.f32)], advanced=True)
        def kernel(src: pto.TensorView, dst: pto.Tile):
            g0, g1, g2, g3, g4 = src.shape
            s0, s1, s2, s3, s4 = src.strides
            ub_rows, ub_cols = dst.shape
            total = g0 + g1 + g2 + g3 + g4
            stride_total = s0 + s1 + s2 + s3 + s4
            area = ub_rows * ub_cols
            if total != 0 or stride_total != area:
                total = area
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(
            [(param.name, param.kind) for param in semantic_kernel.parameters],
            [("src", "tensorview"), ("dst", "tile")],
        )

        text = specialized.mlir_text()
        self.assertEqual(text.count("pto.get_tensor_view_dim"), 5)
        self.assertEqual(text.count("pto.get_tensor_view_stride"), 5)
        self.assertRegex(text, r"%ub_rows_\d+ = arith\.constant 8 : index")
        self.assertRegex(text, r"%ub_cols_\d+ = arith\.constant 64 : index")

    def test_advanced_mode_lowers_compare_predicate_carry_and_rearrangement_families(self) -> None:
        @pto.vkernel(op="advanced_family", dtypes=[(pto.i32, pto.i32, pto.i32, pto.i32)], advanced=True)
        def kernel(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile, scalar: pto.i32):
            all_mask = pto.make_mask(pto.i32, pto.PAT.ALL)
            lhs = pto.vlds(src0[0, 0:])
            rhs = pto.vlds(src1[0, 0:])
            cmp_mask = pto.vcmp(lhs, rhs, all_mask, "lt")
            cmp_scalar_mask = pto.vcmps(lhs, scalar, all_mask, "gt")
            negated = pto.pnot(cmp_mask, all_mask)
            picked = pto.psel(cmp_mask, negated, cmp_scalar_mask)
            packed = pto.ppack(picked, "PART_EVEN")
            unpacked = pto.punpack(packed, "PART_ODD")
            sum_vec, carry_mask = pto.vaddc(lhs, rhs, all_mask)
            diff_vec, borrow_mask = pto.vsubc(lhs, rhs, all_mask)
            sum_with_carry, carry_mask2 = pto.vaddcs(sum_vec, diff_vec, carry_mask, all_mask)
            diff_with_borrow, borrow_mask2 = pto.vsubcs(sum_with_carry, diff_vec, borrow_mask, all_mask)
            low, high = pto.vintlv(sum_with_carry, diff_with_borrow)
            dlow, dhigh = pto.vdintlv(low, high)
            even = pto.vintlvv2(dlow, dhigh, "PART_EVEN")
            odd = pto.vdintlvv2(dlow, dhigh, "PART_ODD")
            selected = pto.vsel(even, odd, unpacked)
            selected_r = pto.vselr(selected, sum_with_carry)
            final = pto.vselrv2(selected_r, diff_with_borrow)
            pto.vsts(final, dst[0, 0:], all_mask)
            return None

        specialized = kernel.specialize(
            dst=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src0=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
            src1=pto.TileSpecialization(shape=(8, 64), memory_space=pto.MemorySpace.UB),
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        vecscope_stmts = [stmt for stmt in semantic_kernel.body if isinstance(stmt, SemanticVecscopeStmt)]
        self.assertEqual(len(vecscope_stmts), 1)

        text = specialized.mlir_text()
        self.assertIn("pto.vecscope {", text)
        self.assertIn('pto.vcmp ', text)
        self.assertIn(', "lt" : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn('pto.vcmps ', text)
        self.assertIn(', "gt" : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>', text)
        self.assertIn(" = pto.pnot ", text)
        self.assertIn(" = pto.psel ", text)
        self.assertIn(' = pto.ppack ', text)
        self.assertIn('"PART_EVEN"', text)
        self.assertIn(' = pto.punpack ', text)
        self.assertIn('"PART_ODD"', text)
        self.assertRegex(
            text,
            r"%sum_vec_\d+, %carry_mask_\d+ = pto\.vaddc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_vec_\d+, %borrow_mask_\d+ = pto\.vsubc %lhs_\d+, %rhs_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%sum_with_carry_\d+, %carry_mask2_\d+ = pto\.vaddcs %sum_vec_\d+, %diff_vec_\d+, %carry_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%diff_with_borrow_\d+, %borrow_mask2_\d+ = pto\.vsubcs %sum_with_carry_\d+, %diff_vec_\d+, %borrow_mask_\d+, %all_mask_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32>, !pto\.mask<b32>, !pto\.mask<b32> -> !pto\.vreg<64xi32>, !pto\.mask<b32>",
        )
        self.assertRegex(
            text,
            r"%low_\d+, %high_\d+ = pto\.vintlv %sum_with_carry_\d+, %diff_with_borrow_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertRegex(
            text,
            r"%dlow_\d+, %dhigh_\d+ = pto\.vdintlv %low_\d+, %high_\d+ : !pto\.vreg<64xi32>, !pto\.vreg<64xi32> -> !pto\.vreg<64xi32>, !pto\.vreg<64xi32>",
        )
        self.assertIn(" = pto.vintlvv2 ", text)
        self.assertIn(" = pto.vdintlvv2 ", text)
        self.assertIn(" = pto.vsel ", text)
        self.assertIn(" = pto.vselr ", text)
        self.assertIn(" = pto.vselrv2 ", text)
        self.assertIn("pto.vsts ", text)

    def test_elementwise_kernel_positive_regression_covers_vecscope_tail_mask_and_dynamic_loop_bound(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, remaining: pto.i32):
            rows = inp.shape[0]
            with pto.strict_vecscope(tile, tile, remaining, 0, rows, 64) as (
                src,
                dst,
                rem,
                lb,
                ub,
                step,
            ):
                for lane in range(lb, ub, step):
                    mask, rem = pto.make_mask(pto.f32, rem)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertEqual(len(semantic_kernel.body), 3)
        self.assertIsInstance(semantic_kernel.body[1], SemanticStrictVecscopeStmt)

        vecscope = semantic_kernel.body[1]
        self.assertIsInstance(vecscope, SemanticStrictVecscopeStmt)
        loop_stmt = vecscope.body[0]
        self.assertIsInstance(loop_stmt, SemanticForStmt)
        self.assertEqual(len(loop_stmt.loop_carried), 1)
        self.assertEqual(loop_stmt.loop_carried[0].name, "rem")

        text = specialized.mlir_text()
        self.assertIn(
            "func.func @kernel(%arg0: !pto.tensor_view<?x?x?x?x?xf32>, %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg2: i32) attributes { pto.tilelang.instance } {",
            text,
        )
        self.assertRegex(
            text,
            r"%rows_\d+ = pto\.get_tensor_view_dim %arg0, %c0 : !pto\.tensor_view<\?x\?x\?x\?x\?xf32> -> index",
        )
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %arg2, %c0, %rows_\d+, %c64\)",
        )
        self.assertRegex(
            text,
            r"scf\.for %lane_\d+ = %lb_\d+ to %ub_\d+ step %step_\d+ iter_args\(%rem_iter_\d+ = %rem_\d+\) -> \(i32\) \{",
        )
        self.assertRegex(
            text,
            r"%mask_\d+, %rem_\d+ = pto\.plt_b32 %rem_iter_\d+ : i32 -> !pto\.mask<b32>, i32",
        )

    def test_if_else_and_sync_ops_lower_to_scf_if_and_authoring_sync_ops(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f32, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, flag: pto.i32):
            pto.set_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            pto.wait_flag(pto.PIPE.MTE2, pto.PIPE.V, pto.EVENT.ID0)
            step = 64
            if flag:
                step = 64
                pto.set_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            else:
                step = 128
                pto.wait_flag(pto.PIPE.V, pto.PIPE.MTE3, pto.EVENT.ID0)
            with pto.strict_vecscope(tile, tile, 0, 256, step) as (src, dst, lb, ub, vec_step):
                for lane in range(lb, ub, vec_step):
                    mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                    vec = pto.vlds(src, lane)
                    pto.vsts(vec, dst, lane, mask)
            pto.pipe_barrier(pto.PIPE.ALL)
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(16, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        semantic_kernel = analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIsInstance(semantic_kernel.body[0], SemanticSetFlagStmt)
        self.assertIsInstance(semantic_kernel.body[1], SemanticWaitFlagStmt)
        self.assertIsInstance(semantic_kernel.body[3], SemanticIfStmt)
        self.assertIsInstance(semantic_kernel.body[5], SemanticPipeBarrierStmt)

        text = specialized.mlir_text()
        self.assertIn('pto.set_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_MTE2", "PIPE_V", "EVENT_ID0"]', text)
        self.assertIn("= arith.cmpi ne, %arg2, %c0_i32 : i32", text)
        self.assertRegex(text, r"%step_\d+ = scf\.if %tmp_\d+ -> \(index\) \{")
        self.assertIn('pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertIn('pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]', text)
        self.assertRegex(text, r"scf\.yield %step_\d+ : index")
        self.assertIn("%step_2 = arith.constant 128 : index", text)
        self.assertRegex(
            text,
            r"pto\.strict_vecscope\(%tmp_\d+, %tmp_\d+, %c0, %c256, %step_\d+\)",
        )
        self.assertIn("scf.for %lane_", text)
        self.assertIn("pto.barrier #pto.pipe<PIPE_ALL>", text)

    def test_strict_vecscope_rejects_implicit_capture_during_semantic_analysis(self) -> None:
        @pto.vkernel(op="eltwise", dtypes=[(pto.f32, pto.f16, pto.i32)], advanced=True)
        def kernel(inp: pto.TensorView, tile: pto.Tile, scale: pto.i32):
            with pto.strict_vecscope(inp, tile) as (vin, vtmp):
                leaked = scale
            return None

        specialized = kernel.specialize(
            tile=pto.TileSpecialization(
                shape=(8, 16),
                memory_space=pto.MemorySpace.UB,
            )
        )

        with self.assertRaises(ValueError) as ctx:
            analyze_frontend_kernel(build_frontend_kernel_node(specialized))
        self.assertIn("implicit capture of 'scale' is not allowed", str(ctx.exception))


class TileLangDSLDiagnosticsTests(unittest.TestCase):
    def test_matcher_feature_validation_rejects_invalid_constraints_and_priority(self) -> None:
        def kernel(x: pto.TensorView):
            return None

        with self.assertRaises(TypeError) as constraints_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], constraints=[123])(kernel)
        self.assertIn("constraints[0] must be callable", str(constraints_ctx.exception))

        with self.assertRaises(TypeError) as priority_ctx:
            pto.vkernel(op="x", dtypes=[(pto.f32,)], priority=True)(kernel)
        self.assertIn("priority must be an int", str(priority_ctx.exception))

    def test_advanced_mode_keeps_vreduce_rejected_until_authoring_op_exists(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.i32,)], advanced=True)
            def kernel(x: pto.Tile):
                pto.vreduce(x)
                return None

        self.assertIn("advanced family surface `pto.vreduce`", str(ctx.exception))

    def test_unsupported_python_syntax_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                while True:
                    return None

        self.assertIn("unsupported Python syntax `while`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_arbitrary_external_call_reports_source_location(self) -> None:
        def helper():
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                helper()
                return None

        self.assertIn("arbitrary external call `helper`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_unsupported_pto_surface_reports_source_location(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32,)])
            def kernel(x: pto.TensorView):
                pto.not_a_real_surface(x)
                return None

        self.assertIn("unsupported op surface `pto.not_a_real_surface`", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_strict_vecscope_requires_advanced_mode(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f32)])
            def kernel(x: pto.TensorView, tile: pto.Tile):
                with pto.strict_vecscope(tile, tile, 0, 256, 64) as (lhs, rhs, lb, ub, step):
                    pass
                return None

        self.assertIn("surface `pto.strict_vecscope` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_advanced_family_requires_advanced_mode(self) -> None:
        with self.assertRaises(pto.TileLangFrontendError) as ctx:

            @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f32)])
            def kernel(x: pto.TensorView, tile: pto.Tile):
                mask = pto.make_mask(pto.f32, pto.PAT.ALL)
                pto.vcmp(tile, tile, mask, "lt")
                return None

        self.assertIn("surface `pto.vcmp` requires advanced=True", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_missing_specialization_reports_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as ctx:
            kernel.mlir_text()

        self.assertIn("requires specialize() bindings for bare Tile parameters", str(ctx.exception))
        self.assertIn(f"{__file__}:", str(ctx.exception))

    def test_dynamic_shape_and_illegal_profile_report_source_location(self) -> None:
        @pto.vkernel(op="x", dtypes=[(pto.f32, pto.f16)])
        def kernel(x: pto.TensorView, tile: pto.Tile):
            return None

        with self.assertRaises(pto.TileLangFrontendError) as dynamic_ctx:
            kernel.specialize(tile={"shape": (16, "n"), "memory_space": "ub"})
        self.assertIn("dynamic physical Tile shape is not supported", str(dynamic_ctx.exception))
        self.assertIn(f"{__file__}:", str(dynamic_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as rank_ctx:
            kernel.specialize(tile={"shape": (4, 4, 4), "memory_space": "ub"})
        self.assertIn("v1 only supports rank-1 or rank-2 Tile shapes", str(rank_ctx.exception))
        self.assertIn(f"{__file__}:", str(rank_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as space_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "gm"})
        self.assertIn("v1 only supports MemorySpace.UB", str(space_ctx.exception))
        self.assertIn(f"{__file__}:", str(space_ctx.exception))

        with self.assertRaises(pto.TileLangFrontendError) as valid_shape_ctx:
            kernel.specialize(tile={"shape": (4, 4), "memory_space": "ub", "valid_shape": (5, 4)})
        self.assertIn("valid_shape axis 0=5 must be <= shape axis 0=4", str(valid_shape_ctx.exception))
        self.assertIn(f"{__file__}:", str(valid_shape_ctx.exception))


if __name__ == "__main__":
    unittest.main()
