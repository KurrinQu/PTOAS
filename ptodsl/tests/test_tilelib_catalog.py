# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Table-driven selection and render coverage for the PTODSL TileLib catalog."""

import unittest

import ptodsl.tilelib as tilelib
from ptodsl.tilelib import ScalarType, TileSpec, select


# op -> (template name, rendered vector op, parameter names, representative dtype)
CATALOG = {
    "pto.tabs": ("template_tabs", "pto.vabs", ("src", "dst"), "f32"),
    "pto.tand": ("template_tand", "pto.vand", ("src0", "src1", "dst"), "i32"),
    "pto.tcolexpand": ("template_tcolexpand", "pto.vlds", ("src", "dst"), "f32"),
    "pto.tcolmax": ("template_tcolmax", "pto.vmax", ("src", "dst"), "f32"),
    "pto.tcolmin": ("template_tcolmin", "pto.vmin", ("src", "dst"), "f32"),
    "pto.tcolprod": ("template_tcolprod", "pto.vmul", ("src", "dst"), "f32"),
    "pto.tcolsum": ("template_tcolsum", "pto.vadd", ("src", "dst"), "f32"),
    "pto.tneg": ("template_tneg", "pto.vneg", ("src", "dst"), "f32"),
    "pto.tnot": ("template_tnot", "pto.vnot", ("src", "dst"), "i32"),
    "pto.tor": ("template_tor", "pto.vor", ("src0", "src1", "dst"), "i32"),
    "pto.trelu": ("template_trelu", "pto.vrelu", ("src", "dst"), "f32"),
    "pto.trowexpand": ("template_trowexpand", "pto.vdup", ("src", "dst"), "f32"),
    "pto.tshl": ("template_tshl", "pto.vshl", ("src0", "src1", "dst"), "i32"),
    "pto.tshr": ("template_tshr", "pto.vshr", ("src0", "src1", "dst"), "i32"),
    "pto.txor": (
        "template_txor",
        "pto.vxor",
        ("src0", "src1", "tmp", "dst"),
        "i32",
    ),
}

COLUMN_REDUCTIONS = {"pto.tcolmax", "pto.tcolmin", "pto.tcolprod", "pto.tcolsum"}
SPECIAL_VALID_SHAPES = {
    ("pto.tcolexpand", "src"): (1, 64),
    ("pto.trowexpand", "src"): (8, 1),
}
SHARED_RENDERED_OPS = (
    "pto.tile_buf_addr",
    "memref.subview",
    "scf.for",
    "pto.vlds",
    "pto.vsts",
    "pto.tilelang.instance",
)


def _specs(op, parameter_names, dtype_name):
    dtype = ScalarType(dtype_name)
    specs = {}
    for name in parameter_names:
        valid_shape = SPECIAL_VALID_SHAPES.get((op, name), (8, 64))
        if op in COLUMN_REDUCTIONS and name == "dst":
            valid_shape = (1, 64)
        specs[name] = TileSpec(
            shape=(8, 64),
            dtype=dtype,
            valid_shape=valid_shape,
        )
    return specs


class TileLibCatalogTest(unittest.TestCase):
    def test_tilelib_does_not_duplicate_the_public_ptodsl_surface(self):
        for name in (
            "Tile",
            "PostUpdate",
            "get_lanes",
            "make_mask",
            "vlds",
            "vadd",
            "vsts",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(tilelib, name))

    def test_each_catalog_entry_selects_and_renders(self):
        for op, (name, vector_op, parameter_names, dtype_name) in CATALOG.items():
            with self.subTest(op=op):
                specs = _specs(op, parameter_names, dtype_name)
                descriptor = select(op, "a5", specs)
                self.assertEqual(descriptor.name, name)

                mlir = descriptor.specialize(**specs).mlir_text()
                self.assertIn(vector_op, mlir)
                for shared_op in SHARED_RENDERED_OPS:
                    self.assertIn(shared_op, mlir)
                self.assertNotIn("pto.castptr", mlir)

    def test_declared_dtype_signatures_are_selectable(self):
        for op, (_, _, parameter_names, representative_dtype) in CATALOG.items():
            first_specs = _specs(op, parameter_names, representative_dtype)
            descriptor = select(op, "a5", first_specs)
            for signature in descriptor.metadata.dtypes:
                with self.subTest(op=op, signature=signature):
                    specs = {}
                    for operand, dtype_name in zip(parameter_names, signature):
                        valid_shape = SPECIAL_VALID_SHAPES.get(
                            (op, operand),
                            (8, 64),
                        )
                        if op in COLUMN_REDUCTIONS and operand == "dst":
                            valid_shape = (1, 64)
                        specs[operand] = TileSpec(
                            shape=(8, 64),
                            dtype=ScalarType(dtype_name),
                            valid_shape=valid_shape,
                        )
                    selected = select(op, "a5", specs)
                    self.assertEqual(selected.name, descriptor.name)
                    self.assertIn(
                        CATALOG[op][1],
                        selected.specialize(**specs).mlir_text(),
                    )


if __name__ == "__main__":
    unittest.main()
