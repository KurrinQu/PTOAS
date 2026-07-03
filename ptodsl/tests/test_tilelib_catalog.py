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
from ptodsl.tilelib import ScalarSpec, ScalarType, TileSpec, select


# op -> (template name, rendered vector op, parameter names, representative dtype)
CATALOG = {
    "pto.tabs": ("template_tabs", "pto.vabs", ("src", "dst"), "f32"),
    "pto.tand": ("template_tand", "pto.vand", ("src0", "src1", "dst"), "i32"),
    "pto.tands": ("template_tands", "pto.vand", ("src", "scalar", "dst"), "i32"),
    "pto.tcmp": ("template_tcmp", "pto.vcmp", ("src0", "src1", "dst"), "f32"),
    "pto.tcmps": ("template_tcmps", "pto.vcmps", ("src", "scalar", "dst"), "f32"),
    "pto.tcolexpand": ("template_tcolexpand", "pto.vlds", ("src", "dst"), "f32"),
    "pto.tcolexpandadd": ("template_tcolexpandadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpanddiv": ("template_tcolexpanddiv", "pto.vdiv", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandexpdif": (
        "template_tcolexpandexpdif_f32",
        "pto.vexpdif",
        ("src0", "src1", "dst"),
        "f32",
    ),
    "pto.tcolexpandmax": ("template_tcolexpandmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandmin": ("template_tcolexpandmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandmul": ("template_tcolexpandmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.tcolexpandsub": ("template_tcolexpandsub", "pto.vsub", ("src0", "src1", "dst"), "f32"),
    "pto.tcolargmax": ("template_tcolargmax_f32_to_i32", "pto.vcmp", ("src", "tmp", "dst"), "f32"),
    "pto.tcolargmin": ("template_tcolargmin_f32_to_i32", "pto.vcmp", ("src", "tmp", "dst"), "f32"),
    "pto.tcolmax": ("template_tcolmax", "pto.vmax", ("src", "dst"), "f32"),
    "pto.tcolmin": ("template_tcolmin", "pto.vmin", ("src", "dst"), "f32"),
    "pto.tcolprod": ("template_tcolprod", "pto.vmul", ("src", "dst"), "f32"),
    "pto.tcolsum": ("template_tcolsum", "pto.vadd", ("src", "dst"), "f32"),
    "pto.texpands": ("template_texpands", "pto.vdup", ("scalar", "dst"), "f32"),
    "pto.textract": ("template_textract_vec2vec_nd", "pto.vlds", ("src", "index_row", "index_col", "dst"), "f32"),
    "pto.tlrelu": ("template_tlrelu", "pto.vlrelu", ("src", "slope", "dst"), "f32"),
    "pto.tlog": ("template_tlog", "pto.vln", ("src", "dst"), "f32"),
    "pto.tdivs": ("template_tdivs", "pto.vdiv", ("src", "scalar", "dst"), "f32"),
    "pto.tcvt": ("template_tcvt_f32_to_i32", "pto.vcvt", ("src", "dst"), "f32"),
    "pto.texp": ("template_texp", "pto.vexp", ("src", "dst"), "f32"),
    "pto.tfmod": ("template_tfmod", "pto.vtrc", ("src0", "src1", "dst"), "f32"),
    "pto.tfmods": ("template_tfmods", "pto.vtrc", ("src", "scalar", "dst"), "f32"),
    "pto.tfillpad": ("template_tfillpad", "pto.vsts", ("src", "dst"), "f32"),
    "pto.tfillpad_expand": ("template_tfillpad_expand", "pto.vsts", ("src", "dst"), "f32"),
    "pto.tfillpad_inplace": ("template_tfillpad_inplace", "pto.vdup", ("src", "dst"), "f32"),
    "pto.tinsert": ("template_tinsert_vec_to_vec_nd_basic", "pto.vsts", ("src", "index_row", "index_col", "dst"), "f32"),
    "pto.tneg": ("template_tneg", "pto.vneg", ("src", "dst"), "f32"),
    "pto.tnot": ("template_tnot", "pto.vnot", ("src", "dst"), "i32"),
    "pto.tor": ("template_tor", "pto.vor", ("src0", "src1", "dst"), "i32"),
    "pto.tors": ("template_tors", "pto.vor", ("src", "scalar", "dst"), "i32"),
    "pto.tpartadd": ("template_tpartadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmax": ("template_tpartmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmin": ("template_tpartmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.tpartmul": ("template_tpartmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.tprelu": ("template_tprelu", "pto.vprelu", ("src0", "src1", "tmp", "dst"), "f32"),
    "pto.trandom": ("template_trandom", "pto.vmull", ("key0", "key1", "counter0", "counter1", "counter2", "counter3", "dst"), "ui32"),
    "pto.trelu": ("template_trelu", "pto.vrelu", ("src", "dst"), "f32"),
    "pto.trecip": ("template_trecip", "pto.vdiv", ("src", "dst"), "f32"),
    "pto.trem": ("template_trem", "pto.vtrc", ("src0", "src1", "tmp", "dst"), "f32"),
    "pto.trems": ("template_trems", "pto.vtrc", ("src", "scalar", "tmp", "dst"), "f32"),
    "pto.trsqrt": ("template_trsqrt", "pto.vsqrt", ("src", "dst"), "f32"),
    "pto.trowargmax": ("template_trowargmax", "pto.vdintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowargmin": ("template_trowargmin", "pto.vdintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowexpand": ("template_trowexpand", "pto.vdup", ("src", "dst"), "f32"),
    "pto.trowexpandadd": ("template_trowexpandadd", "pto.vadd", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpanddiv": ("template_trowexpanddiv", "pto.vdiv", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandexpdif": (
        "template_trowexpandexpdif_f32",
        "pto.vexpdif",
        ("src0", "src1", "dst"),
        "f32",
    ),
    "pto.trowexpandmax": ("template_trowexpandmax", "pto.vmax", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandmin": ("template_trowexpandmin", "pto.vmin", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandmul": ("template_trowexpandmul", "pto.vmul", ("src0", "src1", "dst"), "f32"),
    "pto.trowexpandsub": ("template_trowexpandsub", "pto.vsub", ("src0", "src1", "dst"), "f32"),
    "pto.trowmax": ("template_trowmax", "pto.vcmax", ("src", "tmp", "dst"), "f32"),
    "pto.trowmin": ("template_trowmin", "pto.vcmin", ("src", "tmp", "dst"), "f32"),
    "pto.trowprod": ("template_trowprod", "pto.vintlv", ("src", "tmp", "dst"), "f32"),
    "pto.trowsum": ("template_trowsum", "pto.vcadd", ("src", "tmp", "dst"), "f32"),
    "pto.tsel": ("template_tsel", "pto.vsel", ("mask", "src0", "src1", "tmp", "dst"), "f32"),
    "pto.tsels": ("template_tsels", "pto.vsel", ("mask", "src", "tmp", "scalar", "dst"), "f32"),
    "pto.tshl": ("template_tshl", "pto.vshl", ("src0", "src1", "dst"), "i32"),
    "pto.tshls": ("template_tshls", "pto.vshls", ("src", "scalar", "dst"), "i32"),
    "pto.tshr": ("template_tshr", "pto.vshr", ("src0", "src1", "dst"), "i32"),
    "pto.tshrs": ("template_tshrs", "pto.vshrs", ("src", "scalar", "dst"), "i32"),
    "pto.tmrgsort": ("template_tmrgsort_multi_list2", "pto.vmrgsort4", ("src0", "src1", "tmp", "dst", "ex_vec"), "f32"),
    "pto.tsort32": ("template_tsort32", "pto.vbitsort", ("src", "idx", "dst"), "f32"),
    "pto.tadds": ("template_tadds", "pto.vadds", ("src", "scalar", "dst"), "f32"),
    "pto.tmaxs": ("template_tmaxs", "pto.vmaxs", ("src", "scalar", "dst"), "f32"),
    "pto.tmins": ("template_tmins", "pto.vmins", ("src", "scalar", "dst"), "f32"),
    "pto.tmuls": ("template_tmuls", "pto.vmuls", ("src", "scalar", "dst"), "f32"),
    "pto.txor": (
        "template_txor",
        "pto.vxor",
        ("src0", "src1", "tmp", "dst"),
        "i32",
    ),
    "pto.txors": ("template_txors", "pto.vxor", ("src", "scalar", "tmp", "dst"), "i32"),
    "pto.tsubs": ("template_tsubs", "pto.vadds", ("src", "scalar", "dst"), "f32"),
    "pto.tsqrt": ("template_tsqrt", "pto.vsqrt", ("src", "dst"), "f32"),
}

COLUMN_REDUCTIONS = {"pto.tcolmax", "pto.tcolmin", "pto.tcolprod", "pto.tcolsum"}
ARG_COLUMN_REDUCTIONS = {"pto.tcolargmax", "pto.tcolargmin"}
ROW_REDUCTIONS = {
    "pto.trowargmax",
    "pto.trowargmin",
    "pto.trowmax",
    "pto.trowmin",
    "pto.trowprod",
    "pto.trowsum",
}
SPECIAL_VALID_SHAPES = {
    ("pto.tcolexpand", "src"): (1, 64),
    ("pto.trowexpand", "src"): (8, 1),
}
for _op in (
    "pto.trowexpandadd",
    "pto.trowexpanddiv",
    "pto.trowexpandexpdif",
    "pto.trowexpandmax",
    "pto.trowexpandmin",
    "pto.trowexpandmul",
    "pto.trowexpandsub",
):
    SPECIAL_VALID_SHAPES[(_op, "src1")] = (8, 1)
for _op in (
    "pto.tcolexpandadd",
    "pto.tcolexpanddiv",
    "pto.tcolexpandexpdif",
    "pto.tcolexpandmax",
    "pto.tcolexpandmin",
    "pto.tcolexpandmul",
    "pto.tcolexpandsub",
):
    SPECIAL_VALID_SHAPES[(_op, "src1")] = (1, 64)
SHARED_RENDERED_OPS = (
    "pto.tile_buf_addr",
    "memref.subview",
    "scf.for",
    "pto.vsts",
    "pto.tilelang.instance",
)
OPS_WITHOUT_TILE_LOAD = {"pto.texpands"}
OPS_WITHOUT_TILE_LOAD = OPS_WITHOUT_TILE_LOAD | {"pto.trandom", "pto.tsort32"}
OPS_WITHOUT_TILE_LOAD = OPS_WITHOUT_TILE_LOAD | {"pto.tfillpad_inplace"}
OPS_WITHOUT_VECTOR_STORE = {"pto.tcmp", "pto.tcmps", "pto.tsort32"}
OPS_WITHOUT_MEMREF_SUBVIEW = {"pto.tsort32"}
OPS_WITHOUT_LOOP = {"pto.tmrgsort"}
OPS_ALLOWING_CASTPTR = {"pto.tsel", "pto.tsels"}
SCALAR_OPERANDS = {
    "scalar",
    "slope",
    "index_row",
    "index_col",
    "key0",
    "key1",
    "counter0",
    "counter1",
    "counter2",
    "counter3",
    "block_len",
    "ex_vec",
}
SPECIAL_SCALAR_DTYPES = {
    ("pto.tshls", "scalar"): "i16",
    ("pto.tshrs", "scalar"): "i16",
    ("pto.textract", "index_row"): "i32",
    ("pto.textract", "index_col"): "i32",
    ("pto.tinsert", "index_row"): "i32",
    ("pto.tinsert", "index_col"): "i32",
    ("pto.trandom", "key0"): "i32",
    ("pto.trandom", "key1"): "i32",
    ("pto.trandom", "counter0"): "i32",
    ("pto.trandom", "counter1"): "i32",
    ("pto.trandom", "counter2"): "i32",
    ("pto.trandom", "counter3"): "i32",
    ("pto.tmrgsort", "ex_vec"): "i32",
}
SPECIAL_OPERAND_DTYPES = {
    ("pto.tcmp", "dst"): "i8",
    ("pto.tcmps", "dst"): "ui8",
    ("pto.trandom", "dst"): "ui32",
    ("pto.tsort32", "idx"): "i32",
    ("pto.trowargmax", "dst"): "i32",
    ("pto.trowargmin", "dst"): "i32",
}
FLOAT_REMAINDER_OPS = {"pto.tfmod", "pto.tfmods", "pto.trem", "pto.trems"}
FLOAT_REMAINDER_DTYPES = {"f16", "bf16", "f32"}


def _specs(op, parameter_names, dtype_name):
    dtype = ScalarType(dtype_name)
    specs = {}
    for name in parameter_names:
        if name in SCALAR_OPERANDS:
            scalar_dtype = SPECIAL_SCALAR_DTYPES.get((op, name), dtype_name)
            specs[name] = ScalarSpec(dtype=ScalarType(scalar_dtype), value=1)
            continue
        operand_dtype = SPECIAL_OPERAND_DTYPES.get(
            (op, name),
            "i8" if name == "mask" else dtype_name,
        )
        valid_shape = SPECIAL_VALID_SHAPES.get((op, name), (8, 64))
        if op in COLUMN_REDUCTIONS and name == "dst":
            valid_shape = (1, 64)
        if op in ARG_COLUMN_REDUCTIONS and name == "dst":
            operand_dtype = "i32"
            valid_shape = (1, 64)
        if op == "pto.tcvt" and name == "dst":
            operand_dtype = "i32"
        if op in ROW_REDUCTIONS and name == "dst":
            valid_shape = (8, 1)
        specs[name] = TileSpec(
            shape=(8, 64),
            dtype=ScalarType(operand_dtype),
            valid_shape=valid_shape,
        )
    return specs


def _expected_rendered_op(op, signature):
    if op in FLOAT_REMAINDER_OPS and not any(
        dtype in FLOAT_REMAINDER_DTYPES for dtype in signature
    ):
        return "pto.vdiv"
    return CATALOG[op][1]


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
                shared_ops = SHARED_RENDERED_OPS
                if op in OPS_WITHOUT_VECTOR_STORE:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "pto.vsts"
                    )
                if op in OPS_WITHOUT_MEMREF_SUBVIEW:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "memref.subview"
                    )
                if op in OPS_WITHOUT_LOOP:
                    shared_ops = tuple(
                        shared_op for shared_op in shared_ops
                        if shared_op != "scf.for"
                    )
                for shared_op in shared_ops:
                    self.assertIn(shared_op, mlir)
                if op not in OPS_WITHOUT_TILE_LOAD:
                    self.assertIn("pto.vlds", mlir)
                if op not in OPS_ALLOWING_CASTPTR:
                    self.assertNotIn("pto.castptr", mlir)

    def test_declared_dtype_signatures_are_selectable(self):
        for op, (_, _, parameter_names, representative_dtype) in CATALOG.items():
            first_specs = _specs(op, parameter_names, representative_dtype)
            descriptor = select(op, "a5", first_specs)
            for signature in descriptor.metadata.dtypes:
                with self.subTest(op=op, signature=signature):
                    specs = {}
                    for operand, dtype_name in zip(parameter_names, signature):
                        if operand in SCALAR_OPERANDS:
                            specs[operand] = ScalarSpec(
                                dtype=ScalarType(dtype_name),
                                value=1,
                            )
                            continue
                        valid_shape = SPECIAL_VALID_SHAPES.get(
                            (op, operand),
                            (8, 64),
                        )
                        if op in COLUMN_REDUCTIONS and operand == "dst":
                            valid_shape = (1, 64)
                        if op in ARG_COLUMN_REDUCTIONS and operand == "dst":
                            valid_shape = (1, 64)
                        if op in ROW_REDUCTIONS and operand == "dst":
                            valid_shape = (8, 1)
                        specs[operand] = TileSpec(
                            shape=(8, 64),
                            dtype=ScalarType(dtype_name),
                            valid_shape=valid_shape,
                        )
                    selected = select(op, "a5", specs)
                    self.assertEqual(selected.name, descriptor.name)
                    self.assertIn(
                        _expected_rendered_op(op, signature),
                        selected.specialize(**specs).mlir_text(),
                    )


if __name__ == "__main__":
    unittest.main()
