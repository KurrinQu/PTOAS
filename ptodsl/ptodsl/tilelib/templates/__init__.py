# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Lazy loader for ported, per-architecture TileLib templates."""

from functools import lru_cache
from importlib import import_module


_TEMPLATE_MODULES = {
    ("a5", "pto.tabs"): ".a5.tabs",
    ("a5", "pto.tadd"): ".a5.tadd",
    ("a5", "pto.tadds"): ".a5.tadds",
    ("a5", "pto.tand"): ".a5.tand",
    ("a5", "pto.tands"): ".a5.tands",
    ("a5", "pto.tcolmax"): ".a5.tcolmax",
    ("a5", "pto.tcolexpand"): ".a5.tcolexpand",
    ("a5", "pto.tcolexpandadd"): ".a5.tcolexpandadd",
    ("a5", "pto.tcolexpanddiv"): ".a5.tcolexpanddiv",
    ("a5", "pto.tcolexpandexpdif"): ".a5.tcolexpandexpdif",
    ("a5", "pto.tcolexpandmax"): ".a5.tcolexpandmax",
    ("a5", "pto.tcolexpandmin"): ".a5.tcolexpandmin",
    ("a5", "pto.tcolexpandmul"): ".a5.tcolexpandmul",
    ("a5", "pto.tcolexpandsub"): ".a5.tcolexpandsub",
    ("a5", "pto.tcolmin"): ".a5.tcolmin",
    ("a5", "pto.tcolprod"): ".a5.tcolprod",
    ("a5", "pto.tcolsum"): ".a5.tcolsum",
    ("a5", "pto.tdiv"): ".a5.tdiv",
    ("a5", "pto.tdivs"): ".a5.tdivs",
    ("a5", "pto.texp"): ".a5.texp",
    ("a5", "pto.texpands"): ".a5.texpand",
    ("a5", "pto.tlrelu"): ".a5.tlrelu",
    ("a5", "pto.tlog"): ".a5.tlog",
    ("a5", "pto.tmax"): ".a5.tmax",
    ("a5", "pto.tmaxs"): ".a5.tmaxs",
    ("a5", "pto.tmin"): ".a5.tmin",
    ("a5", "pto.tmins"): ".a5.tmins",
    ("a5", "pto.tmul"): ".a5.tmul",
    ("a5", "pto.tmuls"): ".a5.tmuls",
    ("a5", "pto.tneg"): ".a5.tneg",
    ("a5", "pto.tnot"): ".a5.tnot",
    ("a5", "pto.tor"): ".a5.tor",
    ("a5", "pto.tors"): ".a5.tors",
    ("a5", "pto.trelu"): ".a5.trelu",
    ("a5", "pto.trecip"): ".a5.trecip",
    ("a5", "pto.trsqrt"): ".a5.trsqrt",
    ("a5", "pto.trowexpand"): ".a5.trowexpand",
    ("a5", "pto.trowexpandadd"): ".a5.trowexpandadd",
    ("a5", "pto.trowexpanddiv"): ".a5.trowexpanddiv",
    ("a5", "pto.trowexpandexpdif"): ".a5.trowexpandexpdif",
    ("a5", "pto.trowexpandmax"): ".a5.trowexpandmax",
    ("a5", "pto.trowexpandmin"): ".a5.trowexpandmin",
    ("a5", "pto.trowexpandmul"): ".a5.trowexpandmul",
    ("a5", "pto.trowexpandsub"): ".a5.trowexpandsub",
    ("a5", "pto.tshl"): ".a5.tshl",
    ("a5", "pto.tshls"): ".a5.tshls",
    ("a5", "pto.tshr"): ".a5.tshr",
    ("a5", "pto.tshrs"): ".a5.tshrs",
    ("a5", "pto.tsub"): ".a5.tsub",
    ("a5", "pto.tsubs"): ".a5.tsubs",
    ("a5", "pto.tsqrt"): ".a5.tsqrt",
    ("a5", "pto.txor"): ".a5.txor",
    ("a5", "pto.txors"): ".a5.txors",
}


@lru_cache(maxsize=None)
def load_template(op: str, target: str) -> bool:
    """Import and register only the template module for ``(target, op)``.

    Both this cache and Python's module cache make repeated requests no-ops.
    Returns ``False`` when this TileLib has no module for the requested pair.
    """
    module_name = _TEMPLATE_MODULES.get((target, op))
    if module_name is None:
        return False
    import_module(module_name, package=__name__)
    return True


__all__ = ["load_template"]
