"""Support-matrix definitions and diagnostics for TileLang DSL v1."""

from __future__ import annotations

FOLLOW_UP_CHANGE = "extend-tilelang-dsl-matcher-and-advanced-surface"

# Tier definitions for TileLang DSL surface classification
# These tiers represent the user-facing support level of language features:
# - STABLE: Core surface that is fully supported and recommended for general use
# - ADVANCED: Features requiring advanced=True, suitable for expert users

STABLE_TIER = "stable"
ADVANCED_TIER = "advanced"

# Tier metadata for PTO calls and language constructs
# This provides a unified source of truth for documentation and testing

SUPPORTED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "dma_load",
        "dma_store",
        "set_flag",
        "wait_flag",
        "pipe_barrier",
        "barrier",
    }
)

SUPPORTED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "make_mask",
        "vlds",
        "vsts",
        "vabs",
        "vrelu",
        "vexp",
        "vnot",
        "vadd",
        "vsub",
        "vmul",
        "vdiv",
        "vmax",
        "vmin",
        "vand",
        "vor",
        "vxor",
        "vadds",
        "vsubs",
        "vmuls",
        "vdivs",
        "vmaxs",
        "vmins",
    }
)

ADVANCED_VECSCOPE_PTO_CALLS = frozenset(
    {
        "vcmp",
        "vcmps",
        "vsel",
        "vselr",
        "vselrv2",
        "pnot",
        "psel",
        "ppack",
        "punpack",
        "vaddc",
        "vsubc",
        "vaddcs",
        "vsubcs",
        "vintlv",
        "vdintlv",
        "vintlvv2",
        "vdintlvv2",
    }
)

ADVANCED_EXPR_PTO_CALLS = frozenset(
    {
        "ptr",
        "castptr",
        "addptr",
    }
)

ADVANCED_TOPLEVEL_PTO_CALLS = frozenset(
    {
        "strict_vecscope",
        "copy_gm_to_ubuf",
        "copy_ubuf_to_gm",
        "copy_ubuf_to_ubuf",
        "set_loop2_stride_outtoub",
        "set_loop1_stride_outtoub",
        "set_loop_size_outtoub",
        "set_loop2_stride_ubtoout",
        "set_loop1_stride_ubtoout",
        "set_loop_size_ubtoout",
    }
)

DEFERRED_PTO_SURFACES = frozenset(
    {
        "vreduce",
    }
)

# Public surface groupings used by the guide, migration notes, and tests.
# These groupings intentionally mirror the user-facing authoring tiers rather
# than the internal lowering organization.

STABLE_TENSORVIEW_SURFACES = frozenset({"TensorView"})
STABLE_TILE_SURFACES = frozenset({"Tile"})
STABLE_HIGH_LEVEL_DMA_SURFACES = frozenset({"pto.dma_load", "pto.dma_store"})
STABLE_BASE_VECTOR_SURFACES = frozenset(
    f"pto.{name}" for name in sorted(SUPPORTED_VECSCOPE_PTO_CALLS)
)

ADVANCED_RAW_POINTER_SURFACES = frozenset(
    {
        "ptr",
        "pto.ptr",
        "PointerType",
        "GMPtr",
        "UBPtr",
        "UBRef",
        "pto.castptr",
        "pto.addptr",
    }
)
ADVANCED_LOW_LEVEL_DMA_SURFACES = frozenset(
    {
        "pto.copy_gm_to_ubuf",
        "pto.copy_ubuf_to_gm",
        "pto.copy_ubuf_to_ubuf",
        "pto.set_loop2_stride_outtoub",
        "pto.set_loop1_stride_outtoub",
        "pto.set_loop_size_outtoub",
        "pto.set_loop2_stride_ubtoout",
        "pto.set_loop1_stride_ubtoout",
        "pto.set_loop_size_ubtoout",
    }
)
ADVANCED_EXPLICIT_VECSCOPE_SURFACES = frozenset({"pto.strict_vecscope"})
ADVANCED_TILE_HELPER_SURFACES = frozenset(
    {
        "tile.slice",
        "tile.reshape",
        "tile.to_ubref",
        "tile.as_ptr",
        "tile.to_memref",
        "pto.tile_from_ptr",
        "pto.tile_from_memref",
        "pto.tile_with_strides",
        "pto.tile_config",
    }
)
STABLE_TILE_INDEXING_SURFACES = frozenset(
    {
        "tile[start:]",
        "tile[row, col:]",
    }
)

AUTHORING_TIER_SURFACE_GROUPS = {
    "TensorView": STABLE_TENSORVIEW_SURFACES,
    "Tile": STABLE_TILE_SURFACES,
    "dma_load/store": STABLE_HIGH_LEVEL_DMA_SURFACES,
    "base_vector_ops": STABLE_BASE_VECTOR_SURFACES,
    "tile_indexing_sugar": STABLE_TILE_INDEXING_SURFACES,
    "strict_vecscope": ADVANCED_EXPLICIT_VECSCOPE_SURFACES,
    "raw_pointer_family": ADVANCED_RAW_POINTER_SURFACES,
    "low_level_dma_family": ADVANCED_LOW_LEVEL_DMA_SURFACES,
    "tile_helper_family": ADVANCED_TILE_HELPER_SURFACES,
}

AUTHORING_TIER_GROUP_TIERS = {
    "TensorView": STABLE_TIER,
    "Tile": STABLE_TIER,
    "dma_load/store": STABLE_TIER,
    "base_vector_ops": STABLE_TIER,
    "tile_indexing_sugar": STABLE_TIER,
    "strict_vecscope": ADVANCED_TIER,
    "raw_pointer_family": ADVANCED_TIER,
    "low_level_dma_family": ADVANCED_TIER,
    "tile_helper_family": ADVANCED_TIER,
}


def unsupported_feature_message(feature: str) -> str:
    return (
        f"{feature} is not supported in TileLang DSL v1; "
        f"see follow-up change `{FOLLOW_UP_CHANGE}`"
    )


def deferred_surface_message(name: str) -> str:
    return unsupported_feature_message(f"advanced family surface `pto.{name}`")


def advanced_mode_message(name: str) -> str:
    return f"surface `pto.{name}` requires advanced=True in TileLang DSL"


# Tier mapping for PTO calls
def get_pto_call_tier(call_name: str) -> str:
    """Return the tier of a PTO call.

    Args:
        call_name: Name of the PTO call (without 'pto.' prefix)

    Returns:
        One of STABLE_TIER or ADVANCED_TIER

    Raises:
        KeyError: If the PTO call is not part of the supported DSL surface
    """
    if call_name in SUPPORTED_TOPLEVEL_PTO_CALLS:
        return STABLE_TIER
    if call_name in SUPPORTED_VECSCOPE_PTO_CALLS:
        return STABLE_TIER
    if call_name in ADVANCED_VECSCOPE_PTO_CALLS:
        return ADVANCED_TIER
    if call_name in ADVANCED_EXPR_PTO_CALLS:
        return ADVANCED_TIER
    if call_name in ADVANCED_TOPLEVEL_PTO_CALLS:
        return ADVANCED_TIER
    raise KeyError(unsupported_feature_message(f"pto.{call_name}"))


UNSUPPORTED_LANGUAGE_CONSTRUCTS = frozenset(
    {
        "pto.get_buf",
        "pto.rls_buf",
        "pto.dma_copy",
        "pto.vreduce",
        "pto.tile",
        "pto.memref",
        "pto.vreg",
        "pto.mask_b8",
        "pto.mask_b16",
        "pto.mask_b32",
        "BLayout",
        "SLayout",
        "PadValue",
        "SyncOpType",
    }
)


# Tier mapping for language constructs (non-PTO-call features)
# These are higher-level abstractions in the TileLang DSL
LANGUAGE_CONSTRUCT_TIERS = {
    # Stable tier constructs
    "TensorView": STABLE_TIER,
    "Tile": STABLE_TIER,
    "dma_load": STABLE_TIER,
    "dma_store": STABLE_TIER,
    "PadMode": STABLE_TIER,
    "tile[start:]": STABLE_TIER,
    "tile[row, col:]": STABLE_TIER,
    # Advanced tier constructs
    "ptr": ADVANCED_TIER,  # raw pointer constructor
    "UBRef": ADVANCED_TIER,  # UB reference type
    "GMPtr": ADVANCED_TIER,  # GM pointer type
    "UBPtr": ADVANCED_TIER,  # UB pointer type
    "strict_vecscope": ADVANCED_TIER,  # explicit vecscope management
    "pto.strict_vecscope": ADVANCED_TIER,
    "tile.slice": ADVANCED_TIER,
    "tile.reshape": ADVANCED_TIER,
    "tile.to_ubref": ADVANCED_TIER,
    "tile.as_ptr": ADVANCED_TIER,
    "tile.to_memref": ADVANCED_TIER,
    "pto.tile_from_ptr": ADVANCED_TIER,
    "pto.tile_from_memref": ADVANCED_TIER,
    "pto.tile_with_strides": ADVANCED_TIER,
    "pto.tile_config": ADVANCED_TIER,
}


def get_feature_tier(feature_name: str) -> str:
    """Return the tier of a TileLang DSL feature.

    Args:
        feature_name: Name of the feature, which can be:
            - A PTO call name (e.g., 'vadd', 'ptr')
            - A language construct (e.g., 'TensorView', 'dma_load')
            - A qualified construct (e.g., 'tile.slice', 'pto.tile_from_ptr')

    Returns:
        One of STABLE_TIER or ADVANCED_TIER

    Raises:
        KeyError: If the feature is documented but not part of the supported DSL surface
    """
    # First check if it's a known language construct
    if feature_name in LANGUAGE_CONSTRUCT_TIERS:
        return LANGUAGE_CONSTRUCT_TIERS[feature_name]
    if feature_name in UNSUPPORTED_LANGUAGE_CONSTRUCTS:
        raise KeyError(unsupported_feature_message(feature_name))

    # Check if it's a PTO call (might be qualified with 'pto.' prefix)
    call_name = feature_name
    if feature_name.startswith("pto."):
        call_name = feature_name[4:]

    # Check PTO call tier
    return get_pto_call_tier(call_name)


def get_surface_group_tier(group_name: str) -> str:
    """Return the authoring tier for a documented public-surface group."""

    return AUTHORING_TIER_GROUP_TIERS[group_name]


__all__ = [
    "DEFERRED_PTO_SURFACES",
    "FOLLOW_UP_CHANGE",
    "ADVANCED_EXPR_PTO_CALLS",
    "ADVANCED_TOPLEVEL_PTO_CALLS",
    "ADVANCED_VECSCOPE_PTO_CALLS",
    "SUPPORTED_TOPLEVEL_PTO_CALLS",
    "SUPPORTED_VECSCOPE_PTO_CALLS",
    "STABLE_TIER",
    "ADVANCED_TIER",
    "STABLE_TENSORVIEW_SURFACES",
    "STABLE_TILE_SURFACES",
    "STABLE_HIGH_LEVEL_DMA_SURFACES",
    "STABLE_BASE_VECTOR_SURFACES",
    "STABLE_TILE_INDEXING_SURFACES",
    "ADVANCED_EXPLICIT_VECSCOPE_SURFACES",
    "ADVANCED_RAW_POINTER_SURFACES",
    "ADVANCED_LOW_LEVEL_DMA_SURFACES",
    "ADVANCED_TILE_HELPER_SURFACES",
    "AUTHORING_TIER_SURFACE_GROUPS",
    "AUTHORING_TIER_GROUP_TIERS",
    "UNSUPPORTED_LANGUAGE_CONSTRUCTS",
    "LANGUAGE_CONSTRUCT_TIERS",
    "advanced_mode_message",
    "deferred_surface_message",
    "unsupported_feature_message",
    "get_pto_call_tier",
    "get_feature_tier",
    "get_surface_group_tier",
]
