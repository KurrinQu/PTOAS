# OpLib Transforms Specification

## ADDED Requirements

### Requirement: Correctness in level3 compilation

The OpLib transformation pipeline SHALL ensure correctness for level3 compilation by preserving critical hardware-related attributes and fixing layout inference.

#### Scenario: Fix address space and layout issues

- `AllocTileOp` address space attributes SHALL be preserved during OpLib instantiation and inlining.
- Layout inference SHALL correctly calculate strides for `SubView` and `Reshape` operations.
- The EmitC codegen SHALL generate necessary type casts for hardware compatibility.
