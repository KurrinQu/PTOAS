# Compare and Bitwise Lowering Specification

## ADDED Requirements

### Requirement: Support Compare and Bitwise OpLib Lowering

The OpLib lowering pipeline SHALL be expanded to support 14 operators from the Compare and Bitwise families, handling integer types and the currently aligned comparison/selection modes.

#### Scenario: Lower Compare and Select operators

- The `tcmp`, `tcmps`, and `tsel` operators SHALL be lowered to OpLib calls.
- Comparison modes SHALL be correctly mapped to OpLib templates.
- Byte-mask tiles SHALL be handled for mask-based selection operators.
- `tsels` MAY remain outside the active OpLib lowering path until its PTO IR contract and select-scalar templates are aligned.

#### Scenario: Lower Bitwise and Shift operators

- Bitwise operators (`tand`, `tor`, `txor`, `tnot`) and shift operators (`tshl`, `tshr`) SHALL be lowered to OpLib calls.
- Both tile-tile and tile-scalar variants SHALL be supported.
- Integer data types SHALL be supported in the OpLib lowering flow for these operators.
