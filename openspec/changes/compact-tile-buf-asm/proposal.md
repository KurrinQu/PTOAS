## Why

The current `!pto.tile_buf` custom assembly always prints a full keyed parameter list, which makes PTO IR difficult to read in fusion-heavy regions and noisy in diffs. We want the default textual form to emphasize semantically important differences while keeping existing IR fully parseable.

## What Changes

- Change the default `!pto.tile_buf` printer from verbose keyed syntax to a compact canonical syntax.
- Elide redundant fields from the default printer when they match type defaults or can be derived from shape.
- Accept both the existing verbose keyed syntax and the new compact syntax in the parser.
- Keep `tile_buf` semantics, layout rules, verifier behavior, and lowering behavior unchanged.
- Do not add alias printing or a separate verbose-print mode in this change.

## Capabilities

### New Capabilities
- `tile-buf-asm`: Compact canonical assembly syntax and backward-compatible parsing for `!pto.tile_buf`.

### Modified Capabilities
- None.

## Impact

Affected areas:
- `lib/PTO/IR/PTOTypeDefs.cpp` custom parser/printer for `TileBufType`
- `lib/PTO/IR/PTOAttrs.cpp` default config interaction used for field elision
- `tools/ptoas/ptoas.cpp` parser arch-scoping behavior that affects `loc=left`
- `test/basic/` textual IR parser/printer regression coverage
- Real-world readability smoke checks using existing sample IR
