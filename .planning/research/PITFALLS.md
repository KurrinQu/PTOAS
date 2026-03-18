# Pitfalls Research

## Pitfall 1: Losing PTO Template Semantics During Lowering

Why it matters:
The whole project exists because information is currently lost during library instantiation. A naive PTO-to-A5VM lowering can repeat the same mistake.

Warning signs:

- lowering helpers only emit hardcoded one-off ops
- no preserved distinction between layout/mode/variant cases
- adding a second PTO op requires rewriting the whole path

Prevention:

- design PTO lowering helpers around the same semantic selection points used by PTO templates
- keep variant-defining information explicit in the lowering inputs or `a5vm` attributes
- document how each lowered case maps back to PTO wrapper/builtin structure

Phase relevance:
Phase 1 and Phase 2

## Pitfall 2: Over-Scoping v1 Beyond the `Abs` Slice

Why it matters:
The backend surface is large. Trying to cover too many PTO ops before the architecture works will slow everything down.

Warning signs:

- adding ops not exercised by `Abs`
- building generic infrastructure without a concrete validation path
- deferring `runop.sh -t Abs` until late

Prevention:

- use `Abs` as the acceptance gate from the start
- only implement `TLOAD`, `TABS`, `TSTORE`, and shared helpers required by that path
- treat everything else as explicit future work

Phase relevance:
All phases

## Pitfall 3: Modeling Illegal or Ambiguous Vector Types

Why it matters:
`a5vm` has a strict 256-byte vector-width constraint. If types are not normalized, intrinsic naming and legality checks will drift.

Warning signs:

- vector element counts derived inconsistently
- ops accepting arbitrary vector lengths
- mismatches between vector type spelling and intrinsic suffix spelling

Prevention:

- centralize legal-vector construction in one helper
- verify byte width in the type or op verifier
- derive intrinsic suffixes from canonicalized vector type data

Phase relevance:
Phase 1

## Pitfall 4: Breaking the Existing Pass Pipeline

Why it matters:
The user explicitly wants only the `emitc` slot replaced, not a pipeline redesign.

Warning signs:

- changing frontend semantics to satisfy backend codegen
- making unrelated dialects illegal
- coupling the new backend to assumptions that only work for `Abs`

Prevention:

- follow current `PTOToEmitC` insertion points and pass responsibilities
- keep non-backend dialect handling unchanged where possible
- validate that the new pass boundary still accepts existing MLIR before the final emit step

Phase relevance:
Phase 1 and Phase 3

## Pitfall 5: Guessing HIVM Intrinsic Names Too Early

Why it matters:
The exact final intrinsic mapping is not fully known and must be confirmed externally.

Warning signs:

- embedding a large guessed intrinsic catalog up front
- locking naming logic before inspecting the builtins exercised by the sample

Prevention:

- first implement the framework and sample path
- then extract the exact intrinsic inventory implied by used builtins and parameterization
- confirm the final mapping with the user before expanding coverage

Phase relevance:
Phase 3
