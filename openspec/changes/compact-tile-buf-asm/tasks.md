## 1. Parser And Printer Refactor

- [x] 1.1 Refactor `TileBufType::parse` to support both legacy keyed syntax and compact positional syntax.
- [x] 1.2 Keep one shared semantic construction path after parsing so both syntaxes produce identical `TileBufType` values.
- [x] 1.3 Update `TileBufType::print` to emit canonical compact syntax with deterministic keyed suffix ordering.
- [x] 1.4 Implement default-field elision using `TileBufConfigAttr::getDefault(...)` and `validShape == shape`.
- [x] 1.5 Preserve existing parser-time normalization behavior for `loc=left` under scoped target arch.

## 2. Regression Coverage

- [x] 2.1 Add focused parse/print tests for fully default compact tile buf output.
- [x] 2.2 Add tests for non-default `valid`, `blayout`, `slayout`, `fractal`, and `pad` suffix printing.
- [x] 2.3 Add compatibility tests proving legacy keyed syntax still parses and reprints canonically in compact form.
- [x] 2.4 Add negative tests for malformed compact syntax and invalid compact field combinations.
- [x] 2.5 Add A3/A5 coverage for `loc=left` normalization under compact syntax.

## 3. Real-IR Validation

- [x] 3.1 Run focused `ptoas` checks on `test/basic` inputs affected by parser/printer changes.
- [x] 3.2 Run a readability smoke check on `test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto`.
- [x] 3.3 Confirm no semantic regressions by verifying equivalent legacy and compact forms lower identically for representative cases.
