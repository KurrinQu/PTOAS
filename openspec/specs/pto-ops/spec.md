# pto-ops Specification

## Purpose

TBD - created by archiving change complete-missing-ops. Update Purpose after archive.

## Requirements

### Requirement: Support basic arithmetic and activation operators

The PTO dialect SHALL include support for additional basic arithmetic and activation operators to ensure full coverage of common machine learning operations.

#### Scenario: Define new operators in PTO dialect

- The `pto` dialect SHALL support `trem` (remainder), `trems` (scalar remainder), `tprelu` (parameterized ReLU), and `tlrelu` (Leaky ReLU) operators.
- These operators SHALL be defined in `PTOOps.td`.
- These operators SHALL have appropriate C++ verifiers in `PTO.cpp`.
