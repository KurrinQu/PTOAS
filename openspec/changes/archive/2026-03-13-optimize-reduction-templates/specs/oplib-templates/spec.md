# OpLib Templates Specification

## ADDED Requirements

### Requirement: Optimized Reduction Seed System

The OpLib reduction templates SHALL be optimized using a seed-based system to reduce code duplication and improve maintainability.

#### Scenario: Consolidate reduction templates

- A unified reduction framework template SHALL be implemented in `reduction_templates.mlir`.
- Specific reduction operations (Sum, Max, Min) SHALL be implemented as "seed" functions.
- The instantiation logic SHALL support injecting seed functions into the unified framework.
