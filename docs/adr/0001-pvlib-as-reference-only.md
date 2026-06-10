# ADR 0001: pvlib is validation ground truth, never a runtime dependency

## Status
Accepted

## Decision
pvlib is installed only via the [dev] extra and may only be imported in
validation/ and tests/. core/, registry/, and api/ must never import it.
A guard test (tests/test_no_pvlib_in_core.py) enforces this.

## Rationale
Bankability requires that we can state precisely how our continuous solvers
differ from the established reference implementation. If pvlib leaks into the
runtime, that comparison becomes circular and the differentiation claim
(continuous vs binned Perez, transient vs steady-state thermal) collapses.
