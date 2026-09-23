"""Baseline, naive relaxation, and FSAL-R time-stepping loops.

These three share the great majority of their structure, so they live in one
module under one owner rather than being split across files.

    baseline  plain embedded RK with FSAL reuse; no relaxation. The invariant
              drifts. Reference point for cost.
    naive     relaxation applied after the step is accepted. Because integration
              continues from the relaxed point, the cached FSAL value is for the
              wrong point and must be recomputed -- one extra RHS call per step,
              exactly cancelling the FSAL saving.
    fsalr     as naive, but the next step's first stage is obtained by
              interpolating between two already-computed stage values rather
              than by a fresh RHS call. Same cost as baseline.

Gate G3: baseline and fsalr must report identical integer RHS counts; naive must
be higher by the ratio s/(s-1).

Interfaces to be fixed in the Phase 0 contract session.
"""
