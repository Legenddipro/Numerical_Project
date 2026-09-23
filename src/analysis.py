"""Analysis and figures.

Consumes the raw CSV written by montecarlo.py and produces:

    work-precision diagrams   error against number of RHS calls, in the format
                              of Figure 1 of the base paper
    energy-drift comparisons  invariant error over time, baseline against the
                              relaxed variants
    root-finder comparison    iterations, attainable precision, and failure rate
                              for the four methods
    Monte Carlo summaries     distributions rather than single-trajectory values

Interfaces to be fixed in the Phase 0 contract session.
"""
