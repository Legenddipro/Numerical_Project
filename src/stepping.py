"""Error-based step-size control.

Houses the PID controller (which sets the next step size from the last three
error estimates), the scaled error norm comparing the main and embedded
solutions, and the heuristic that picks the very first step size.

Deliberately separate from the solver loops so that all four method variants
share one controller implementation and differences between them cannot be
attributed to differing step-size logic.

Interfaces to be fixed in the Phase 0 contract session.
"""
