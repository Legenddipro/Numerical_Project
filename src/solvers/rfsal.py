"""R-FSAL time-stepping loop.

Structurally distinct from the three variants in classic.py, hence its own
module and its own owner.

R-FSAL reorders the step: it computes stages 1..s-1 only (stage s is skipped,
since b[-1] == 0 means it never contributed to the propagated solution anyway),
relaxes immediately, then makes a single right-hand-side call at the relaxed
point. That one value serves twice -- exactly as the next step's cached first
stage, and, via extrapolation with factor 1/gamma, as the approximation of
f(u_np1) that the embedded solution requires.

So the approximation sits in a different place than in FSAL-R: R-FSAL's FSAL
reuse is exact, and it is the error estimate that is approximated.

Gate G3: RHS count must match baseline exactly.

Interfaces to be fixed in the Phase 0 contract session.
"""
