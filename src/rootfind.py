"""Root-finding methods for the relaxation parameter gamma.

Each step solves one scalar equation for gamma:

    r(gamma) = eta(u_n + gamma * (u_np1 - u_n)) - eta(u_n) = 0

Four methods are compared on cost, attainable precision, and failure behaviour:

    newton      fast, needs d(eta)/d(gamma), started from gamma_0 = 1
    bisection   slow but robust; fails loudly when the bracket has no sign change
    golden      a MINIMIZER, so it operates on |r(gamma)|, not r(gamma);
                fails silently when no root is bracketed, so results must be
                post-checked against |r(gamma*)| ~ 0
    toms748     Algorithm 748 of Alefeld-Potra-Shi via scipy.optimize --
                the identical algorithm used by the base paper, included as the
                reference baseline for the comparison

Note that gamma = 0 always satisfies the equation. Bracketing methods must
exclude it.

Interfaces to be fixed in the Phase 0 contract session.
"""
