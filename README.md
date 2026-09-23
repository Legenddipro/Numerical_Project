# Performance Optimization and Monte Carlo Stress-Testing of Relaxation FSAL Runge–Kutta Schemes

**Application to Conservative Systems: Nonlinear Pendulum Analysis**

Course project — Numerical Analysis, Simulation and Modeling
Section C1, Group 4 — Department of CSE

## Team

| Member | Student ID |
|---|---|
| Prachurja Dhar | 2105150 |
| MD. Abir Hossain | 2105133 |
| Asikur Rahman | 2105144 |
| MD. Shadman Shafie | 2105134 |
| Tousif Fahmeed Quadir | 2105127 |

## What this project does

Relaxation Runge–Kutta methods conserve a physical invariant (e.g. energy) exactly by
scaling each numerical update with a relaxation parameter $\gamma$. Combining relaxation
with the First-Same-as-Last (FSAL) technique normally costs one extra right-hand-side
evaluation per step, cancelling FSAL's efficiency gain. Bleecke & Ranocha (2026) resolve
this with two new schemes, **FSAL-R** and **R-FSAL**.

This project re-implements those schemes in Python and extends the original study in
three directions:

1. **New application.** The base paper's headline experiment is a PDE (Benjamin–Bona–Mahony).
   We apply the methods to a conservative ODE — the nonlinear pendulum
   $\theta'' + \tfrac{g}{L}\sin\theta = 0$, with invariant
   $E = \tfrac12\omega^2 - \tfrac{g}{L}\cos\theta$.

2. **Generalized root-finding for $\gamma$.** The base paper uses a single bracketing
   method (Algorithm 748 of Alefeld–Potra–Shi). We benchmark four:
   Newton–Raphson, bisection, golden-section search, and Algorithm 748 — comparing
   cost, attainable precision, and failure behaviour.

3. **Monte Carlo robustness study.** Rather than a single fixed initial condition, we
   sweep many randomly sampled $(\theta_0, \omega_0)$ and report distributions of
   RHS-evaluation counts, energy drift, and global error.

## Repository structure

```
src/
  problems.py        pendulum RHS, invariant, gradient, reference solution, IC sampling
  tableaus.py        BS3 and DP5 Butcher tableaus, PID controller gains
  stepping.py        PID step-size controller, error estimate, initial step-size heuristic
  rootfind.py        Newton, bisection, golden-section, Algorithm 748
  solvers/
    classic.py       baseline (no relaxation), naive relaxation+FSAL, FSAL-R
    rfsal.py         R-FSAL
  montecarlo.py      initial-condition sampling and sweep harness
  analysis.py        work-precision diagrams, drift plots, summary tables
tests/               one test module per source module
docs/
  ALGORITHMS.md      pseudocode transcription of the four solver loops
results/             generated data (not version-controlled)
```

## Status

**Phase 0 — Foundation.** Repository scaffolded. Interface contract (function signatures
and return types) to be agreed by the whole team before implementation begins.

## Development workflow

- `main` is protected; no direct pushes.
- One branch per person per task: `feature/<name>-<module>`.
- All changes land via reviewed Pull Request.
- **One owner per file.** Two people must never edit the same file — this is the
  primary defence against merge conflicts.

## Running

To be documented once the implementation lands.

## Acknowledgements and citation

This project builds directly on the methods and reference implementation of:

```bibtex
@article{bleecke2026relaxation,
  title={Relaxation {R}unge-{K}utta methods with First-Same-as-Last
         Structure},
  author={Bleecke, Sebastian and Ranocha, Hendrik},
  journal={Journal of Scientific Computing},
  volume={106},
  pages={18},
  year={2026},
  doi={10.1007/s10915-025-03130-6},
  eprint={2311.14050},
  eprinttype={arxiv},
  eprintclass={math.NA}
}
```

Our solver loops are ported from the authors' reproducibility repository, which their
README asks to be cited alongside the article:

```bibtex
@misc{bleecke2026relaxationRepro,
  title={Reproducibility repository for
         "{R}elaxation {R}unge-{K}utta methods with First-Same-as-Last
         Structure"},
  author={Bleecke, Sebastian and Ranocha, Hendrik},
  year={2023},
  howpublished={\url{https://github.com/ranocha/2023_FSAL_relaxation}},
  doi={10.5281/zenodo.10201246}
}
```

Original repository: https://github.com/ranocha/2023_FSAL_relaxation (MIT licensed).

## License

MIT — see [LICENSE](LICENSE).
