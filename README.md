# Regression Calibration as a Baseline for Machine-Learned Covariates

[Paper](https://gsood.com/research/papers/calibration_baseline.pdf)

An ML prediction used as a control can change adjustment for confounding. Established regression calibration is a useful baseline: learn the true covariate from its prediction **and all observed downstream regressors**, then use the calibrated value downstream.

The paper separates the population argument from the difficulty of learning calibration. A population linear projection can suffice even when it is not the conditional mean. Finite labeled samples, regularization, insufficient independent variation, and transport to a different population each create distinct problems. The contribution is an exposition and diagnostic framework for this application, with exact counterexamples and illustrative simulations; it is not a new calibration estimator or evidence of superiority over PPI.

## What is established, and what this repository does

Regression calibration predates modern ML. The paper anchors the argument in Carroll and Stefanski (1990), Boe et al. (2023), and the literature on predicted covariates, including Haber, Sampson, and Graubard (2021). It distinguishes this model-based baseline from moment corrections and prediction-assisted inference, including Fong and Tyler (2021) and Kluger et al. (2025).

For a linear model `Y = beta_D D + beta_X X + beta_Z Z + error`:

1. Fit calibration of `X` on the surrogate, `D`, and `Z` in an independent labeled sample.
2. Predict calibrated `X` in the target sample and regress `Y` on `D`, calibrated `X`, and `Z`.
3. For regular fixed-dimensional fits, resample both independent samples and refit both stages to account for calibration uncertainty.

This requires the stated outcome exclusion condition, sufficient rank, and valid calibration for the target population. It does not by itself establish causality. The bootstrap statement does not extend automatically to arbitrary regularized or flexible calibration.

## Illustrations

- Omitting treatment from calibration changes its population coefficient from 0.5 to 1 in an exact confounding example.
- A fixed ridge penalty of 0.1, even with treatment unpenalized, changes the limiting treatment coefficient from 0.5 to 0.4 in that example.
- Adding irrelevant calibration predictors increases fitting complexity while leaving the outcome regression and population support fixed.
- Separate analytic examples show why stable conditional means do not transport a restricted projection, and why more labels within the same restricted support cannot learn calibration elsewhere.

The simulation uses a frozen, independently trained affine ridge prediction to isolate these mechanisms. It is not a benchmark of modern text or image learners. The infeasible oracle is labeled as such; nonunique OLS fits are counted, not silently pseudoinverted. The primary figure uses medians and quartiles: near-zero fitted calibration slopes create heavy ratio tails, so finite-run RMSE is only a descriptive summary. All reported numbers and exhibits are generated from code.

## Reproduce

Python 3.11+ and a LaTeX installation with `latexmk`, `pdflatex`, and `bibtex` are required. The lockfile records the dependency resolution used for replication.

```bash
uv sync --extra dev
uv run make check
uv run ./build.sh
```

For a smoke run without replacing the paper's full results:

```bash
uv run python make_tables.py --quick --output /tmp/calibration-smoke
```

To compile the included full exhibits without rerunning simulations:

```bash
make paper
```

The full design uses 400 replications per cell and 400 inference replications with 399 bootstrap resamples each. `analysis_plan.json` records the design and its retrospective revision context. This is not a preregistered study.

## Files

| File | Purpose |
|---|---|
| `paper.tex`, `paper.pdf` | Manuscript and compiled paper |
| `references.bib` | Source-checked bibliography |
| `calibration.py` | OLS, HC1, calibration, two-sample bootstrap |
| `make_tables.py` | Simulation, numerical summaries, tables, figure |
| `analysis_plan.json` | Estimand, design, seeds, failure policy |
| `results/` | Replication results and environment metadata |
| `tables/`, `figures/` | Generated manuscript exhibits |
| `tests/` | Analytical and numerical regression checks |

This repository was renamed from `tworeg` and revised to remove unsupported novelty claims, inaccurate references, and unvalidated comparisons. Its history contains the final revision only, as requested by the repository owner.
