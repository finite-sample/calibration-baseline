"""Generate the paper's illustrative simulations and all numerical exhibits."""

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.linear_model import Ridge

from calibration import IdentificationError, bootstrap_rc, hc1, ols, rc_fit

matplotlib.use("Agg")

METHODS = (
    "Oracle",
    "Naive",
    "Validation only",
    "RC: small",
    "RC: expanded",
    "RC: ridge",
)
COLORS = ("#555555", "#b34a36", "#8b6a9b", "#0072b2", "#d58900", "#009e73")


def sample(rng, n, q, predictor):
    x = rng.normal(scale=np.sqrt(2), size=n)
    d = x / 2 + rng.normal(scale=np.sqrt(0.5), size=n)
    raw = x + rng.normal(size=n)
    prediction = predictor.predict(raw[:, None])
    noise = rng.normal(size=(n, q))
    y = 0.5 * d + x + rng.normal(size=n)
    return np.column_stack((np.ones(n), d)), np.column_stack((prediction, noise)), x, y


def reduced(data):
    v, p, x, y = data
    return v, p[:, :1], x, y


def summarize(errors):
    errors = np.asarray(errors)
    if len(errors) == 0:
        return dict(
            mean_error=None,
            rmse=None,
            median_error=None,
            median_absolute_error=None,
            absolute_q25=None,
            absolute_q75=None,
        )
    absolute = np.abs(errors)
    return dict(
        mean_error=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        median_error=float(np.median(errors)),
        median_absolute_error=float(np.median(absolute)),
        absolute_q25=float(np.quantile(absolute, 0.25)),
        absolute_q75=float(np.quantile(absolute, 0.75)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    plan = json.loads(Path(__file__).with_name("analysis_plan.json").read_text())
    reps = 20 if args.quick else plan["replications"]
    inference_reps = 20 if args.quick else plan["inference_replications"]
    draws = 49 if args.quick else plan["bootstrap_draws"]
    root = args.output
    for folder in ("tables", "figures", "results"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    train_rng = np.random.default_rng(plan["training_seed"])
    train_x = train_rng.normal(scale=np.sqrt(2), size=plan["n_training"])
    train_raw = train_x + train_rng.normal(size=len(train_x))
    predictor = Ridge(alpha=1).fit(train_raw[:, None], train_x)
    records, summary = [], []
    for n_cal in plan["n_calibration"]:
        for q in plan["extra_predictors"]:
            by_method = {name: [] for name in METHODS}
            for rep in range(reps):
                rng = np.random.default_rng([plan["seed"], n_cal, q, rep])
                cal = sample(rng, n_cal, q, predictor)
                target = sample(rng, plan["n_target"], q, predictor)
                for method in METHODS:
                    record = dict(n_cal=n_cal, q=q, rep=rep, method=method)
                    try:
                        if method == "Oracle":
                            coef = ols(
                                np.column_stack((target[0], target[2])), target[3]
                            )
                            m = target[2]
                        elif method == "Naive":
                            m = target[1][:, 0]
                            coef = ols(np.column_stack((target[0], m)), target[3])
                        elif method == "Validation only":
                            coef = ols(np.column_stack((cal[0], cal[2])), cal[3])
                            m = None
                        else:
                            small = method == "RC: small"
                            penalty = (
                                plan["ridge_penalty"] if method == "RC: ridge" else 0
                            )
                            coef, m = rc_fit(
                                reduced(cal) if small else cal,
                                reduced(target) if small else target,
                                penalty=penalty,
                            )
                        error = float(coef[1] - 0.5)
                        mse = (
                            None if m is None else float(np.mean((m - target[2]) ** 2))
                        )
                        record.update(error=error, x_mse=mse, status="ok")
                        by_method[method].append(record)
                    except IdentificationError:
                        record.update(error=None, x_mse=None, status="rank failure")
                    records.append(record)
            for method, values in by_method.items():
                mse = [v["x_mse"] for v in values if v["x_mse"] is not None]
                summary.append(
                    dict(
                        n_cal=n_cal,
                        q=q,
                        method=method,
                        successes=len(values),
                        failures=reps - len(values),
                        **summarize([v["error"] for v in values]),
                        x_mse=float(np.mean(mse)) if mse else None,
                    )
                )
            print(f"Finished n_cal={n_cal}, q={q}, replications={reps}", flush=True)
    inference = []
    for n_cal in plan["n_calibration"]:
        counts = {name: [] for name in ("Second stage HC1", "Two-sample bootstrap")}
        widths = {name: [] for name in counts}
        for rep in range(inference_reps):
            rng = np.random.default_rng([plan["seed"], 999, n_cal, rep])
            cal = sample(rng, n_cal, 0, predictor)
            target = sample(rng, plan["n_target"], 0, predictor)
            _, m = rc_fit(cal, target)
            coef, se = hc1(np.column_stack((target[0], m)), target[3])
            boot_rng = np.random.default_rng([plan["seed"], 1000, n_cal, rep])
            ci = bootstrap_rc(cal, target, boot_rng, draws)[:, 1]
            intervals = {
                "Second stage HC1": coef[1]
                + np.array([-1, 1]) * 1.959963984540054 * se[1],
                "Two-sample bootstrap": ci,
            }
            for method, (lower, upper) in intervals.items():
                counts[method].append(lower <= 0.5 <= upper)
                widths[method].append(upper - lower)
                records.append(
                    dict(
                        n_cal=n_cal,
                        q=0,
                        rep=rep,
                        method=method,
                        lower=float(lower),
                        upper=float(upper),
                        status="ok",
                    )
                )
        for method, values in counts.items():
            coverage = float(np.mean(values))
            inference.append(
                dict(
                    n_cal=n_cal,
                    method=method,
                    coverage=coverage,
                    mcse=float(np.sqrt(coverage * (1 - coverage) / inference_reps)),
                    width=float(np.mean(widths[method])),
                    failures=0,
                )
            )
        print(
            f"Finished inference n_cal={n_cal}, replications={inference_reps}",
            flush=True,
        )
    metadata = dict(
        plan=plan,
        quick=args.quick,
        replications=reps,
        inference_replications=inference_reps,
        bootstrap_draws=draws,
        frozen_prediction=dict(
            intercept=float(predictor.intercept_), slope=float(predictor.coef_[0])
        ),
        versions=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            sklearn=sklearn.__version__,
            matplotlib=matplotlib.__version__,
        ),
    )
    (root / "results/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "results/summary.json").write_text(
        json.dumps(dict(estimates=summary, inference=inference), indent=2) + "\n"
    )
    with (root / "results/replications.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (root / "results/summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    write_exhibits(root, summary, inference, metadata)


def write_exhibits(root, summary, inference, metadata):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, n_cal in zip(axes, metadata["plan"]["n_calibration"]):
        for method, color in zip(METHODS, COLORS):
            rows = [r for r in summary if r["n_cal"] == n_cal and r["method"] == method]
            rows = [r for r in rows if r["rmse"] is not None]
            ax.errorbar(
                [r["q"] for r in rows],
                [r["median_absolute_error"] for r in rows],
                yerr=[
                    [r["median_absolute_error"] - r["absolute_q25"] for r in rows],
                    [r["absolute_q75"] - r["median_absolute_error"] for r in rows],
                ],
                label=method,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.2,
                capsize=2,
            )
        ax.set(
            title=f"Calibration sample: {n_cal}", xlabel="Additional noise predictors"
        )
        ax.set_xticks(metadata["plan"]["extra_predictors"])
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Absolute treatment-coefficient error")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(root / "figures/dimension.pdf", bbox_inches="tight")
    plt.close(fig)
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Calibration complexity and treatment-coefficient error.}",
        r"\label{tab:dimension}",
        r"\begin{tabular}{rrlrrrr}\toprule",
        r"$n_c$ & $q$ & Method & Median error & Run RMSE & $X$ MSE & Failed \\\midrule",
    ]
    for row in summary:
        if row["q"] not in (0, 100):
            continue
        if row["q"] == 0 and row["method"] == "RC: expanded":
            continue
        if row["q"] == 100 and not row["method"].startswith("RC:"):
            continue
        values = [
            "---" if row[key] is None else f"{row[key]:.3f}"
            for key in ("median_error", "rmse", "x_mse")
        ]
        lines.append(
            f"{row['n_cal']} & {row['q']} & {row['method']} & "
            + " & ".join(values)
            + f" & {row['failures']} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize",
        f"Each cell uses {metadata['replications']} independent replications. "
        r"The full grid is in the replication files. At $q=0$, expanded OLS equals "
        r"small RC. $q$ counts added calibration predictors; the downstream regression "
        r"always contains "
        r"an intercept, treatment, and one true, predicted, or calibrated covariate. "
        r"$X$ MSE is evaluated on target observations. "
        r"Validation-only OLS makes no imputation. "
        r"Failed denotes nonunique OLS fits; "
        r"no pseudoinverse coefficient is substituted. "
        r"Oracle OLS uses target labels unavailable to the feasible methods. "
        r"Run RMSE is descriptive: small OLS calibration has infinite "
        r"finite-sample population MSE in this design.",
        r"\end{minipage}\end{table}",
    ]
    (root / "tables/dimension.tex").write_text("\n".join(lines) + "\n")
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{Uncertainty from estimating the small calibration model.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{rlrrr}\toprule",
        r"$n_c$ & Interval & Coverage & MC SE & Mean width \\\midrule",
    ]
    for row in inference:
        lines.append(
            f"{row['n_cal']} & {row['method']} & {row['coverage']:.3f} & "
            f"{row['mcse']:.3f} & {row['width']:.3f} \\\\"
        )
    lines += [
        r"\bottomrule\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize",
        f"Nominal coverage is 0.95; {metadata['inference_replications']} "
        "independent replications "
        f"and {metadata['bootstrap_draws']} resamples per bootstrap interval. "
        r"Both samples are independently resampled and both regressions refit. "
        r"Second-stage HC1 treats estimated calibration as fixed. "
        r"No fits failed. These results concern only the fixed, "
        r"three-parameter OLS calibration.",
        r"\end{minipage}\end{table}",
    ]
    (root / "tables/coverage.tex").write_text("\n".join(lines) + "\n")
    small = next(
        r
        for r in summary
        if r["n_cal"] == 300 and r["q"] == 0 and r["method"] == "RC: small"
    )
    validation = next(
        r
        for r in summary
        if r["n_cal"] == 300 and r["q"] == 0 and r["method"] == "Validation only"
    )
    expanded = next(
        r
        for r in summary
        if r["n_cal"] == 300 and r["q"] == 100 and r["method"] == "RC: expanded"
    )
    validation_median = validation["median_absolute_error"]
    (root / "tables/findings.tex").write_text(
        f"\\newcommand{{\\SmallMedian}}{{{small['median_absolute_error']:.3f}}}\n"
        f"\\newcommand{{\\ValidationMedian}}{{{validation_median:.3f}}}\n"
        f"\\newcommand{{\\ExpandedMedian}}{{{expanded['median_absolute_error']:.3f}}}\n"
    )
    (root / "tables/settings.tex").write_text(
        f"\\newcommand{{\\SimulationReps}}{{{metadata['replications']}}}\n"
        f"\\newcommand{{\\InferenceReps}}{{{metadata['inference_replications']}}}\n"
        f"\\newcommand{{\\BootstrapDraws}}{{{metadata['bootstrap_draws']}}}\n"
    )


if __name__ == "__main__":
    main()
