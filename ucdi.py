"""
UCDI: Uncertainty-Calibrated Decision Intelligence
Reproducible experiment for: "Selective Decision Intelligence Under Predictive
Uncertainty: Calibration, Human Deferral, and Error Concentration in AI Automation"

Usage:
    python ucdi.py   (data: data/credit_default_taiwan.csv)

Produces: results/table1_models.csv, results/table2_selective.csv,
          results/seeds.csv, results/summary.json,
          figures/fig1_reliability.png, figures/fig2_risk_coverage.png
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss)

TARGET = "default payment next month"


def load(path):
    if path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=1)  # original UCI file has a two-row header
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={"default.payment.next.month": TARGET})
    y = df[TARGET].astype(int).values
    X = df.drop(columns=[c for c in ("ID", TARGET) if c in df.columns]).astype(float)
    return X, y


def split(X, y, seed):
    # 60 / 20 / 20 stratified: train, calibration, test (non-overlapping)
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xca, Xte, yca, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    return Xtr, Xca, Xte, ytr, yca, yte


def base_models(seed):
    return {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                               n_jobs=-1, random_state=seed),
        "GradBoost": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400,
                                                   max_leaf_nodes=31, l2_regularization=1.0,
                                                   random_state=seed),
    }


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(accuracy=accuracy_score(y, yhat), roc_auc=roc_auc_score(y, p),
                pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                log_loss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)), ece=ece(y, p))


def fit_all(Xtr, Xca, ytr, yca, seed):
    out = {}
    for name, m in base_models(seed).items():
        m.fit(Xtr, ytr)
        out[(name, "raw")] = m
        for meth in ("sigmoid", "isotonic"):
            c = CalibratedClassifierCV(FrozenEstimator(m), method=meth)
            c.fit(Xca, yca)
            out[(name, meth)] = c
    return out


def uncertainty(p):
    # distance of the calibrated probability from a confident decision
    return 1.0 - np.abs(2 * p - 1)  # 0 = certain, 1 = maximally uncertain


def selective(y, p, coverage):
    """Automate the `coverage` share of most-confident cases, defer the rest."""
    n = len(y)
    u = uncertainty(p)
    order = np.argsort(u, kind="stable")
    k = int(round(coverage * n))
    auto = np.zeros(n, bool); auto[order[:k]] = True
    err = (p >= 0.5).astype(int) != y
    sel_acc = 1 - err[auto].mean() if k else np.nan
    captured = err[~auto].sum() / max(err.sum(), 1)
    return dict(coverage=coverage, deferral=1 - coverage, selective_accuracy=sel_acc,
                selective_risk=1 - sel_acc, full_risk=err.mean(),
                error_capture=captured, lift=captured / max(1 - coverage, 1e-9)), auto, err


def random_deferral_test(err, deferral, trials, rng):
    n = len(err); k = int(round(deferral * n)); tot = err.sum()
    caps = np.array([err[rng.choice(n, k, replace=False)].sum() / tot for _ in range(trials)])
    return caps


def graphical_abstract(t2, summary, a):
    """Code-generated graphical abstract (1328 x 531 px minimum, Elsevier spec)."""
    r = t2.set_index("coverage").loc[a.coverage]
    fig = plt.figure(figsize=(13.28, 5.31)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#333333", lw=1.2, zorder=1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=12.5, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 2.5, body, ha="center", va="center", fontsize=10.5, linespacing=1.5)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=1.8, color="#333333"))
    ax.text(50, 38, "UCDI: Uncertainty-Calibrated Decision Intelligence for selective AI automation",
            ha="center", va="top", fontsize=15, weight="bold")
    d = summary["data"]
    box(1, 9, 17, 23, "Enterprise data", f"Credit-default records\nn = {d['n']:,}, {d['features']} predictors\n60/20/20 train/\ncalibration/test", "#e8eef7")
    box(22, 9, 17, 23, "Knowledge layer", "LogReg, RF, GB\nraw / sigmoid /\nisotonic calibration\nmodel chosen on\ncalibration split", "#fdf0e6")
    box(43, 9, 17, 23, "Uncertainty gate", "u(x) = 1 - |2p(x) - 1|\nrank cases by u\nautomate lowest-u\ncoverage c = 0.80", "#eef6ee")
    box(65, 22, 33, 11, "Automate 80% of decisions", f"selective accuracy {r.selective_accuracy:.3f}  (vs {1-r.full_risk:.3f} full)", "#dff0df")
    box(65, 6, 33, 13, "Defer 20% to human review", f"captures {100*r.error_capture:.1f}% of all model errors\n{r.lift:.2f}x random deferral (p {'< 0.001' if r.perm_p < 0.001 else '= %.3f' % r.perm_p})", "#fbe3e3")
    arrow(18, 20.5, 22, 20.5); arrow(39, 20.5, 43, 20.5); arrow(60, 22, 65, 27.5); arrow(60, 19, 65, 12.5)
    s = summary["seeds"]
    ax.text(50, 2.2, f"Robust over {a.seeds} random splits: selective accuracy {s['sel_acc_mean']:.3f} +/- {s['sel_acc_sd']:.3f}, "
            f"error-capture lift {s['lift_mean']:.2f}x +/- {s['lift_sd']:.2f}", ha="center", fontsize=11, style="italic")
    fig.savefig(f"{a.fig}/graphical_abstract.png", dpi=200); plt.close(fig)


def main(a):
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.fig, exist_ok=True)
    X, y = load(a.data)
    Xtr, Xca, Xte, ytr, yca, yte = split(X, y, a.seed)
    info = dict(n=len(y), features=X.shape[1], prevalence=float(y.mean()),
                n_train=len(ytr), n_cal=len(yca), n_test=len(yte),
                prev_train=float(ytr.mean()), prev_cal=float(yca.mean()), prev_test=float(yte.mean()))

    models = fit_all(Xtr, Xca, ytr, yca, a.seed)
    probs = {k: m.predict_proba(Xte)[:, 1] for k, m in models.items()}
    t1 = pd.DataFrame([dict(model=k[0], calibration=k[1], **metrics(yte, p),
                            capture_at_cov=selective(yte, p, a.coverage)[0]["error_capture"],
                            sel_acc_at_cov=selective(yte, p, a.coverage)[0]["selective_accuracy"])
                       for k, p in probs.items()])
    t1.to_csv(f"{a.out}/table1_models.csv", index=False)

    # choose decision model on the CALIBRATION split (no test leakage): lowest Brier
    ca_probs = {k: m.predict_proba(Xca)[:, 1] for k, m in models.items() if k[1] == "raw"}
    best_base = min(ca_probs, key=lambda k: brier_score_loss(yca, ca_probs[k]))[0]
    key = (best_base, "isotonic")
    p = probs[key]

    rng = np.random.default_rng(a.seed)
    rows = []
    for cov in (1.0, 0.95, 0.90, 0.85, a.coverage, 0.70, 0.60, 0.50):
        r, _, err = selective(yte, p, cov)
        if cov < 1.0:
            caps = random_deferral_test(err, 1 - cov, a.trials, rng)
            r["random_capture_mean"] = caps.mean()
            r["perm_p"] = (1 + (caps >= r["error_capture"]).sum()) / (a.trials + 1)
        rows.append(r)
    t2 = pd.DataFrame(rows).drop_duplicates("coverage").sort_values("coverage", ascending=False)
    t2.to_csv(f"{a.out}/table2_selective.csv", index=False)

    # robustness across seeds
    srows = []
    for s in range(a.seeds):
        Xtr_, Xca_, Xte_, ytr_, yca_, yte_ = split(X, y, 100 + s)
        bm = base_models(100 + s)[best_base].fit(Xtr_, ytr_)
        c = CalibratedClassifierCV(FrozenEstimator(bm), method="isotonic").fit(Xca_, yca_)
        r, _, _ = selective(yte_, c.predict_proba(Xte_)[:, 1], a.coverage)
        srows.append(dict(seed=100 + s, **r))
    sd = pd.DataFrame(srows); sd.to_csv(f"{a.out}/seeds.csv", index=False)

    summary = dict(data=info, decision_model=f"{key[0]}+{key[1]}",
                   seeds=dict(sel_acc_mean=sd.selective_accuracy.mean(), sel_acc_sd=sd.selective_accuracy.std(),
                              lift_mean=sd.lift.mean(), lift_sd=sd.lift.std(),
                              capture_mean=sd.error_capture.mean()))
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=2, default=float)

    # ---------- Figure 1: reliability diagrams ----------
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, name in zip(axes, ["LogReg", "RandomForest", "GradBoost"]):
        ax.plot([0, 1], [0, 1], ls="--", c="0.6", lw=1, label="Perfect calibration")
        for meth, col, mk in (("raw", "#1f4e79", "o"), ("sigmoid", "#c55a11", "s"), ("isotonic", "#2e7d32", "^")):
            fp, mp = calibration_curve(yte, probs[(name, meth)], n_bins=10, strategy="quantile")
            e = t1.query("model==@name and calibration==@meth").ece.iloc[0]
            ax.plot(mp, fp, marker=mk, ms=4, lw=1.3, c=col, label=f"{meth} (ECE={e:.3f})")
        ax.set_title(name); ax.set_xlabel("Mean predicted probability"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed default frequency")
    fig.tight_layout(); fig.savefig(f"{a.fig}/fig1_reliability.png", dpi=600); plt.close(fig)

    # ---------- Figure 2: risk-coverage and error capture ----------
    covs = np.linspace(0.3, 1.0, 71)
    curves = {k: [selective(yte, probs[k], c)[0] for c in covs] for k in [(n, "isotonic") for n in ("LogReg", "RandomForest", "GradBoost")]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = {"LogReg": "#1f4e79", "RandomForest": "#c55a11", "GradBoost": "#2e7d32"}
    for k, rs in curves.items():
        a1.plot(covs, [r["selective_risk"] for r in rs], c=cols[k[0]], lw=1.6, label=f"{k[0]} (isotonic)")
        a2.plot(1 - covs, [r["error_capture"] for r in rs], c=cols[k[0]], lw=1.6, label=f"{k[0]} (isotonic)")
    a2.plot([0, .7], [0, .7], ls="--", c="0.5", lw=1, label="Random deferral")
    a1.axvline(a.coverage, c="k", ls=":", lw=1); a2.axvline(1 - a.coverage, c="k", ls=":", lw=1)
    a1.set_xlabel("Automation coverage"); a1.set_ylabel("Selective risk (error rate on automated cases)")
    a2.set_xlabel("Share of cases deferred to human review"); a2.set_ylabel("Share of all model errors captured")
    a1.set_title("(a) Risk-coverage"); a2.set_title("(b) Error concentration in deferred cases")
    for ax in (a1, a2): ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{a.fig}/fig2_risk_coverage.png", dpi=600); plt.close(fig)
    graphical_abstract(t2, summary, a)
    print(json.dumps(summary, indent=2, default=float)); print(t1.round(4)); print(t2.round(4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/credit_default_taiwan.csv")
    ap.add_argument("--out", default="results"); ap.add_argument("--fig", default="figures")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--coverage", type=float, default=0.80); ap.add_argument("--trials", type=int, default=1000)
    main(ap.parse_args())
