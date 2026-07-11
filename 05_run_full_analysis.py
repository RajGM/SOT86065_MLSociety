"""
MST Incremental Prompt Decomposition — Full Unified Analysis
==============================================================
Merges all 45 analyses: Base (1-12), Extended (E1-E10),
Distribution (D1-D12), Advanced (A1-A11).

All figures → figures_all/
All console stats → stdout (redirect to results.txt if desired)

Usage:
    cd F:\ML&Society\analysis
    pip install pandas matplotlib seaborn scipy numpy
    python run_full_analysis.py
    python run_full_analysis.py > results.txt 2>&1   # save all output
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from itertools import combinations

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "mst_audit_grok-imagine-image.csv")
OUT_DIR  = os.path.dirname(__file__)
FIG_DIR  = os.path.join(OUT_DIR, "figures_all")
os.makedirs(FIG_DIR, exist_ok=True)

MST_COL  = "MST_value_p1"
CONF_COL = "confidence_p1"

HIGH_PRESTIGE = ["surgeon", "lawyer", "architect", "airline pilot", "university professor",
                 "diplomat", "investment banker", "judge", "software engineer", "dentist"]
MID_PRESTIGE  = ["teacher", "nurse", "electrician", "police officer", "journalist",
                 "accountant", "pharmacist", "social worker", "chef", "librarian"]
LOW_PRESTIGE  = ["cleaner", "domestic worker", "street vendor", "rickshaw driver",
                 "construction laborer", "waste collector", "farm worker", "security guard",
                 "factory worker", "maid"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})
sns.set_style("whitegrid")
PALETTE = sns.color_palette("viridis", 10)

BINS = np.arange(0.5, 11.5, 1)
BIN_CENTERS = np.arange(1, 11)


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

def load_data():
    df = pd.read_csv(CSV_PATH)
    df[MST_COL]  = pd.to_numeric(df[MST_COL], errors="coerce")
    df[CONF_COL] = pd.to_numeric(df[CONF_COL], errors="coerce")
    df = df.dropna(subset=[MST_COL])
    df[MST_COL] = df[MST_COL].astype(int)

    def prestige(occ):
        if occ in HIGH_PRESTIGE: return "High"
        if occ in MID_PRESTIGE:  return "Mid"
        if occ in LOW_PRESTIGE:  return "Low"
        return ""
    df["prestige"] = df["occupation"].fillna("").apply(prestige)
    return df


# ═══════════════════════════════════════════════════════════════
# STATISTICAL HELPERS
# ═══════════════════════════════════════════════════════════════

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    if pooled == 0: return 0.0
    return (np.mean(a) - np.mean(b)) / pooled

def bootstrap_ci(data, stat_fn=np.mean, n_boot=5000, ci=0.95):
    rng = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1-ci)/2*100)
    hi = np.percentile(boots, (1+ci)/2*100)
    return lo, hi

def pct_dist(values):
    counts = np.zeros(10)
    for v in values:
        if 1 <= v <= 10:
            counts[int(v) - 1] += 1
    total = counts.sum()
    return (counts / total * 100) if total > 0 else counts

def print_dist(values, label=""):
    pcts = pct_dist(values)
    parts = [f"MST{i+1}:{pcts[i]:5.1f}%" for i in range(10) if pcts[i] > 0]
    print(f"  {label:30s} n={len(values):5d}  " + "  ".join(parts))

def ks_report(a, b, label_a, label_b):
    if len(a) < 2 or len(b) < 2: return
    stat, p = stats.ks_2samp(a, b)
    print(f"    KS test ({label_a} vs {label_b}): D={stat:.3f}, p={p:.2e}")
    return stat, p

def cliffs_delta_fast(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    return np.sign(a[:, None] - b[None, :]).mean()

def section(prefix, num, title):
    print(f"\n{'='*60}")
    print(f"{prefix}{num}. {title}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════
#  BASE ANALYSES (1–12)
# ═══════════════════════════════════════════════════════════════

def analysis_01_descriptive_overview(df):
    section("", 1, "DESCRIPTIVE OVERVIEW")
    print(f"Total images audited: {len(df)}")
    print(f"MST range: {df[MST_COL].min()} – {df[MST_COL].max()}")
    print(f"MST mean: {df[MST_COL].mean():.2f} ± {df[MST_COL].std():.2f}")
    print(f"MST median: {df[MST_COL].median():.1f}")
    print(f"Mean confidence: {df[CONF_COL].mean():.3f}")
    print("\nMST distribution:")
    for val in sorted(df[MST_COL].unique()):
        n = (df[MST_COL] == val).sum()
        pct = n / len(df) * 100
        print(f"  MST {val}: {n:5d} ({pct:5.1f}%)")
    print("\nBy section:")
    for sec, grp in df.groupby("section"):
        print(f"  {sec:20s}: n={len(grp):5d}, mean={grp[MST_COL].mean():.2f}, sd={grp[MST_COL].std():.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[MST_COL].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Monk Skin Tone (MST)")
    ax.set_ylabel("Count")
    ax.set_title("Overall MST Distribution (Grok)")
    ax.set_xticks(range(1, 11))
    for i, v in zip(counts.index, counts.values):
        ax.text(i, v + 10, str(v), ha="center", fontsize=9)
    fig.savefig(os.path.join(FIG_DIR, "01_mst_histogram.png"))
    plt.close(fig)


def analysis_02_incremental_layers(df):
    section("", 2, "INCREMENTAL LAYER ANALYSIS")
    ltrack = df[df["section"] == "L-track"]
    layers = ["L0", "L1", "L2", "L3", "L4", "L5"]
    print("L-track (gendered) — mean MST per layer:")
    layer_means = {}
    for layer in layers:
        sub = ltrack[ltrack["layer_id"] == layer]
        if len(sub) == 0: continue
        m = sub[MST_COL].mean(); s = sub[MST_COL].std()
        layer_means[layer] = m
        delta = ""
        prev = layers[layers.index(layer) - 1] if layers.index(layer) > 0 else None
        if prev and prev in layer_means:
            d = m - layer_means[prev]; delta = f" (Δ = {d:+.3f})"
        print(f"  {layer}: mean={m:.3f}, sd={s:.3f}, n={len(sub)}{delta}")

    ntrack = df[df["section"] == "N-track"]
    nlayers = ["N2", "N3", "N4", "N5"]
    l0_mean = ltrack[ltrack["layer_id"] == "L0"][MST_COL].mean()
    print(f"\nN-track (neutral) — mean MST per layer:")
    print(f"  N0 (=L0): mean={l0_mean:.3f}")
    for layer in nlayers:
        sub = ntrack[ntrack["layer_id"] == layer]
        if len(sub) == 0: continue
        print(f"  {layer}: mean={sub[MST_COL].mean():.3f}, sd={sub[MST_COL].std():.3f}, n={len(sub)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    l_data = []
    for layer in layers:
        sub = ltrack[ltrack["layer_id"] == layer]
        if len(sub) > 0:
            l_data.append({"layer": layer, "mean": sub[MST_COL].mean(),
                           "ci_lo": bootstrap_ci(sub[MST_COL].values)[0],
                           "ci_hi": bootstrap_ci(sub[MST_COL].values)[1]})
    ldf = pd.DataFrame(l_data)
    ax.errorbar(ldf["layer"], ldf["mean"],
                yerr=[ldf["mean"]-ldf["ci_lo"], ldf["ci_hi"]-ldf["mean"]],
                marker="o", capsize=5, linewidth=2, color="steelblue")
    ax.set_title("L-track: MST by Layer (Gendered)"); ax.set_xlabel("Layer"); ax.set_ylabel("Mean MST"); ax.set_ylim(3.5, 7.5)

    ax = axes[1]
    n_all_layers = ["N0"] + nlayers
    n_data = []
    for layer in n_all_layers:
        sub = ltrack[ltrack["layer_id"] == "L0"] if layer == "N0" else ntrack[ntrack["layer_id"] == layer]
        if len(sub) > 0:
            n_data.append({"layer": layer, "mean": sub[MST_COL].mean(),
                           "ci_lo": bootstrap_ci(sub[MST_COL].values)[0],
                           "ci_hi": bootstrap_ci(sub[MST_COL].values)[1]})
    ndf = pd.DataFrame(n_data)
    ax.errorbar(ndf["layer"], ndf["mean"],
                yerr=[ndf["mean"]-ndf["ci_lo"], ndf["ci_hi"]-ndf["mean"]],
                marker="s", capsize=5, linewidth=2, color="darkorange")
    ax.set_title("N-track: MST by Layer (Neutral)"); ax.set_xlabel("Layer"); ax.set_ylabel("Mean MST"); ax.set_ylim(3.5, 7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "02_layer_progression.png")); plt.close(fig)


def analysis_03_single_cue_effects(df):
    section("", 3, "SINGLE-CUE EFFECTS")
    baseline = df[(df["section"] == "L-track") & (df["layer_id"] == "L0")][MST_COL].values
    baseline_mean = baseline.mean()
    print(f"Baseline (L0): mean={baseline_mean:.3f}, n={len(baseline)}")
    single = df[df["section"] == "Single-cue"]
    cue_map = {"S1": ("Gender", "gender"), "S2": ("Occupation", "occupation"),
               "S3": ("Income", "income"), "S4": ("Setting", "setting"), "S5": ("Neighborhood", "neighborhood")}
    fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.flatten()
    for idx, (sid, (cue_name, col)) in enumerate(cue_map.items()):
        sub = single[single["layer_id"] == sid]
        print(f"\n  {sid} ({cue_name}):")
        ax = axes[idx]; vals_list = []; labels = []
        for val, grp in sub.groupby(col):
            m = grp[MST_COL].mean(); d = cohens_d(grp[MST_COL].values, baseline)
            print(f"    {val}: mean={m:.3f}, n={len(grp)}, d={d:+.3f} vs baseline")
            vals_list.append(grp[MST_COL].values); labels.append(val)
        if vals_list:
            ax.violinplot(vals_list, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(labels)+1)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_title(f"{cue_name} (Single-cue)"); ax.set_ylabel("MST")
            ax.axhline(baseline_mean, color="red", linestyle="--", alpha=0.7, label="Baseline"); ax.legend(fontsize=8)
    axes[5].set_visible(False)
    fig.suptitle("Single-Cue MST Distributions vs Baseline", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "03_single_cue_effects.png")); plt.close(fig)


def analysis_04_income_effect(df):
    section("", 4, "INCOME EFFECT")
    has_income = df[df["income"].isin(["high", "low"])]
    high = has_income[has_income["income"] == "high"][MST_COL].values
    low = has_income[has_income["income"] == "low"][MST_COL].values
    d = cohens_d(low, high); t_stat, p_val = stats.mannwhitneyu(high, low, alternative="two-sided")
    print(f"  High income: mean={high.mean():.3f}, sd={high.std():.3f}, n={len(high)}")
    print(f"  Low income:  mean={low.mean():.3f}, sd={low.std():.3f}, n={len(low)}")
    print(f"  Cohen's d (low vs high): {d:+.3f}")
    print(f"  Mann-Whitney U p-value:  {p_val:.2e}")
    ltrack = df[df["section"] == "L-track"]
    print("\n  Income effect by L-track layer:")
    for layer in ["L3", "L4", "L5"]:
        sub = ltrack[ltrack["layer_id"] == layer]
        h = sub[sub["income"] == "high"][MST_COL].values; l = sub[sub["income"] == "low"][MST_COL].values
        if len(h) > 1 and len(l) > 1:
            print(f"    {layer}: high={h.mean():.3f}, low={l.mean():.3f}, d={cohens_d(l, h):+.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.violinplot(data=has_income, x="income", y=MST_COL, order=["high", "low"],
                   palette=["#4CAF50", "#F44336"], ax=axes[0], inner="box")
    axes[0].set_title(f"Income Effect on MST (d={cohens_d(low, high):+.2f})"); axes[0].set_ylabel("MST")
    income_by_layer = []
    for layer in ["L3", "L4", "L5"]:
        sub = ltrack[ltrack["layer_id"] == layer]
        for inc in ["high", "low"]:
            s = sub[sub["income"] == inc]
            if len(s) > 0: income_by_layer.append({"layer": layer, "income": inc, "mean_mst": s[MST_COL].mean()})
    idf = pd.DataFrame(income_by_layer)
    if not idf.empty:
        sns.barplot(data=idf, x="layer", y="mean_mst", hue="income", palette=["#4CAF50", "#F44336"], ax=axes[1])
        axes[1].set_title("Income Effect by Layer"); axes[1].set_ylabel("Mean MST"); axes[1].set_ylim(3.5, 7.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "04_income_effect.png")); plt.close(fig)


def analysis_05_occupation(df):
    section("", 5, "OCCUPATION ANALYSIS")
    has_occ = df[df["occupation"] != ""]
    print("  By prestige band:")
    for band in ["High", "Mid", "Low"]:
        sub = has_occ[has_occ["prestige"] == band]
        if len(sub) > 0: print(f"    {band}: mean={sub[MST_COL].mean():.3f}, sd={sub[MST_COL].std():.3f}, n={len(sub)}")
    high_vals = has_occ[has_occ["prestige"] == "High"][MST_COL].values
    low_vals = has_occ[has_occ["prestige"] == "Low"][MST_COL].values
    if len(high_vals) > 1 and len(low_vals) > 1:
        print(f"    Cohen's d (Low vs High prestige): {cohens_d(low_vals, high_vals):+.3f}")
    groups = [has_occ[has_occ["prestige"] == b][MST_COL].values for b in ["High", "Mid", "Low"]]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        grand_mean = has_occ[MST_COL].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
        ss_total = sum((has_occ[MST_COL].values - grand_mean)**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        print(f"    ANOVA: F={f_stat:.2f}, p={p_val:.2e}, η²={eta_sq:.4f}")
    occ_means = has_occ.groupby("occupation")[MST_COL].mean().sort_values()
    print("\n  Lightest 5 occupations:")
    for occ, m in occ_means.head(5).items(): print(f"    {occ}: {m:.3f}")
    print("  Darkest 5 occupations:")
    for occ, m in occ_means.tail(5).items(): print(f"    {occ}: {m:.3f}")
    fig, ax = plt.subplots(figsize=(10, 8))
    occ_order = occ_means.index.tolist()
    ax.barh(range(len(occ_order)), [occ_means[o] for o in occ_order],
            color=[{"High": "#4CAF50", "Mid": "#FFC107", "Low": "#F44336"}.get(
                "High" if o in HIGH_PRESTIGE else "Mid" if o in MID_PRESTIGE else "Low", "#999") for o in occ_order],
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(occ_order))); ax.set_yticklabels(occ_order, fontsize=8)
    ax.set_xlabel("Mean MST"); ax.set_title("Mean MST by Occupation (Green=High, Yellow=Mid, Red=Low prestige)")
    ax.axvline(has_occ[MST_COL].mean(), color="black", linestyle="--", alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "05_occupation_mst.png")); plt.close(fig)


def analysis_06_gender(df):
    section("", 6, "GENDER EFFECT")
    has_gender = df[df["gender"].isin(["man", "woman"])]
    man = has_gender[has_gender["gender"] == "man"][MST_COL].values
    woman = has_gender[has_gender["gender"] == "woman"][MST_COL].values
    neutral = df[df["section"] == "N-track"][MST_COL].values
    d_mw = cohens_d(woman, man)
    print(f"  Man:     mean={man.mean():.3f}, sd={man.std():.3f}, n={len(man)}")
    print(f"  Woman:   mean={woman.mean():.3f}, sd={woman.std():.3f}, n={len(woman)}")
    print(f"  Neutral: mean={neutral.mean():.3f}, sd={neutral.std():.3f}, n={len(neutral)}")
    print(f"  Cohen's d (woman vs man): {d_mw:+.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    for label, vals in [("Man", man), ("Woman", woman), ("Neutral", neutral)]:
        for v in vals: data.append({"Gender": label, "MST": v})
    gdf = pd.DataFrame(data)
    sns.violinplot(data=gdf, x="Gender", y="MST", order=["Man", "Woman", "Neutral"],
                   palette=["#2196F3", "#E91E63", "#9E9E9E"], ax=ax, inner="box")
    ax.set_title(f"Gender Effect on MST (d={d_mw:+.2f} woman vs man)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "06_gender_effect.png")); plt.close(fig)


def analysis_07_geography(df):
    section("", 7, "SETTING & NEIGHBORHOOD EFFECT")
    has_setting = df[df["setting"].isin(["village", "city"])]
    village = has_setting[has_setting["setting"] == "village"][MST_COL].values
    city = has_setting[has_setting["setting"] == "city"][MST_COL].values
    d_vc = cohens_d(village, city)
    print(f"  Village: mean={village.mean():.3f}, n={len(village)}")
    print(f"  City:    mean={city.mean():.3f}, n={len(city)}")
    print(f"  Cohen's d (village vs city): {d_vc:+.3f}")
    has_nbr = df[df["neighborhood"].isin(["gated community", "suburb", "slum"])]
    print("\n  Neighborhood:")
    for nbr in ["gated community", "suburb", "slum"]:
        sub = has_nbr[has_nbr["neighborhood"] == nbr][MST_COL].values
        print(f"    {nbr}: mean={sub.mean():.3f}, n={len(sub)}")
    gated = has_nbr[has_nbr["neighborhood"] == "gated community"][MST_COL].values
    slum = has_nbr[has_nbr["neighborhood"] == "slum"][MST_COL].values
    d_gs = cohens_d(slum, gated)
    print(f"    Cohen's d (slum vs gated community): {d_gs:+.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.violinplot(data=has_setting, x="setting", y=MST_COL, order=["city", "village"],
                   palette=["#42A5F5", "#8D6E63"], ax=axes[0], inner="box")
    axes[0].set_title(f"Setting Effect (d={d_vc:+.2f})"); axes[0].set_ylabel("MST")
    sns.violinplot(data=has_nbr, x="neighborhood", y=MST_COL, order=["gated community", "suburb", "slum"],
                   palette=["#4CAF50", "#FFC107", "#F44336"], ax=axes[1], inner="box")
    axes[1].set_title(f"Neighborhood Effect (d={d_gs:+.2f} slum vs gated)"); axes[1].set_ylabel("MST")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "07_geography_effect.png")); plt.close(fig)


def analysis_08_interactions(df):
    section("", 8, "TWO-CUE INTERACTION EFFECTS")
    baseline = df[(df["section"] == "L-track") & (df["layer_id"] == "L0")][MST_COL].mean()
    single = df[df["section"] == "Single-cue"]; twocue = df[df["section"] == "Two-cue"]
    single_means = {}
    for sid, col in [("S1","gender"),("S2","occupation"),("S3","income"),("S4","setting"),("S5","neighborhood")]:
        sub = single[single["layer_id"] == sid]
        for val, grp in sub.groupby(col): single_means[(col, val)] = grp[MST_COL].mean()
    print(f"  Baseline (L0): {baseline:.3f}")
    print(f"  Interaction = Observed - (SingleA + SingleB - Baseline)")
    interactions = []
    x_configs = {"X1":("gender","income"),"X2":("gender","occupation"),"X3":("gender","setting"),
                 "X4":("gender","neighborhood"),"X5":("occupation","income"),"X8":("income","setting"),
                 "X9":("income","neighborhood"),"X10":("setting","neighborhood")}
    for xid, (col_a, col_b) in x_configs.items():
        sub = twocue[twocue["layer_id"] == xid]
        if len(sub) == 0: continue
        observed = sub[MST_COL].mean()
        expected_parts = []
        for a in sub[col_a].unique():
            for b in sub[col_b].unique():
                sa = single_means.get((col_a, a), baseline); sb = single_means.get((col_b, b), baseline)
                expected_parts.append(sa + sb - baseline)
        expected = np.mean(expected_parts) if expected_parts else baseline
        interaction = observed - expected
        interactions.append({"pair": xid, "cues": f"{col_a}×{col_b}", "observed": observed, "expected": expected, "interaction": interaction})
        print(f"  {xid} ({col_a}×{col_b}): obs={observed:.3f}, exp={expected:.3f}, interaction={interaction:+.3f}")
    if interactions:
        idf = pd.DataFrame(interactions); fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#F44336" if x < 0 else "#4CAF50" for x in idf["interaction"]]
        ax.bar(idf["cues"], idf["interaction"], color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Interaction Effect (MST units)")
        ax.set_title("Two-Cue Interaction Effects\n(Positive = darker than expected, Negative = lighter)")
        ax.set_xticklabels(idf["cues"], rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "08_interaction_effects.png")); plt.close(fig)


def analysis_09_multi_cue(df):
    section("", 9, "MULTI-CUE STACKS")
    multi = df[df["section"] == "Multi-cue"]
    baseline = df[(df["section"] == "L-track") & (df["layer_id"] == "L0")][MST_COL].mean()
    print(f"  Baseline (L0): {baseline:.3f}")
    for cid in ["C1","C2","C3","C4","C5","C6","C7","C8"]:
        sub = multi[multi["layer_id"] == cid]
        if len(sub) == 0: continue
        m = sub[MST_COL].mean(); cues = sub["cues"].iloc[0]
        print(f"  {cid} ({cues}): MST={m:.1f}, Δ={m - baseline:+.2f}")
    for cid in ["C9","C10"]:
        sub = multi[multi["layer_id"] == cid]
        if len(sub) > 0:
            print(f"\n  {cid} gradient (n={len(sub)}): mean={sub[MST_COL].mean():.3f}, range={sub[MST_COL].min()}–{sub[MST_COL].max()}")
    fig, ax = plt.subplots(figsize=(10, 5)); stack_data = []
    for cid in ["C1","C2","C3","C4","C5","C6","C7","C8"]:
        sub = multi[multi["layer_id"] == cid]
        if len(sub) > 0: stack_data.append({"config": cid, "MST": sub[MST_COL].mean(), "cues": sub["cues"].iloc[0][:30]})
    if stack_data:
        sdf = pd.DataFrame(stack_data)
        colors = ["#F44336" if m > baseline else "#4CAF50" for m in sdf["MST"]]
        ax.barh(range(len(sdf)), sdf["MST"], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(sdf))); ax.set_yticklabels([f"{r['config']}: {r['cues']}" for _, r in sdf.iterrows()], fontsize=8)
        ax.axvline(baseline, color="black", linestyle="--", label=f"Baseline ({baseline:.1f})")
        ax.set_xlabel("MST"); ax.set_title("Multi-Cue Stacks: Aligned vs Contradiction"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "09_multi_cue_stacks.png")); plt.close(fig)


def analysis_10_order_effect(df):
    section("", 10, "ORDER-EFFECT ANALYSIS")
    order = df[df["section"] == "Order-effect"]
    if len(order) == 0: print("  No order-effect data found."); return
    order_occs = order["occupation"].unique()
    l5 = df[(df["section"] == "L-track") & (df["layer_id"] == "L5")]
    l5_subset = l5[l5["occupation"].isin(order_occs)]
    print(f"  Baseline (L5, same 6 occupations): mean={l5_subset[MST_COL].mean():.3f}, n={len(l5_subset)}")
    orderings = order["layer_id"].unique()
    for oid in sorted(orderings):
        sub = order[order["layer_id"] == oid]
        m = sub[MST_COL].mean(); s = sub[MST_COL].std(); d = cohens_d(sub[MST_COL].values, l5_subset[MST_COL].values)
        cues = sub["cues"].iloc[0]
        print(f"  {oid} ({cues}): mean={m:.3f}, sd={s:.3f}, n={len(sub)}, d={d:+.3f}")
    groups = [order[order["layer_id"] == oid][MST_COL].values for oid in sorted(orderings)]
    groups.append(l5_subset[MST_COL].values)
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n  ANOVA (4 orderings + baseline): F={f_stat:.3f}, p={p_val:.4f}")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = [{"ordering": "Baseline\n(G→O→I→S→N)", "MST": v} for v in l5_subset[MST_COL].values]
    for oid in sorted(orderings):
        sub = order[order["layer_id"] == oid]; cues = sub["cues"].iloc[0]
        for v in sub[MST_COL].values: plot_data.append({"ordering": f"{oid}\n({cues})", "MST": v})
    odf = pd.DataFrame(plot_data)
    sns.boxplot(data=odf, x="ordering", y="MST", ax=ax, palette="Set2")
    ax.set_title(f"Order Effect on MST (ANOVA p={p_val:.4f})"); ax.set_ylabel("MST"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "10_order_effect.png")); plt.close(fig)


def analysis_11_income_prestige_heatmap(df):
    section("", 11, "INCOME × OCCUPATION PRESTIGE INTERACTION")
    has_both = df[(df["income"].isin(["high", "low"])) & (df["prestige"] != "")]
    pivot = has_both.groupby(["prestige", "income"])[MST_COL].mean().unstack()
    pivot = pivot.reindex(index=["High", "Mid", "Low"], columns=["high", "low"])
    print(f"\n{pivot.to_string()}")
    for band in ["High", "Mid", "Low"]:
        sub = has_both[has_both["prestige"] == band]
        h = sub[sub["income"] == "high"][MST_COL].values; l = sub[sub["income"] == "low"][MST_COL].values
        if len(h) > 1 and len(l) > 1: print(f"  Income effect within {band} prestige: d={cohens_d(l, h):+.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn_r", ax=ax, vmin=4, vmax=7, linewidths=1, cbar_kws={"label": "Mean MST"})
    ax.set_title("Mean MST: Income × Occupation Prestige"); ax.set_ylabel("Prestige Band"); ax.set_xlabel("Income")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "11_income_prestige_heatmap.png")); plt.close(fig)


def analysis_12_effect_size_summary(df):
    section("", 12, "EFFECT SIZE SUMMARY (Cohen's d)")
    baseline = df[(df["section"] == "L-track") & (df["layer_id"] == "L0")][MST_COL].values
    has_income = df[df["income"].isin(["high", "low"])]; has_gender = df[df["gender"].isin(["man", "woman"])]
    has_setting = df[df["setting"].isin(["village", "city"])]; has_nbr = df[df["neighborhood"].isin(["gated community", "slum"])]
    has_occ = df[df["prestige"].isin(["High", "Low"])]
    effects = []
    h = has_income[has_income["income"] == "high"][MST_COL].values; l = has_income[has_income["income"] == "low"][MST_COL].values
    effects.append(("Income (low vs high)", cohens_d(l, h), len(h)+len(l)))
    hp = has_occ[has_occ["prestige"] == "High"][MST_COL].values; lp = has_occ[has_occ["prestige"] == "Low"][MST_COL].values
    effects.append(("Occupation (low vs high prestige)", cohens_d(lp, hp), len(hp)+len(lp)))
    v = has_setting[has_setting["setting"] == "village"][MST_COL].values; c = has_setting[has_setting["setting"] == "city"][MST_COL].values
    effects.append(("Setting (village vs city)", cohens_d(v, c), len(v)+len(c)))
    s = has_nbr[has_nbr["neighborhood"] == "slum"][MST_COL].values; g = has_nbr[has_nbr["neighborhood"] == "gated community"][MST_COL].values
    effects.append(("Neighborhood (slum vs gated)", cohens_d(s, g), len(s)+len(g)))
    m = has_gender[has_gender["gender"] == "man"][MST_COL].values; w = has_gender[has_gender["gender"] == "woman"][MST_COL].values
    effects.append(("Gender (woman vs man)", cohens_d(w, m), len(m)+len(w)))
    print(f"\n  {'Comparison':<40} {'d':>8} {'n':>8}")
    print(f"  {'-'*56}")
    for name, d, n in sorted(effects, key=lambda x: abs(x[1]), reverse=True):
        size = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
        print(f"  {name:<40} {d:+8.3f} {n:>8}  ({size})")
    fig, ax = plt.subplots(figsize=(9, 5))
    effects_sorted = sorted(effects, key=lambda x: x[1])
    names = [e[0] for e in effects_sorted]; ds = [e[1] for e in effects_sorted]
    colors = ["#F44336" if d > 0 else "#4CAF50" for d in ds]
    ax.barh(range(len(names)), ds, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9); ax.axvline(0, color="black", linewidth=0.8)
    for t in [0.2, 0.5, 0.8]: ax.axvline(t, color="gray", linestyle=":", alpha=0.5); ax.axvline(-t, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (positive = darker)"); ax.set_title("Effect Size Summary: Which Cues Shift Skin Tone?")
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "12_effect_size_summary.png")); plt.close(fig)




# ═══════════════════════════════════════════════════════════════
#  EXTENDED ANALYSES (E1–E10)
# ═══════════════════════════════════════════════════════════════

def analysis_E1_ltrack_vs_ntrack(df):
    section("E", 1, "L-TRACK vs N-TRACK: GENDER MODERATION")
    ltrack = df[df["section"] == "L-track"]; ntrack = df[df["section"] == "N-track"]
    pairs = [("L2","N2"),("L3","N3"),("L4","N4"),("L5","N5")]
    results = []
    for l_layer, n_layer in pairs:
        l_data = ltrack[ltrack["layer_id"] == l_layer][MST_COL].values
        n_data = ntrack[ntrack["layer_id"] == n_layer][MST_COL].values
        if len(l_data) < 2 or len(n_data) < 2: continue
        d = cohens_d(l_data, n_data); u_stat, u_p = stats.mannwhitneyu(l_data, n_data, alternative="two-sided")
        print(f"  {l_layer} vs {n_layer}: L={np.mean(l_data):.3f}, N={np.mean(n_data):.3f}, Δ={np.mean(l_data)-np.mean(n_data):+.3f}, d={d:+.3f}, p={u_p:.4f}")
        results.append((l_layer, n_layer, np.mean(l_data), np.mean(n_data), d, u_p))
    if results:
        fig, ax = plt.subplots(figsize=(8, 5))
        layers = [r[0] for r in results]; l_means = [r[2] for r in results]; n_means = [r[3] for r in results]; x = np.arange(len(layers))
        ax.bar(x - 0.15, l_means, 0.3, label="L-track (gendered)", color="steelblue")
        ax.bar(x + 0.15, n_means, 0.3, label="N-track (neutral)", color="darkorange")
        for i, r in enumerate(results):
            sig = "***" if r[5]<0.001 else "**" if r[5]<0.01 else "*" if r[5]<0.05 else "ns"
            ax.text(i, max(r[2],r[3])+0.05, sig, ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_ylabel("Mean MST")
        ax.set_title("L-track (gendered) vs N-track (neutral) at Each Layer"); ax.legend(); ax.set_ylim(4.0, 6.5)
        fig.savefig(os.path.join(FIG_DIR, "E1_ltrack_vs_ntrack.png")); plt.close(fig)


def analysis_E2_x5_occupation_income(df):
    section("E", 2, "X5: OCCUPATION × INCOME — AMPLIFICATION?")
    x5 = df[(df["section"] == "Two-cue") & (df["layer_id"] == "X5")]
    if len(x5) == 0: print("  No X5 data."); return
    s3 = df[(df["section"] == "Single-cue") & (df["layer_id"] == "S3")]
    s3_high = s3[s3["income"]=="high"][MST_COL].values; s3_low = s3[s3["income"]=="low"][MST_COL].values
    if len(s3_high)>0 and len(s3_low)>0: print(f"  Pure income effect (S3): Δ = {np.mean(s3_low)-np.mean(s3_high):+.3f}")
    results = []
    for occ in sorted(x5["occupation"].dropna().unique()):
        occ_data = x5[x5["occupation"]==occ]
        high = occ_data[occ_data["income"]=="high"][MST_COL].values; low = occ_data[occ_data["income"]=="low"][MST_COL].values
        if len(high)>0 and len(low)>0:
            delta = np.mean(low)-np.mean(high); results.append((occ, np.mean(high), np.mean(low), delta))
    if results:
        results.sort(key=lambda r: r[3], reverse=True)
        print(f"\n  Income effect (low − high MST) per occupation:")
        for occ,h,l,d in results: print(f"    {occ:25s}: high={h:.1f}, low={l:.1f}, Δ={d:+.1f}")
        fig, ax = plt.subplots(figsize=(10, 8))
        occs=[r[0] for r in results]; deltas=[r[3] for r in results]
        colors=["crimson" if d>0 else "seagreen" for d in deltas]; y=np.arange(len(occs))
        ax.barh(y, deltas, color=colors); ax.set_yticks(y); ax.set_yticklabels(occs)
        ax.axvline(0, color="black", linewidth=0.8); ax.set_xlabel("Income effect on MST (low − high)")
        ax.set_title("X5: Income Effect per Occupation\n(positive = low income darkens)"); ax.invert_yaxis()
        fig.savefig(os.path.join(FIG_DIR, "E2_x5_income_by_occupation.png")); plt.close(fig)


def analysis_E3_x8_income_setting(df):
    section("E", 3, "X8: INCOME × SETTING COMPOUNDING")
    x8 = df[(df["section"]=="Two-cue")&(df["layer_id"]=="X8")]
    if len(x8)==0: print("  No X8 data."); return
    cells = {}
    for inc in ["high","low"]:
        for sett in ["city","village"]:
            sub = x8[(x8["income"]==inc)&(x8["setting"]==sett)][MST_COL].values; cells[(inc,sett)]=sub
            print(f"  {inc:5s} + {sett:7s}: mean={np.mean(sub):.3f}, n={len(sub)}")
    if len(cells.get(("low","village"),[]))>0 and len(cells.get(("high","city"),[]))>0:
        lv=cells[("low","village")]; hc=cells[("high","city")]; d=cohens_d(lv,hc)
        print(f"\n  Max compounding (low+village vs high+city): Δ={np.mean(lv)-np.mean(hc):+.3f}, d={d:+.3f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = pd.DataFrame([[np.mean(cells.get(("high","city"),[np.nan])),np.mean(cells.get(("high","village"),[np.nan]))],
                            [np.mean(cells.get(("low","city"),[np.nan])),np.mean(cells.get(("low","village"),[np.nan]))]],
                           index=["high income","low income"], columns=["city","village"])
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", vmin=4, vmax=7, ax=ax)
    ax.set_title("X8: Income × Setting Mean MST")
    fig.savefig(os.path.join(FIG_DIR, "E3_x8_income_setting.png")); plt.close(fig)


def analysis_E4_x9_income_neighborhood(df):
    section("E", 4, "X9: INCOME × NEIGHBORHOOD COMPOUNDING")
    x9 = df[(df["section"]=="Two-cue")&(df["layer_id"]=="X9")]
    if len(x9)==0: print("  No X9 data."); return
    cells = {}
    for inc in ["high","low"]:
        for nb in ["gated community","suburb","slum"]:
            sub = x9[(x9["income"]==inc)&(x9["neighborhood"]==nb)][MST_COL].values; cells[(inc,nb)]=sub
            print(f"  {inc:5s} + {nb:16s}: mean={np.mean(sub):.3f}, n={len(sub)}")
    if len(cells.get(("low","slum"),[]))>0 and len(cells.get(("high","gated community"),[]))>0:
        ls=cells[("low","slum")]; hg=cells[("high","gated community")]; d=cohens_d(ls,hg)
        print(f"\n  Max compounding (low+slum vs high+gated): Δ={np.mean(ls)-np.mean(hg):+.3f}, d={d:+.3f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    matrix = pd.DataFrame([[np.mean(cells.get(("high","gated community"),[np.nan])),np.mean(cells.get(("high","suburb"),[np.nan])),np.mean(cells.get(("high","slum"),[np.nan]))],
                            [np.mean(cells.get(("low","gated community"),[np.nan])),np.mean(cells.get(("low","suburb"),[np.nan])),np.mean(cells.get(("low","slum"),[np.nan]))]],
                           index=["high income","low income"], columns=["gated community","suburb","slum"])
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", vmin=4, vmax=7, ax=ax)
    ax.set_title("X9: Income × Neighborhood Mean MST")
    fig.savefig(os.path.join(FIG_DIR, "E4_x9_income_neighborhood.png")); plt.close(fig)


def analysis_E5_x10_setting_neighborhood(df):
    section("E", 5, "X10: SETTING × NEIGHBORHOOD — WHICH CUE DOMINATES?")
    x10 = df[(df["section"]=="Two-cue")&(df["layer_id"]=="X10")]
    if len(x10)==0: print("  No X10 data."); return
    cells = {}
    for sett in ["city","village"]:
        for nb in ["gated community","suburb","slum"]:
            sub = x10[(x10["setting"]==sett)&(x10["neighborhood"]==nb)][MST_COL].values; cells[(sett,nb)]=sub
            print(f"  {sett:7s} + {nb:16s}: mean={np.mean(sub):.3f}, n={len(sub)}")
    vg=cells.get(("village","gated community"),[]); cs=cells.get(("city","slum"),[])
    if len(vg)>0 and len(cs)>0:
        d=cohens_d(cs,vg)
        print(f"\n  Contradiction (city+slum vs village+gated): Δ={np.mean(cs)-np.mean(vg):+.3f}, d={d:+.3f}")
        print(f"  → {'neighborhood' if abs(np.mean(cs)-np.mean(vg))>0.1 else 'tie'} appears to dominate")
    fig, ax = plt.subplots(figsize=(6, 4))
    matrix = pd.DataFrame([[np.mean(cells.get(("city","gated community"),[np.nan])),np.mean(cells.get(("city","suburb"),[np.nan])),np.mean(cells.get(("city","slum"),[np.nan]))],
                            [np.mean(cells.get(("village","gated community"),[np.nan])),np.mean(cells.get(("village","suburb"),[np.nan])),np.mean(cells.get(("village","slum"),[np.nan]))]],
                           index=["city","village"], columns=["gated community","suburb","slum"])
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", vmin=4, vmax=7, ax=ax)
    ax.set_title("X10: Setting × Neighborhood Mean MST")
    fig.savefig(os.path.join(FIG_DIR, "E5_x10_setting_neighborhood.png")); plt.close(fig)


def analysis_E6_c5_vs_c6(df):
    section("E", 6, "C5 vs C6: OCCUPATION vs ALL OTHER SIGNALS")
    mc = df[df["section"]=="Multi-cue"]; baseline = df[df["layer_id"]=="L0"][MST_COL].values
    c5=mc[mc["layer_id"]=="C5"][MST_COL].values; c6=mc[mc["layer_id"]=="C6"][MST_COL].values
    c1=mc[mc["layer_id"]=="C1"][MST_COL].values; c2=mc[mc["layer_id"]=="C2"][MST_COL].values
    print(f"  L0 baseline: {np.mean(baseline):.1f}")
    if len(c2): print(f"  C2 (aligned light): {np.mean(c2):.1f}")
    if len(c1): print(f"  C1 (aligned dark):  {np.mean(c1):.1f}")
    if len(c5): print(f"  C5 (surgeon+low context): MST={np.mean(c5):.1f}")
    if len(c6): print(f"  C6 (cleaner+high context): MST={np.mean(c6):.1f}")
    if len(c5) and len(c6):
        print(f"  C5−C6 = {np.mean(c5)-np.mean(c6):+.1f}")
        print(f"  → {'Context overpower occupation' if np.mean(c5)>np.mean(c6) else 'Occupation overpowers context'}")
    fig, ax = plt.subplots(figsize=(7, 5)); labels=[]; vals=[]; colors_bar=[]
    for lbl,data,clr in [("C2\nAligned light",c2,"seagreen"),("C6\nCleaner+high",c6,"goldenrod"),
                          ("C5\nSurgeon+low",c5,"darkorange"),("C1\nAligned dark",c1,"crimson")]:
        if len(data): labels.append(lbl); vals.append(np.mean(data)); colors_bar.append(clr)
    if vals:
        x=np.arange(len(vals)); ax.bar(x, vals, color=colors_bar, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
        ax.axhline(np.mean(baseline), color="gray", linestyle="--", label=f"Baseline ({np.mean(baseline):.1f})")
        ax.set_ylabel("Mean MST"); ax.set_title("C5 vs C6: When Occupation Contradicts Context"); ax.legend(); ax.set_ylim(3, 8)
    fig.savefig(os.path.join(FIG_DIR, "E6_c5_vs_c6_contradiction.png")); plt.close(fig)


def analysis_E7_x2_gender_occupation(df):
    section("E", 7, "X2: GENDER × OCCUPATION STEREOTYPES")
    x2 = df[(df["section"]=="Two-cue")&(df["layer_id"]=="X2")]
    if len(x2)==0: print("  No X2 data."); return
    results = []
    for occ in sorted(x2["occupation"].dropna().unique()):
        occ_data=x2[x2["occupation"]==occ]
        man_mst=occ_data[occ_data["gender"]=="man"][MST_COL].values
        woman_mst=occ_data[occ_data["gender"]=="woman"][MST_COL].values
        if len(man_mst)>0 and len(woman_mst)>0:
            gap=np.mean(woman_mst)-np.mean(man_mst); results.append((occ, np.mean(man_mst), np.mean(woman_mst), gap))
    if results:
        results.sort(key=lambda r: r[3])
        print(f"  {'Occupation':25s}  {'Man':>5s}  {'Woman':>5s}  {'Gap':>6s}")
        print(f"  {'-'*50}")
        for occ,m,w,g in results: print(f"  {occ:25s}  {m:5.1f}  {w:5.1f}  {g:+5.1f}")
        fig, ax = plt.subplots(figsize=(10, 8)); occs=[r[0] for r in results]; gaps=[r[3] for r in results]
        colors=["steelblue" if g<0 else "crimson" for g in gaps]; y=np.arange(len(occs))
        ax.barh(y, gaps, color=colors); ax.set_yticks(y); ax.set_yticklabels(occs, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8); ax.set_xlabel("Gender gap (woman − man MST)")
        ax.set_title("X2: Gender × Occupation\n(blue=woman lighter, red=woman darker)"); ax.invert_yaxis()
        fig.savefig(os.path.join(FIG_DIR, "E7_x2_gender_occupation.png")); plt.close(fig)


def analysis_E8_x6_x7_occupation_geography(df):
    section("E", 8, "X6 & X7: OCCUPATION × GEOGRAPHY")
    for xid, geo_col, geo_vals, title in [
        ("X6","setting",["city","village"],"X6: Setting Effect per Occupation"),
        ("X7","neighborhood",["gated community","suburb","slum"],"X7: Neighborhood Effect per Occupation")]:
        xdata = df[(df["section"]=="Two-cue")&(df["layer_id"]==xid)]
        if len(xdata)==0: print(f"  No {xid} data."); continue
        print(f"\n  --- {xid}: Occupation × {geo_col.title()} ---")
        for band in ["High","Mid","Low"]:
            band_data = xdata[xdata["prestige"]==band]
            if len(band_data)==0: continue
            means = {}
            for gv in geo_vals:
                sub = band_data[band_data[geo_col]==gv][MST_COL].values; means[gv] = np.mean(sub) if len(sub)>0 else np.nan
            vals_str = ", ".join([f"{gv}={means[gv]:.2f}" for gv in geo_vals])
            print(f"    {band:5s} prestige: {vals_str}")
        fig, ax = plt.subplots(figsize=(6, 4))
        matrix_data = []
        for band in ["High","Mid","Low"]:
            row = []
            for gv in geo_vals:
                sub = xdata[(xdata["prestige"]==band)&(xdata[geo_col]==gv)][MST_COL].values
                row.append(np.mean(sub) if len(sub)>0 else np.nan)
            matrix_data.append(row)
        matrix = pd.DataFrame(matrix_data, index=["High","Mid","Low"], columns=geo_vals)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", vmin=4, vmax=7, ax=ax)
        ax.set_title(f"{title}\n(by Prestige Band)"); ax.set_ylabel("Occupation Prestige")
        fig.savefig(os.path.join(FIG_DIR, f"E8_{xid.lower()}_prestige_{geo_col}.png")); plt.close(fig)


def analysis_E9_per_occupation_effect(df):
    section("E", 9, "PER-OCCUPATION EFFECT SIZE vs BASELINE")
    occ_df = df[df["occupation"].notna() & (df["occupation"] != "")]
    grand_mean = occ_df[MST_COL].mean(); grand_std = occ_df[MST_COL].std()
    results = []
    for occ in sorted(occ_df["occupation"].unique()):
        occ_vals = occ_df[occ_df["occupation"]==occ][MST_COL].values
        if len(occ_vals)<2: continue
        es = (np.mean(occ_vals)-grand_mean)/grand_std; ci_lo,ci_hi=bootstrap_ci(occ_vals)
        results.append((occ, np.mean(occ_vals), es, ci_lo, ci_hi, len(occ_vals)))
    results.sort(key=lambda r: r[1])
    print(f"  Grand mean: {grand_mean:.3f}, SD: {grand_std:.3f}")
    print(f"\n  {'Occupation':25s}  {'Mean':>5s}  {'ES':>6s}  {'95% CI':>14s}  {'n':>4s}")
    for occ,m,es,lo,hi,n in results: print(f"  {occ:25s}  {m:5.2f}  {es:+5.2f}  [{lo:5.2f}, {hi:5.2f}]  {n:4d}")
    fig, ax = plt.subplots(figsize=(8, 10)); y=np.arange(len(results))
    means=[r[1] for r in results]; los=[r[3] for r in results]; his=[r[4] for r in results]; labels=[r[0] for r in results]
    colors = ["seagreen" if r[0] in HIGH_PRESTIGE else "goldenrod" if r[0] in MID_PRESTIGE else "crimson" for r in results]
    ax.barh(y, means, xerr=[[m-l for m,l in zip(means,los)],[h-m for m,h in zip(means,his)]],
            color=colors, edgecolor="black", linewidth=0.3, capsize=3)
    ax.axvline(grand_mean, color="gray", linestyle="--", label=f"Grand mean ({grand_mean:.2f})")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.set_xlabel("Mean MST (with 95% bootstrap CI)")
    ax.set_title("Per-Occupation MST with Confidence Intervals"); ax.legend(); ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "E9_per_occupation_forest.png")); plt.close(fig)


def analysis_E10_gradient_surfaces(df):
    section("E", 10, "C9 & C10: SOCIOECONOMIC GRADIENT SURFACES")
    mc = df[df["section"]=="Multi-cue"]
    c9=mc[mc["layer_id"]=="C9"]
    if len(c9)>0:
        print("  C9: Income × Setting × Neighborhood gradient:")
        for inc in ["high","low"]:
            for sett in ["city","village"]:
                for nb in ["gated community","suburb","slum"]:
                    sub=c9[(c9["income"]==inc)&(c9["setting"]==sett)&(c9["neighborhood"]==nb)]
                    if len(sub)>0: print(f"    {inc:5s}+{sett:7s}+{nb:16s}: MST={sub[MST_COL].values[0]}")
        fig, ax = plt.subplots(figsize=(7, 5)); row_labels=[]; matrix_data=[]
        for inc in ["high","low"]:
            for sett in ["city","village"]:
                row_labels.append(f"{inc}+{sett}"); row=[]
                for nb in ["gated community","suburb","slum"]:
                    sub=c9[(c9["income"]==inc)&(c9["setting"]==sett)&(c9["neighborhood"]==nb)]
                    row.append(sub[MST_COL].values[0] if len(sub)>0 else np.nan)
                matrix_data.append(row)
        matrix=pd.DataFrame(matrix_data, index=row_labels, columns=["gated community","suburb","slum"])
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", vmin=3, vmax=8, ax=ax)
        ax.set_title("C9: Income × Setting × Neighborhood Gradient")
        fig.savefig(os.path.join(FIG_DIR, "E10a_c9_gradient.png")); plt.close(fig)
    c10=mc[mc["layer_id"]=="C10"]
    if len(c10)>0:
        print("\n  C10: Gender × Income × Neighborhood gradient:")
        for gen in ["man","woman"]:
            for inc in ["high","low"]:
                for nb in ["gated community","suburb","slum"]:
                    sub=c10[(c10["gender"]==gen)&(c10["income"]==inc)&(c10["neighborhood"]==nb)]
                    if len(sub)>0: print(f"    {gen:6s}+{inc:5s}+{nb:16s}: MST={sub[MST_COL].values[0]}")
        fig, ax = plt.subplots(figsize=(7, 5)); row_labels=[]; matrix_data=[]
        for gen in ["man","woman"]:
            for inc in ["high","low"]:
                row_labels.append(f"{gen}+{inc}"); row=[]
                for nb in ["gated community","suburb","slum"]:
                    sub=c10[(c10["gender"]==gen)&(c10["income"]==inc)&(c10["neighborhood"]==nb)]
                    row.append(sub[MST_COL].values[0] if len(sub)>0 else np.nan)
                matrix_data.append(row)
        matrix=pd.DataFrame(matrix_data, index=row_labels, columns=["gated community","suburb","slum"])
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", vmin=3, vmax=8, ax=ax)
        ax.set_title("C10: Gender × Income × Neighborhood Gradient")
        fig.savefig(os.path.join(FIG_DIR, "E10b_c10_gradient.png")); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  DISTRIBUTION ANALYSES (D1–D12)
# ═══════════════════════════════════════════════════════════════

def analysis_D1_overall(df):
    section("D", 1, "OVERALL MST DISTRIBUTION SHAPE")
    vals = df[MST_COL].values; skew = stats.skew(vals); kurt = stats.kurtosis(vals)
    mode_val = stats.mode(vals, keepdims=True).mode[0]
    print(f"  n={len(vals)}, mean={np.mean(vals):.2f}, median={np.median(vals):.1f}, mode={mode_val}")
    print(f"  skewness={skew:.3f} ({'left-skewed' if skew<-0.5 else 'right-skewed' if skew>0.5 else 'roughly symmetric'})")
    print(f"  kurtosis={kurt:.3f}"); print_dist(vals, "Overall")
    in_56 = np.sum((vals>=5)&(vals<=6)); in_47 = np.sum((vals>=4)&(vals<=7))
    print(f"\n  Concentration: {in_56/len(vals)*100:.1f}% in MST 5-6, {in_47/len(vals)*100:.1f}% in MST 4-7")
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, _, patches = ax.hist(vals, bins=BINS, edgecolor="black", color="steelblue", alpha=0.8)
    for i, p in enumerate(patches):
        if BIN_CENTERS[i]==mode_val: p.set_facecolor("crimson")
    total=len(vals)
    for i, c in enumerate(counts):
        if c>0: ax.text(BIN_CENTERS[i], c+total*0.005, f"{c/total*100:.1f}%", ha="center", fontsize=9)
    ax.set_xlabel("MST Value"); ax.set_ylabel("Count")
    ax.set_title(f"Overall MST Distribution (n={total})\nskew={skew:.2f}, mode={mode_val}"); ax.set_xticks(BIN_CENTERS)
    fig.savefig(os.path.join(FIG_DIR, "D1_overall_distribution.png")); plt.close(fig)


def analysis_D2_income_dist(df):
    section("D", 2, "INCOME: DISTRIBUTION SHIFT")
    has_income = df[df["income"].isin(["high","low"])]
    high=has_income[has_income["income"]=="high"][MST_COL].values; low=has_income[has_income["income"]=="low"][MST_COL].values
    print_dist(high, "High income"); print_dist(low, "Low income")
    ks_report(high, low, "high", "low")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(high, bins=BINS, density=True, alpha=0.6, color="seagreen", edgecolor="black", label=f"High (n={len(high)})")
    axes[0].hist(low, bins=BINS, density=True, alpha=0.6, color="crimson", edgecolor="black", label=f"Low (n={len(low)})")
    axes[0].set_xlabel("MST"); axes[0].set_ylabel("Density"); axes[0].set_title("Income: Overlaid Distributions"); axes[0].legend()
    pct_h=pct_dist(high); pct_l=pct_dist(low); w=0.35
    axes[1].bar(BIN_CENTERS-w/2, pct_h, w, label="High", color="seagreen", edgecolor="black", linewidth=0.5)
    axes[1].bar(BIN_CENTERS+w/2, pct_l, w, label="Low", color="crimson", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("MST"); axes[1].set_ylabel("%"); axes[1].set_title("Income: Side-by-Side"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "D2_income_distribution.png")); plt.close(fig)


def analysis_D3_prestige_dist(df):
    section("D", 3, "OCCUPATION PRESTIGE: DISTRIBUTION SHIFT")
    has_occ = df[df["prestige"].isin(["High","Mid","Low"])]
    groups = {}
    for band in ["High","Mid","Low"]:
        groups[band]=has_occ[has_occ["prestige"]==band][MST_COL].values; print_dist(groups[band], f"{band} prestige")
    ks_report(groups["High"], groups["Low"], "High", "Low")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors={"High":"seagreen","Mid":"goldenrod","Low":"crimson"}
    for band in ["High","Mid","Low"]:
        ax.hist(groups[band], bins=BINS, density=True, alpha=0.5, color=colors[band], edgecolor="black", linewidth=0.5, label=f"{band} (n={len(groups[band])})")
    ax.set_xlabel("MST"); ax.set_ylabel("Density"); ax.set_title("Prestige: Overlaid MST Distributions"); ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "D3_prestige_distribution.png")); plt.close(fig)


def analysis_D4_neighborhood_dist(df):
    section("D", 4, "NEIGHBORHOOD: 3-WAY DISTRIBUTION")
    has_nb = df[df["neighborhood"].isin(["gated community","suburb","slum"])]
    groups = {}
    for nb in ["gated community","suburb","slum"]:
        groups[nb]=has_nb[has_nb["neighborhood"]==nb][MST_COL].values; print_dist(groups[nb], nb)
    ks_report(groups["gated community"], groups["slum"], "gated", "slum")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    colors={"gated community":"seagreen","suburb":"goldenrod","slum":"crimson"}
    for i, nb in enumerate(["gated community","suburb","slum"]):
        ax=axes[i]; v=groups[nb]; pcts=pct_dist(v)
        ax.bar(BIN_CENTERS, pcts, color=colors[nb], edgecolor="black", linewidth=0.5)
        for j,p in enumerate(pcts):
            if p>0: ax.text(BIN_CENTERS[j], p+0.5, f"{p:.0f}%", ha="center", fontsize=8)
        ax.set_xlabel("MST"); ax.set_title(f"{nb}\n(n={len(v)})"); ax.set_xticks(BIN_CENTERS)
        if i==0: ax.set_ylabel("%")
    fig.suptitle("Neighborhood: MST Distribution Comparison", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "D4_neighborhood_distribution.png")); plt.close(fig)


def analysis_D5_setting_dist(df):
    section("D", 5, "SETTING: VILLAGE vs CITY DISTRIBUTION")
    has_s = df[df["setting"].isin(["village","city"])]
    village=has_s[has_s["setting"]=="village"][MST_COL].values; city=has_s[has_s["setting"]=="city"][MST_COL].values
    print_dist(village, "Village"); print_dist(city, "City"); ks_report(village, city, "village", "city")
    fig, ax = plt.subplots(figsize=(8, 5))
    pct_v=pct_dist(village); pct_c=pct_dist(city); w=0.35
    ax.bar(BIN_CENTERS-w/2, pct_c, w, label=f"City (n={len(city)})", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(BIN_CENTERS+w/2, pct_v, w, label=f"Village (n={len(village)})", color="sienna", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("MST"); ax.set_ylabel("%"); ax.set_title("Setting: City vs Village"); ax.set_xticks(BIN_CENTERS); ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "D5_setting_distribution.png")); plt.close(fig)


def analysis_D6_gender_dist(df):
    section("D", 6, "GENDER: DISTRIBUTION COMPARISON")
    man=df[df["gender"]=="man"][MST_COL].values; woman=df[df["gender"]=="woman"][MST_COL].values
    neutral=df[(df["section"]=="N-track")|(df["layer_id"]=="L0")][MST_COL].values
    print_dist(man, "Man"); print_dist(woman, "Woman"); print_dist(neutral, "Neutral")
    ks_report(man, woman, "man", "woman")
    fig, ax = plt.subplots(figsize=(10, 5)); w=0.25
    ax.bar(BIN_CENTERS-w, pct_dist(man), w, label=f"Man (n={len(man)})", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(BIN_CENTERS, pct_dist(woman), w, label=f"Woman (n={len(woman)})", color="crimson", edgecolor="black", linewidth=0.5)
    ax.bar(BIN_CENTERS+w, pct_dist(neutral), w, label=f"Neutral (n={len(neutral)})", color="gray", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("MST"); ax.set_ylabel("%"); ax.set_title("Gender: MST Distribution"); ax.set_xticks(BIN_CENTERS); ax.legend()
    fig.savefig(os.path.join(FIG_DIR, "D6_gender_distribution.png")); plt.close(fig)


def analysis_D7_layer_distributions(df):
    section("D", 7, "LAYER PROGRESSION: DISTRIBUTION AT EACH LAYER")
    ltrack = df[df["section"]=="L-track"]; layers=["L0","L1","L2","L3","L4","L5"]
    layer_data = {}
    for lyr in layers:
        vals=ltrack[ltrack["layer_id"]==lyr][MST_COL].values; layer_data[lyr]=vals
        if len(vals)>1: print_dist(vals, lyr)
    fig, ax = plt.subplots(figsize=(10, 5)); matrix_data=[]; valid_layers=[]
    for lyr in layers:
        v=layer_data[lyr]
        if len(v)>1: matrix_data.append(pct_dist(v)); valid_layers.append(f"{lyr}\n(n={len(v)})")
    if matrix_data:
        matrix=np.array(matrix_data); active_cols=[i for i in range(10) if matrix[:,i].sum()>0]
        matrix_trimmed=matrix[:,active_cols]; col_labels=[str(i+1) for i in active_cols]
        im=ax.imshow(matrix_trimmed.T, aspect="auto", cmap="YlOrRd", interpolation="nearest", origin="lower")
        ax.set_xticks(range(len(valid_layers))); ax.set_xticklabels(valid_layers)
        ax.set_yticks(range(len(col_labels))); ax.set_yticklabels(col_labels)
        ax.set_xlabel("Layer"); ax.set_ylabel("MST Value"); ax.set_title("L-track: Distribution Shift Across Layers")
        plt.colorbar(im, ax=ax, label="% of images")
        for i in range(len(valid_layers)):
            for j in range(len(col_labels)):
                val=matrix_trimmed[i,j]
                if val>0: ax.text(i, j, f"{val:.0f}", ha="center", va="center", fontsize=8, color="black" if val<40 else "white")
    fig.savefig(os.path.join(FIG_DIR, "D7_layer_distribution_heatmap.png")); plt.close(fig)


def analysis_D8_occupation_extremes(df):
    section("D", 8, "TOP vs BOTTOM 5 OCCUPATIONS: DISTRIBUTIONS")
    has_occ = df[df["occupation"].notna()&(df["occupation"]!="")]
    occ_means=has_occ.groupby("occupation")[MST_COL].mean().sort_values()
    lightest5=occ_means.head(5).index.tolist(); darkest5=occ_means.tail(5).index.tolist()
    print("  Lightest 5:")
    for occ in lightest5: print_dist(has_occ[has_occ["occupation"]==occ][MST_COL].values, occ)
    print("\n  Darkest 5:")
    for occ in darkest5: print_dist(has_occ[has_occ["occupation"]==occ][MST_COL].values, occ)
    light_vals=has_occ[has_occ["occupation"].isin(lightest5)][MST_COL].values
    dark_vals=has_occ[has_occ["occupation"].isin(darkest5)][MST_COL].values
    ks_report(light_vals, dark_vals, "lightest-5", "darkest-5")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
    all_occs = lightest5+darkest5
    for i, occ in enumerate(all_occs):
        row=0 if i<5 else 1; col=i%5; ax=axes[row][col]
        vals=has_occ[has_occ["occupation"]==occ][MST_COL].values; pcts=pct_dist(vals)
        color="seagreen" if i<5 else "crimson"
        ax.bar(BIN_CENTERS, pcts, color=color, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{occ}\n(n={len(vals)}, μ={np.mean(vals):.2f})", fontsize=9)
        ax.set_xticks(BIN_CENTERS); ax.set_xlim(0.5, 10.5)
        if col==0: ax.set_ylabel("% of images")
    fig.suptitle("Lightest 5 (top, green) vs Darkest 5 (bottom, red)", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "D8_occupation_extremes.png")); plt.close(fig)


def analysis_D9_income_prestige_grid(df):
    section("D", 9, "INCOME × PRESTIGE: DISTRIBUTION GRID (2×3)")
    has_both = df[df["income"].isin(["high","low"]) & df["prestige"].isin(["High","Mid","Low"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for i, inc in enumerate(["high","low"]):
        for j, band in enumerate(["High","Mid","Low"]):
            ax=axes[i][j]
            vals=has_both[(has_both["income"]==inc)&(has_both["prestige"]==band)][MST_COL].values
            pcts=pct_dist(vals); color="seagreen" if inc=="high" else "crimson"
            ax.bar(BIN_CENTERS, pcts, color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
            ax.set_title(f"{inc} income + {band} prestige\n(n={len(vals)}, μ={np.mean(vals):.2f})", fontsize=10)
            ax.set_xticks(BIN_CENTERS); ax.set_xlim(0.5, 10.5)
            if j==0: ax.set_ylabel("%"); 
            if i==1: ax.set_xlabel("MST")
            if len(vals)>1: print_dist(vals, f"{inc}+{band}")
    fig.suptitle("Income × Prestige: MST Distribution Grid", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "D9_income_prestige_grid.png")); plt.close(fig)


def analysis_D10_cdfs(df):
    section("D", 10, "CUMULATIVE DISTRIBUTION FUNCTIONS (CDF)")
    has_income=df[df["income"].isin(["high","low"])]; has_nb=df[df["neighborhood"].isin(["gated community","suburb","slum"])]
    has_setting=df[df["setting"].isin(["village","city"])]; has_prestige=df[df["prestige"].isin(["High","Low"])]
    comparisons = [
        ("Income", [("High income",has_income[has_income["income"]=="high"][MST_COL].values,"seagreen"),
                     ("Low income",has_income[has_income["income"]=="low"][MST_COL].values,"crimson")]),
        ("Neighborhood", [("Gated",has_nb[has_nb["neighborhood"]=="gated community"][MST_COL].values,"seagreen"),
                           ("Suburb",has_nb[has_nb["neighborhood"]=="suburb"][MST_COL].values,"goldenrod"),
                           ("Slum",has_nb[has_nb["neighborhood"]=="slum"][MST_COL].values,"crimson")]),
        ("Setting", [("City",has_setting[has_setting["setting"]=="city"][MST_COL].values,"steelblue"),
                      ("Village",has_setting[has_setting["setting"]=="village"][MST_COL].values,"sienna")]),
        ("Prestige", [("High",has_prestige[has_prestige["prestige"]=="High"][MST_COL].values,"seagreen"),
                       ("Low",has_prestige[has_prestige["prestige"]=="Low"][MST_COL].values,"crimson")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10)); axes=axes.flatten()
    for idx, (title, groups) in enumerate(comparisons):
        ax=axes[idx]
        for label, vals, color in groups:
            sorted_v=np.sort(vals); cdf=np.arange(1,len(sorted_v)+1)/len(sorted_v)
            ax.step(sorted_v, cdf, where="post", label=f"{label} (n={len(vals)})", color=color, linewidth=2)
        ax.set_xlabel("MST"); ax.set_ylabel("Cumulative Proportion"); ax.set_title(f"{title}: CDF"); ax.legend(fontsize=9)
        ax.set_xlim(1,10); ax.set_xticks(BIN_CENTERS)
        if len(groups)>=2:
            stat,p=stats.ks_2samp(groups[0][1],groups[-1][1])
            ax.text(0.95, 0.05, f"KS D={stat:.3f}\np={p:.2e}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
            print(f"  {title}: KS D={stat:.3f}, p={p:.2e}")
    fig.suptitle("CDF Comparisons", fontsize=14, y=1.01)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "D10_cdf_comparisons.png")); plt.close(fig)


def analysis_D11_entropy(df):
    section("D", 11, "ENTROPY: WHICH CUE CONSTRAINS MST MOST?")
    def entropy(vals):
        pcts=pct_dist(vals)/100; pcts=pcts[pcts>0]; return -np.sum(pcts*np.log2(pcts))
    overall_H=entropy(df[MST_COL].values)
    print(f"  Overall entropy: {overall_H:.3f} bits (max={np.log2(10):.3f})\n")
    cue_configs=[("Income","income",["high","low"]),("Setting","setting",["city","village"]),
                 ("Neighborhood","neighborhood",["gated community","suburb","slum"]),
                 ("Gender","gender",["man","woman"]),("Prestige band","prestige",["High","Mid","Low"])]
    results=[]
    for cue_name, col, values in cue_configs:
        print(f"  {cue_name}:"); entropies=[]
        for val in values:
            sub=df[df[col]==val][MST_COL].values
            if len(sub)<2: continue
            H=entropy(sub); entropies.append(H); print(f"    {val:20s}: H={H:.3f} bits (n={len(sub)})")
        avg_H=np.mean(entropies) if entropies else np.nan; info_gain=overall_H-avg_H if not np.isnan(avg_H) else np.nan
        print(f"    → avg H={avg_H:.3f}, info gain={info_gain:.3f} bits"); results.append((cue_name, avg_H, info_gain))
    if results:
        fig, ax = plt.subplots(figsize=(8, 5)); names=[r[0] for r in results]; gains=[r[2] for r in results]
        colors=["crimson" if g>0.1 else "goldenrod" if g>0.05 else "gray" for g in gains]
        bars=ax.barh(names, gains, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Information Gain (bits)"); ax.set_title("Which Cue Reduces MST Uncertainty Most?"); ax.axvline(0, color="black", linewidth=0.8)
        for bar, g in zip(bars, gains): ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f"{g:.3f}", va="center", fontsize=10)
        ax.invert_yaxis(); fig.savefig(os.path.join(FIG_DIR, "D11_entropy_info_gain.png")); plt.close(fig)


def analysis_D12_occupation_heatmap(df):
    section("D", 12, "PER-OCCUPATION DISTRIBUTION HEATMAP (30 × MST)")
    has_occ = df[df["occupation"].notna()&(df["occupation"]!="")]
    occ_means=has_occ.groupby("occupation")[MST_COL].mean().sort_values(); occ_order=occ_means.index.tolist()
    matrix_data=[]
    for occ in occ_order: matrix_data.append(pct_dist(has_occ[has_occ["occupation"]==occ][MST_COL].values))
    matrix=np.array(matrix_data); active_cols=[i for i in range(10) if matrix[:,i].sum()>0]
    matrix_trimmed=matrix[:,active_cols]; col_labels=[str(i+1) for i in active_cols]
    fig, ax = plt.subplots(figsize=(10, 12))
    im=ax.imshow(matrix_trimmed, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(occ_order)))
    labels=[]
    for occ in occ_order:
        if occ in HIGH_PRESTIGE: labels.append(("seagreen",occ))
        elif occ in MID_PRESTIGE: labels.append(("goldenrod",occ))
        else: labels.append(("crimson",occ))
    ax.set_yticklabels([l[1] for l in labels], fontsize=9)
    for i, (color,_) in enumerate(labels): ax.get_yticklabels()[i].set_color(color)
    ax.set_xlabel("MST Value"); ax.set_title("Per-Occupation MST Distribution (%)")
    plt.colorbar(im, ax=ax, label="% of images", shrink=0.7)
    for i in range(len(occ_order)):
        for j in range(len(col_labels)):
            val=matrix_trimmed[i,j]
            if val>=5: ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color="black" if val<40 else "white")
    fig.savefig(os.path.join(FIG_DIR, "D12_occupation_distribution_heatmap.png")); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  ADVANCED ANALYSES (A1–A11)
# ═══════════════════════════════════════════════════════════════

def analysis_A1_regression(df):
    section("A", 1, "OLS REGRESSION: MST ~ CUES")
    l5 = df[(df["layer_id"]=="L5")&(df["section"]=="L-track")].copy()
    print(f"  Using L5 data: n={len(l5)}")
    l5["gender_f"]=(l5["gender"]=="woman").astype(int); l5["income_f"]=(l5["income"]=="low").astype(int)
    l5["setting_f"]=(l5["setting"]=="village").astype(int)
    l5["prestige_f"]=l5["prestige"].map({"High":0,"Mid":1,"Low":2}).fillna(1)
    l5["neighborhood_f"]=l5["neighborhood"].map({"gated community":0,"suburb":1,"slum":2}).fillna(1)
    feature_names=["gender (woman)","prestige (low→high)","income (low)","setting (village)","neighborhood (slum→gated)"]
    X_cols=["gender_f","prestige_f","income_f","setting_f","neighborhood_f"]
    X=l5[X_cols].values.astype(float); y=l5[MST_COL].values.astype(float)
    X_with_const=np.column_stack([np.ones(len(X)), X])
    beta=np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred=X_with_const@beta; ss_res=np.sum((y-y_pred)**2); ss_tot=np.sum((y-np.mean(y))**2)
    r2=1-ss_res/ss_tot; adj_r2=1-(1-r2)*(len(y)-1)/(len(y)-len(beta)); rmse=np.sqrt(ss_res/len(y))
    n,k=X_with_const.shape; mse=ss_res/(n-k); var_beta=mse*np.linalg.inv(X_with_const.T@X_with_const)
    se=np.sqrt(np.diag(var_beta)); t_stats=beta/se; p_vals=2*(1-stats.t.cdf(np.abs(t_stats), df=n-k))
    X_std=(X-X.mean(axis=0))/X.std(axis=0); X_std_const=np.column_stack([np.ones(len(X_std)), X_std])
    beta_std=np.linalg.lstsq(X_std_const, y, rcond=None)[0]
    print(f"\n  R²={r2:.4f}, Adj R²={adj_r2:.4f}, RMSE={rmse:.3f}")
    print(f"\n  {'Predictor':30s}  {'β':>7s}  {'β_std':>7s}  {'SE':>7s}  {'t':>7s}  {'p':>10s}")
    print(f"  {'-'*80}")
    print(f"  {'Intercept':30s}  {beta[0]:+7.3f}  {'':>7s}  {se[0]:7.3f}  {t_stats[0]:7.2f}  {p_vals[0]:10.2e}")
    for i, name in enumerate(feature_names):
        sig="***" if p_vals[i+1]<0.001 else "**" if p_vals[i+1]<0.01 else "*" if p_vals[i+1]<0.05 else "ns"
        print(f"  {name:30s}  {beta[i+1]:+7.3f}  {beta_std[i+1]:+7.3f}  {se[i+1]:7.3f}  {t_stats[i+1]:7.2f}  {p_vals[i+1]:10.2e}  {sig}")
    fig, ax = plt.subplots(figsize=(8, 5)); std_betas=beta_std[1:]
    colors=["crimson" if b>0 else "steelblue" for b in std_betas]; y_pos=np.arange(len(feature_names))
    ax.barh(y_pos, std_betas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(feature_names)
    ax.set_xlabel("Standardized β (positive = darker MST)"); ax.set_title(f"OLS Regression Coefficients (R²={r2:.3f})")
    ax.axvline(0, color="black", linewidth=0.8)
    for i, b in enumerate(std_betas): ax.text(b+0.01*np.sign(b), i, f"{b:+.3f}", va="center", fontsize=10)
    ax.invert_yaxis(); fig.savefig(os.path.join(FIG_DIR, "A1_regression_coefficients.png")); plt.close(fig)


def analysis_A2_regression_interactions(df):
    section("A", 2, "OLS WITH INTERACTION TERMS")
    l5 = df[(df["layer_id"]=="L5")&(df["section"]=="L-track")].copy()
    l5["gender_f"]=(l5["gender"]=="woman").astype(int); l5["income_f"]=(l5["income"]=="low").astype(int)
    l5["setting_f"]=(l5["setting"]=="village").astype(int)
    l5["prestige_f"]=l5["prestige"].map({"High":0,"Mid":1,"Low":2}).fillna(1)
    l5["neighborhood_f"]=l5["neighborhood"].map({"gated community":0,"suburb":1,"slum":2}).fillna(1)
    l5["income_x_prestige"]=l5["income_f"]*l5["prestige_f"]; l5["income_x_neighborhood"]=l5["income_f"]*l5["neighborhood_f"]
    l5["gender_x_income"]=l5["gender_f"]*l5["income_f"]; l5["setting_x_neighborhood"]=l5["setting_f"]*l5["neighborhood_f"]
    feature_names=["gender","prestige","income","setting","neighborhood","income×prestige","income×neighborhood","gender×income","setting×neighborhood"]
    X_cols=["gender_f","prestige_f","income_f","setting_f","neighborhood_f","income_x_prestige","income_x_neighborhood","gender_x_income","setting_x_neighborhood"]
    X=l5[X_cols].values.astype(float); y=l5[MST_COL].values.astype(float)
    X_with_const=np.column_stack([np.ones(len(X)), X])
    beta=np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    y_pred=X_with_const@beta; ss_res=np.sum((y-y_pred)**2); ss_tot=np.sum((y-np.mean(y))**2)
    r2=1-ss_res/ss_tot; adj_r2=1-(1-r2)*(len(y)-1)/(len(y)-len(beta))
    n,k=X_with_const.shape; mse=ss_res/(n-k); var_beta=mse*np.linalg.inv(X_with_const.T@X_with_const)
    se=np.sqrt(np.diag(var_beta)); t_stats=beta/se; p_vals=2*(1-stats.t.cdf(np.abs(t_stats), df=n-k))
    print(f"  R²={r2:.4f}, Adj R²={adj_r2:.4f}")
    print(f"\n  {'Predictor':25s}  {'β':>7s}  {'SE':>7s}  {'t':>7s}  {'p':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Intercept':25s}  {beta[0]:+7.3f}  {se[0]:7.3f}  {t_stats[0]:7.2f}  {p_vals[0]:10.2e}")
    for i, name in enumerate(feature_names):
        sig="***" if p_vals[i+1]<0.001 else "**" if p_vals[i+1]<0.01 else "*" if p_vals[i+1]<0.05 else "ns"
        print(f"  {name:25s}  {beta[i+1]:+7.3f}  {se[i+1]:7.3f}  {t_stats[i+1]:7.2f}  {p_vals[i+1]:10.2e}  {sig}")


def analysis_A3_asymmetry(df):
    section("A", 3, "DARKENING vs LIGHTENING ASYMMETRY")
    baseline = df[df["layer_id"]=="L0"][MST_COL].mean()
    print(f"  Baseline (L0): MST = {baseline:.1f}\n")
    cue_pairs=[("Income","income","high","low"),("Setting","setting","city","village"),
               ("Neighborhood","neighborhood","gated community","slum"),("Prestige","prestige","High","Low"),
               ("Gender","gender","man","woman")]
    results=[]
    for name, col, light_val, dark_val in cue_pairs:
        light_data=df[df[col]==light_val][MST_COL].values; dark_data=df[df[col]==dark_val][MST_COL].values
        if len(light_data)<2 or len(dark_data)<2: continue
        light_shift=baseline-np.mean(light_data); dark_shift=np.mean(dark_data)-baseline
        ratio=dark_shift/light_shift if light_shift!=0 else np.inf
        print(f"  {name:35s}: light={np.mean(light_data):.3f}, dark={np.mean(dark_data):.3f}, ratio={ratio:.2f}x")
        results.append((name, light_shift, dark_shift, ratio))
    mc = df[df["section"]=="Multi-cue"]
    c1=mc[mc["layer_id"]=="C1"][MST_COL].values; c2=mc[mc["layer_id"]=="C2"][MST_COL].values
    if len(c1)>0 and len(c2)>0:
        print(f"\n  Multi-cue: C2 (light)={np.mean(c2):.1f}, C1 (dark)={np.mean(c1):.1f}")
    if results:
        fig, ax = plt.subplots(figsize=(10, 6)); names=[r[0] for r in results]
        light_shifts=[-r[1] for r in results]; dark_shifts=[r[2] for r in results]; y_pos=np.arange(len(names))
        ax.barh(y_pos-0.15, light_shifts, 0.3, label="Lightening", color="steelblue", edgecolor="black", linewidth=0.5)
        ax.barh(y_pos+0.15, dark_shifts, 0.3, label="Darkening", color="crimson", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos); ax.set_yticklabels(names); ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("MST shift from baseline"); ax.set_title("Darkening vs Lightening Asymmetry"); ax.legend(); ax.invert_yaxis()
        fig.savefig(os.path.join(FIG_DIR, "A3_asymmetry.png")); plt.close(fig)


def analysis_A4_tail_probabilities(df):
    section("A", 4, "TAIL PROBABILITIES: P(MST≥7) AND P(MST≤4)")
    conditions=[("Overall",df),("High income",df[df["income"]=="high"]),("Low income",df[df["income"]=="low"]),
                ("City",df[df["setting"]=="city"]),("Village",df[df["setting"]=="village"]),
                ("Gated community",df[df["neighborhood"]=="gated community"]),("Suburb",df[df["neighborhood"]=="suburb"]),
                ("Slum",df[df["neighborhood"]=="slum"]),("High prestige",df[df["prestige"]=="High"]),
                ("Mid prestige",df[df["prestige"]=="Mid"]),("Low prestige",df[df["prestige"]=="Low"]),
                ("Man",df[df["gender"]=="man"]),("Woman",df[df["gender"]=="woman"])]
    print(f"  {'Condition':25s}  {'n':>5s}  {'P(≤4)':>6s}  {'P(≥7)':>6s}")
    print(f"  {'-'*50}")
    labels,p_light,p_dark=[],[],[]
    for label, sub in conditions:
        vals=sub[MST_COL].values
        if len(vals)==0: continue
        p4=np.mean(vals<=4)*100; p7=np.mean(vals>=7)*100
        print(f"  {label:25s}  {len(vals):5d}  {p4:5.1f}%  {p7:5.1f}%")
        labels.append(label); p_light.append(p4); p_dark.append(p7)
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, label in enumerate(labels):
        color="gray"
        if "income" in label.lower(): color="seagreen" if "High" in label else "crimson"
        elif "prestige" in label.lower(): color="seagreen" if "High" in label else ("goldenrod" if "Mid" in label else "crimson")
        elif "gated" in label.lower(): color="seagreen"
        elif "slum" in label.lower() or "village" in label.lower(): color="crimson"
        ax.scatter(p_light[i], p_dark[i], s=100, c=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(label, (p_light[i], p_dark[i]), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("P(MST ≤ 4) — light skin (%)"); ax.set_ylabel("P(MST ≥ 7) — dark skin (%)")
    ax.set_title("Tail Probability Map"); ax.axhline(p_dark[0], color="gray", linestyle="--", alpha=0.5)
    ax.axvline(p_light[0], color="gray", linestyle="--", alpha=0.5)
    fig.savefig(os.path.join(FIG_DIR, "A4_tail_probability_map.png")); plt.close(fig)


def analysis_A5_cliffs_delta(df):
    section("A", 5, "CLIFF'S DELTA: NON-PARAMETRIC EFFECT SIZES")
    def interpret_cliff(d):
        d=abs(d)
        if d<0.147: return "negligible"
        if d<0.33: return "small"
        if d<0.474: return "medium"
        return "large"
    comparisons=[("Income (low vs high)",df[df["income"]=="low"][MST_COL].values,df[df["income"]=="high"][MST_COL].values),
                 ("Neighborhood (slum vs gated)",df[df["neighborhood"]=="slum"][MST_COL].values,df[df["neighborhood"]=="gated community"][MST_COL].values),
                 ("Setting (village vs city)",df[df["setting"]=="village"][MST_COL].values,df[df["setting"]=="city"][MST_COL].values),
                 ("Prestige (low vs high)",df[df["prestige"]=="Low"][MST_COL].values,df[df["prestige"]=="High"][MST_COL].values),
                 ("Gender (woman vs man)",df[df["gender"]=="woman"][MST_COL].values,df[df["gender"]=="man"][MST_COL].values)]
    results=[]
    print(f"  {'Comparison':35s}  {'δ':>7s}  {'interpretation':>15s}")
    print(f"  {'-'*60}")
    for name, a, b in comparisons:
        d=cliffs_delta_fast(a,b); interp=interpret_cliff(d)
        print(f"  {name:35s}  {d:+.3f}  {interp:>15s}"); results.append((name, d, interp))
    fig, ax = plt.subplots(figsize=(8, 5)); names=[r[0] for r in results]; deltas=[r[1] for r in results]
    colors=["crimson" if d>0.147 else "goldenrod" if d>0 else "steelblue" for d in deltas]; y_pos=np.arange(len(names))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.5); ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.8)
    for thresh in [0.147, 0.33, 0.474]: ax.axvline(thresh, color="gray", linestyle=":", alpha=0.5); ax.axvline(-thresh, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cliff's δ"); ax.set_title("Cliff's Delta: Non-Parametric Effect Sizes"); ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "A5_cliffs_delta.png")); plt.close(fig)


def analysis_A6_variance(df):
    section("A", 6, "VARIANCE ANALYSIS: MODEL CERTAINTY PER CONDITION")
    conditions=[("Overall",df[MST_COL].values),("High income",df[df["income"]=="high"][MST_COL].values),
                ("Low income",df[df["income"]=="low"][MST_COL].values),("City",df[df["setting"]=="city"][MST_COL].values),
                ("Village",df[df["setting"]=="village"][MST_COL].values),
                ("Gated community",df[df["neighborhood"]=="gated community"][MST_COL].values),
                ("Suburb",df[df["neighborhood"]=="suburb"][MST_COL].values),("Slum",df[df["neighborhood"]=="slum"][MST_COL].values),
                ("High prestige",df[df["prestige"]=="High"][MST_COL].values),("Mid prestige",df[df["prestige"]=="Mid"][MST_COL].values),
                ("Low prestige",df[df["prestige"]=="Low"][MST_COL].values),
                ("Man",df[df["gender"]=="man"][MST_COL].values),("Woman",df[df["gender"]=="woman"][MST_COL].values)]
    print(f"  {'Condition':25s}  {'n':>5s}  {'Mean':>5s}  {'SD':>5s}  {'Var':>6s}")
    print(f"  {'-'*50}")
    labels,sds=[],[]
    for label, vals in conditions:
        if len(vals)<2: continue
        sd=np.std(vals,ddof=1); var=np.var(vals,ddof=1)
        print(f"  {label:25s}  {len(vals):5d}  {np.mean(vals):5.2f}  {sd:5.3f}  {var:6.3f}")
        labels.append(label); sds.append(sd)
    fig, ax = plt.subplots(figsize=(10, 6)); y_pos=np.arange(len(labels))
    colors=[]
    for l in labels:
        if l=="Overall": colors.append("gray")
        elif "low" in l.lower() or "slum" in l.lower() or "village" in l.lower(): colors.append("crimson")
        elif "high" in l.lower() or "gated" in l.lower() or "city" in l.lower(): colors.append("seagreen")
        else: colors.append("goldenrod")
    ax.barh(y_pos, sds, color=colors, edgecolor="black", linewidth=0.5); ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("SD of MST"); ax.set_title("Model Certainty: SD per Condition")
    ax.axvline(sds[0], color="gray", linestyle="--", alpha=0.5, label="Overall SD"); ax.legend(); ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "A6_variance_by_condition.png")); plt.close(fig)


def analysis_A7_effect_decomposition(df):
    section("A", 7, "SEQUENTIAL EFFECT DECOMPOSITION (L-TRACK)")
    ltrack=df[df["section"]=="L-track"]; layers=["L0","L1","L2","L3","L4","L5"]
    cue_names=["Gender","Occupation","Income","Setting","Neighborhood"]
    means=[np.mean(ltrack[ltrack["layer_id"]==lyr][MST_COL].values) if len(ltrack[ltrack["layer_id"]==lyr])>0 else np.nan for lyr in layers]
    total_shift=means[-1]-means[0]; print(f"  Total shift L0→L5: {total_shift:+.3f} MST\n")
    deltas=[]; pct_contributions=[]
    for i in range(1,len(means)):
        delta=means[i]-means[i-1]; pct=(delta/total_shift*100) if total_shift!=0 else 0
        deltas.append(delta); pct_contributions.append(pct)
        print(f"  {layers[i-1]}→{layers[i]} ({cue_names[i-1]:12s}): Δ={delta:+.3f} ({pct:+.0f}%)")
    # N-track
    ntrack=df[df["section"]=="N-track"]; n_layers=["L0","N2","N3","N4","N5"]; n_cue_names=["Occupation","Income","Setting","Neighborhood"]
    n_means=[np.mean(ltrack[ltrack["layer_id"]=="L0"][MST_COL].values)]
    for lyr in ["N2","N3","N4","N5"]:
        vals=ntrack[ntrack["layer_id"]==lyr][MST_COL].values; n_means.append(np.mean(vals) if len(vals)>0 else np.nan)
    n_total=n_means[-1]-n_means[0]; print(f"\n  N-track total shift: {n_total:+.3f}")
    for i in range(1,len(n_means)):
        delta=n_means[i]-n_means[i-1]; pct=(delta/n_total*100) if n_total!=0 else 0
        print(f"  {n_layers[i-1]}→{n_layers[i]} ({n_cue_names[i-1]:12s}): Δ={delta:+.3f} ({pct:+.0f}%)")
    # Waterfall
    fig, ax = plt.subplots(figsize=(10, 6)); cumulative=[means[0]]
    for d in deltas: cumulative.append(cumulative[-1]+d)
    colors=["crimson" if d>0.1 else "steelblue" if d<-0.1 else "goldenrod" for d in deltas]
    x=np.arange(len(cue_names)+2)
    ax.bar(0, means[0], color="gray", edgecolor="black", linewidth=0.5)
    ax.text(0, means[0]+0.02, f"{means[0]:.2f}", ha="center", fontsize=10)
    for i, (d, c) in enumerate(zip(deltas, colors)):
        ax.bar(i+1, d, bottom=cumulative[i], color=c, edgecolor="black", linewidth=0.5)
        ax.text(i+1, cumulative[i]+d+0.02*np.sign(d), f"{d:+.3f}\n({pct_contributions[i]:+.0f}%)", ha="center", fontsize=9)
    ax.bar(len(cue_names)+1, means[-1], color="darkgray", edgecolor="black", linewidth=0.5)
    ax.text(len(cue_names)+1, means[-1]+0.02, f"{means[-1]:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(["Baseline\n(L0)"]+cue_names+["Final\n(L5)"], fontsize=10)
    ax.set_ylabel("Mean MST"); ax.set_title("Effect Decomposition: Waterfall"); ax.set_ylim(4.0, 6.2)
    fig.savefig(os.path.join(FIG_DIR, "A7_effect_decomposition_waterfall.png")); plt.close(fig)


def analysis_A8_marginal_vs_conditional(df):
    section("A", 8, "MARGINAL vs CONDITIONAL EFFECTS")
    print("  Income effect (low − high MST) in different contexts:\n")
    contexts=[("S3 (income alone)",df[df["layer_id"]=="S3"]),("L3 (G+O+I)",df[df["layer_id"]=="L3"]),
              ("L4 (G+O+I+S)",df[df["layer_id"]=="L4"]),("L5 (G+O+I+S+N)",df[df["layer_id"]=="L5"]),
              ("X5 (O+I only)",df[df["layer_id"]=="X5"]),("X8 (I+S only)",df[df["layer_id"]=="X8"]),
              ("X9 (I+N only)",df[df["layer_id"]=="X9"])]
    results=[]
    for label, sub in contexts:
        high=sub[sub["income"]=="high"][MST_COL].values; low=sub[sub["income"]=="low"][MST_COL].values
        if len(high)==0 or len(low)==0: continue
        delta=np.mean(low)-np.mean(high); n=len(high)+len(low)
        print(f"    {label:25s}: high={np.mean(high):.2f}, low={np.mean(low):.2f}, Δ={delta:+.3f}"); results.append((label, delta, n))
    print("\n  Setting effect (village − city MST):\n")
    s_contexts=[("S4 (setting alone)",df[df["layer_id"]=="S4"]),("L4 (G+O+I+S)",df[df["layer_id"]=="L4"]),
                ("L5 (G+O+I+S+N)",df[df["layer_id"]=="L5"]),("X8 (I+S only)",df[df["layer_id"]=="X8"]),
                ("X10 (S+N only)",df[df["layer_id"]=="X10"])]
    s_results=[]
    for label, sub in s_contexts:
        city=sub[sub["setting"]=="city"][MST_COL].values; village=sub[sub["setting"]=="village"][MST_COL].values
        if len(city)==0 or len(village)==0: continue
        delta=np.mean(village)-np.mean(city)
        print(f"    {label:25s}: city={np.mean(city):.2f}, village={np.mean(village):.2f}, Δ={delta:+.3f}"); s_results.append((label, delta))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if results:
        ax=axes[0]; labels_r=[r[0] for r in results]; deltas_r=[r[1] for r in results]; y_pos=np.arange(len(labels_r))
        ax.barh(y_pos, deltas_r, color="crimson", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels_r, fontsize=9); ax.set_xlabel("Δ MST (low−high)")
        ax.set_title("Income Effect Across Contexts"); ax.invert_yaxis()
    if s_results:
        ax=axes[1]; labels_s=[r[0] for r in s_results]; deltas_s=[r[1] for r in s_results]; y_pos=np.arange(len(labels_s))
        ax.barh(y_pos, deltas_s, color="sienna", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels_s, fontsize=9); ax.set_xlabel("Δ MST (village−city)")
        ax.set_title("Setting Effect Across Contexts"); ax.invert_yaxis()
    fig.suptitle("Does Context Change a Cue's Effect?", fontsize=14, y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, "A8_marginal_vs_conditional.png")); plt.close(fig)


def analysis_A9_disparate_impact(df):
    section("A", 9, "DISPARATE IMPACT: FAIRNESS METRICS")
    dark_threshold=6; print(f"  Threshold: MST ≥ {dark_threshold} = 'dark'\n")
    comparisons=[("Low vs High income",df[df["income"]=="low"],df[df["income"]=="high"]),
                 ("Low vs High prestige",df[df["prestige"]=="Low"],df[df["prestige"]=="High"]),
                 ("Slum vs Gated",df[df["neighborhood"]=="slum"],df[df["neighborhood"]=="gated community"]),
                 ("Village vs City",df[df["setting"]=="village"],df[df["setting"]=="city"]),
                 ("Woman vs Man",df[df["gender"]=="woman"],df[df["gender"]=="man"])]
    print(f"  {'Comparison':30s}  {'P(dark|A)':>10s}  {'P(dark|B)':>10s}  {'DI ratio':>9s}  {'4/5 rule':>8s}")
    print(f"  {'-'*75}")
    results=[]
    for name, group_a, group_b in comparisons:
        vals_a=group_a[MST_COL].values; vals_b=group_b[MST_COL].values
        if len(vals_a)==0 or len(vals_b)==0: continue
        p_a=np.mean(vals_a>=dark_threshold); p_b=np.mean(vals_b>=dark_threshold)
        di=p_a/p_b if p_b>0 else np.inf; violation="VIOLATION" if di>1.25 or di<0.8 else "OK"
        print(f"  {name:30s}  {p_a*100:9.1f}%  {p_b*100:9.1f}%  {di:9.2f}  {violation:>8s}")
        results.append((name, p_a, p_b, di, violation))
    fig, ax = plt.subplots(figsize=(9, 5)); names=[r[0] for r in results]; dis=[r[3] for r in results]
    colors=["crimson" if v=="VIOLATION" else "seagreen" for _,_,_,_,v in results]; y_pos=np.arange(len(names))
    ax.barh(y_pos, dis, color=colors, edgecolor="black", linewidth=0.5); ax.axvline(1.0, color="black", linewidth=1)
    ax.axvspan(0.8, 1.25, alpha=0.1, color="green", label="4/5 rule OK"); ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.set_xlabel("DI Ratio"); ax.set_title(f"Disparate Impact: P(MST≥{dark_threshold})"); ax.legend(); ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "A9_disparate_impact.png")); plt.close(fig)


def analysis_A10_three_way(df):
    section("A", 10, "3-WAY: GENDER × INCOME × PRESTIGE")
    has_all=df[df["gender"].isin(["man","woman"])&df["income"].isin(["high","low"])&df["prestige"].isin(["High","Mid","Low"])]
    print(f"  n={len(has_all)}\n")
    print(f"  {'Gender':>6s}  {'Prestige':>8s}  {'High-I':>7s}  {'Low-I':>7s}  {'Δ':>6s}")
    print(f"  {'-'*45}")
    plot_data=[]
    for gen in ["man","woman"]:
        for band in ["High","Mid","Low"]:
            sub=has_all[(has_all["gender"]==gen)&(has_all["prestige"]==band)]
            high=sub[sub["income"]=="high"][MST_COL].values; low=sub[sub["income"]=="low"][MST_COL].values
            if len(high)>0 and len(low)>0:
                delta=np.mean(low)-np.mean(high)
                print(f"  {gen:>6s}  {band:>8s}  {np.mean(high):7.2f}  {np.mean(low):7.2f}  {delta:+5.2f}")
                plot_data.append((gen, band, np.mean(high), np.mean(low), delta))
    if plot_data:
        fig, ax = plt.subplots(figsize=(10, 6)); prestige_bands=["High","Mid","Low"]; x=np.arange(len(prestige_bands)); w=0.2
        for i, gen in enumerate(["man","woman"]):
            for j, inc in enumerate(["high","low"]):
                vals=[]
                for band in prestige_bands:
                    match=[d for d in plot_data if d[0]==gen and d[1]==band]
                    vals.append(match[0][2] if inc=="high" else match[0][3] if match else np.nan)
                offset=(i*2+j-1.5)*w; color="steelblue" if inc=="high" else "crimson"; alpha=1.0 if gen=="man" else 0.6
                ax.bar(x+offset, vals, w, label=f"{gen}+{inc}", color=color, alpha=alpha, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(prestige_bands); ax.set_xlabel("Prestige"); ax.set_ylabel("Mean MST")
        ax.set_title("3-Way: Gender × Income × Prestige"); ax.legend(fontsize=9); ax.set_ylim(4.0, 7.0)
        fig.savefig(os.path.join(FIG_DIR, "A10_three_way_interaction.png")); plt.close(fig)


def analysis_A11_chi_squared(df):
    section("A", 11, "CHI-SQUARED: DISTRIBUTION INDEPENDENCE")
    tests=[("Income (high vs low)","income",["high","low"]),("Setting (city vs village)","setting",["city","village"]),
           ("Neighborhood","neighborhood",["gated community","suburb","slum"]),("Prestige","prestige",["High","Mid","Low"]),
           ("Gender (man vs woman)","gender",["man","woman"])]
    print(f"  {'Test':35s}  {'χ²':>8s}  {'df':>3s}  {'p':>12s}  {'Cramér V':>9s}")
    print(f"  {'-'*75}")
    results=[]
    for name, col, values in tests:
        sub=df[df[col].isin(values)]; ct=pd.crosstab(sub[col], sub[MST_COL])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        n_ct=ct.sum().sum(); k=min(ct.shape)-1; cramers_v=np.sqrt(chi2/(n_ct*k)) if k>0 else 0
        sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        print(f"  {name:35s}  {chi2:8.1f}  {dof:3d}  {p:12.2e}  {cramers_v:9.3f}  {sig}")
        results.append((name, chi2, cramers_v, p))
    fig, ax = plt.subplots(figsize=(8, 5)); names=[r[0] for r in results]; vs=[r[2] for r in results]
    colors=["crimson" if v>0.2 else "goldenrod" if v>0.1 else "gray" for v in vs]; y_pos=np.arange(len(names))
    ax.barh(y_pos, vs, color=colors, edgecolor="black", linewidth=0.5); ax.set_yticks(y_pos); ax.set_yticklabels(names)
    ax.set_xlabel("Cramér's V"); ax.set_title("Chi-Squared: Cue–MST Association Strength")
    ax.axvline(0.1, color="gray", linestyle=":", alpha=0.5); ax.axvline(0.3, color="gray", linestyle=":", alpha=0.5); ax.invert_yaxis()
    fig.savefig(os.path.join(FIG_DIR, "A11_chi_squared_cramers_v.png")); plt.close(fig)


# ═══════════════════════════════════════════════════════════════
#  SAVE SUMMARY STATS
# ═══════════════════════════════════════════════════════════════

def save_stats_csv(df):
    rows = []
    for col in ["gender","income","setting","neighborhood"]:
        for val, grp in df.groupby(col):
            if val == "": continue
            rows.append({"cue": col, "value": val, "mean_mst": grp[MST_COL].mean(), "sd": grp[MST_COL].std(), "n": len(grp)})
    for band in ["High","Mid","Low"]:
        sub = df[df["prestige"]==band]
        if len(sub)>0: rows.append({"cue":"prestige","value":band,"mean_mst":sub[MST_COL].mean(),"sd":sub[MST_COL].std(),"n":len(sub)})
    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(OUT_DIR, "summary_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"\nSummary stats saved to {stats_path}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("MST FULL ANALYSIS — 45 analyses, all figures → figures_all/")
    print("="*60)
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} rows from {CSV_PATH}\n")

    # Base (1-12)
    analysis_01_descriptive_overview(df)
    analysis_02_incremental_layers(df)
    analysis_03_single_cue_effects(df)
    analysis_04_income_effect(df)
    analysis_05_occupation(df)
    analysis_06_gender(df)
    analysis_07_geography(df)
    analysis_08_interactions(df)
    analysis_09_multi_cue(df)
    analysis_10_order_effect(df)
    analysis_11_income_prestige_heatmap(df)
    analysis_12_effect_size_summary(df)

    # Extended (E1-E10)
    analysis_E1_ltrack_vs_ntrack(df)
    analysis_E2_x5_occupation_income(df)
    analysis_E3_x8_income_setting(df)
    analysis_E4_x9_income_neighborhood(df)
    analysis_E5_x10_setting_neighborhood(df)
    analysis_E6_c5_vs_c6(df)
    analysis_E7_x2_gender_occupation(df)
    analysis_E8_x6_x7_occupation_geography(df)
    analysis_E9_per_occupation_effect(df)
    analysis_E10_gradient_surfaces(df)

    # Distribution (D1-D12)
    analysis_D1_overall(df)
    analysis_D2_income_dist(df)
    analysis_D3_prestige_dist(df)
    analysis_D4_neighborhood_dist(df)
    analysis_D5_setting_dist(df)
    analysis_D6_gender_dist(df)
    analysis_D7_layer_distributions(df)
    analysis_D8_occupation_extremes(df)
    analysis_D9_income_prestige_grid(df)
    analysis_D10_cdfs(df)
    analysis_D11_entropy(df)
    analysis_D12_occupation_heatmap(df)

    # Advanced (A1-A11)
    analysis_A1_regression(df)
    analysis_A2_regression_interactions(df)
    analysis_A3_asymmetry(df)
    analysis_A4_tail_probabilities(df)
    analysis_A5_cliffs_delta(df)
    analysis_A6_variance(df)
    analysis_A7_effect_decomposition(df)
    analysis_A8_marginal_vs_conditional(df)
    analysis_A9_disparate_impact(df)
    analysis_A10_three_way(df)
    analysis_A11_chi_squared(df)

    save_stats_csv(df)

    print(f"\n{'='*60}")
    print(f"DONE. 45 analyses complete.")
    print(f"  Figures: {FIG_DIR}/")
    print(f"  Stats:   {os.path.join(OUT_DIR, 'summary_stats.csv')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
