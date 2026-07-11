# Colorism in Text-to-Image Generation: Full Analysis Results

**Model:** GPT (gpt-image-1-mini, OpenAI)
**Auditor:** gpt-5-mini-2025-08-07 (Monk Skin Tone scale, 1–10)
**Dataset:** 2,658 images across 6 experimental sections
**Cues tested:** Gender, Occupation (30, grouped by prestige), Income (high/low), Setting (city/village), Neighborhood (gated community/suburb/slum)

---

## Part 1 — Base Analysis (12 analyses)

### 1. Descriptive Overview

MST histogram across all 2,658 images. Mean MST = 5.29 ± 1.14, median = 6, mode = 6. MST range spans only 2–8 out of a possible 1–10. The distribution is heavily concentrated: 61.6% of all images land on MST 6 alone, and 72.6% cluster in MST 5–6. Mean auditor confidence = 0.891.

**What this means:** GPT has an even stronger central tendency than Grok — over 60% of images collapse to a single MST value (6). The model's default skin tone is medium-dark, and it rarely deviates far from this anchor. Extreme light (MST 1) and extreme dark (MST 9–10) are completely absent.

### 2. Incremental Layer Analysis (L-track and N-track)

L-track adds cues one at a time (Gender → Occupation → Income → Setting → Neighborhood) across layers L0–L5. Mean MST starts at 6.0 (L0, bare prompt), drops sharply with gender (L1 = 4.5) and occupation (L2 = 4.2), then climbs back up as income (L3 = 4.7), setting (L4 = 5.4), and neighborhood (L5 = 5.4) are added.

N-track (no gender) follows a similar U-shaped pattern: L0 = 6.0 → N2 = 4.4 → N3 = 4.7 → N4 = 5.4 → N5 = 5.5.

**What this means:** GPT shows a distinctive non-monotonic pattern. Adding occupation initially lightens skin tone substantially (the model associates named occupations with lighter skin), but geographic cues (setting, neighborhood) push it back toward the dark default. The net L0→L5 shift is only −0.6 MST, masking large opposing effects underneath.

### 3. Single-Cue Effects (S1–S5)

Each cue tested in isolation against baseline (L0, MST = 6.0). These are single-image cells (n=1 per value), so they show direction only. Income shows the clearest signal: high income → MST 4, low income → MST 6. Setting shows no movement at all (both city and village → MST 6). Gender shows a large gap (man = 5, woman = 3) in isolation.

**What this means:** GPT responds most strongly to income and occupation in isolation. Notably, setting alone produces zero shift — the model only responds to geographic context when combined with other cues. The large gender gap in isolation (d = 2.0) contrasts with the negligible gap in multi-cue contexts, suggesting gender effects are swamped by socioeconomic signals.

### 4. Income Effect

Cohen's d = 1.203 (large), Mann-Whitney U p = 1.1e-136. Low income produces MST 5.94 vs high income MST 4.79 — a 1.15-point gap. The effect intensifies at earlier layers: d = 2.40 at L3, d = 1.10 at L4, d = 1.17 at L5.

**What this means:** Income is GPT's strongest colorism driver — even stronger than in Grok (d = 1.20 vs 0.76). The model treats "low income" as a near-deterministic darkening signal: 87% of low-income images land at MST 6. This is the single most powerful bias in the entire experiment.

### 5. Occupation by Prestige Band

One-way ANOVA across High/Mid/Low prestige: F = 83.1, p = 1.1e-35, η² = 0.059. Low-prestige occupations average MST 5.69, high-prestige average 5.13 — a 0.56-point spread. Darkest: rickshaw driver (6.00), domestic worker (5.99), maid (5.87). Lightest: librarian (4.60), journalist (4.82), university professor (4.82).

**What this means:** GPT encodes the same occupational colorism hierarchy as Grok: high-status jobs get lighter skin, low-status jobs get darker skin. Domestic worker and rickshaw driver are nearly locked at MST 6, while intellectual occupations (librarian, professor) average nearly 1.4 points lighter.

### 6. Gender Effect

Cohen's d = −0.190 (small). Man: mean = 5.42, Woman: mean = 5.21, Neutral: mean = 5.32.

**What this means:** Gender has a small but detectable effect in GPT — women are rendered slightly lighter than men, unlike Grok where gender was truly negligible (d = −0.08). However, at d = 0.19, this is still dwarfed by income (d = 1.20) and setting (d = 0.71).

### 7. Setting and Neighborhood Effects

Setting (village vs city): d = 0.71, village is 0.71 MST darker (5.80 vs 5.10). Neighborhood (slum vs gated community): d = 0.61, slum is 0.52 MST darker (5.90 vs 5.38).

**What this means:** Geographic cues are strong colorism triggers in GPT. Slum nearly locks the model at MST 6 (84% of slum images), while suburb allows more variation. The setting effect (d = 0.71) is stronger than in Grok (d = 0.47), suggesting GPT is more sensitive to geographic context.

### 8. Two-Cue Interaction Effects

For each two-cue pair (X1–X10), the interaction is calculated as observed minus additive expectation. All interactions are strongly positive — meaning the observed MST is much higher (darker) than predicted from single-cue effects alone. Largest interaction: X4 (gender × neighborhood) = +2.33.

**What this means:** GPT shows massive subadditivity in the opposite direction from what two-cue linearity would predict. The single-cue effects (which shift heavily toward light) do not compound when combined — the model snaps back toward its dark default (MST 6). Two lightening cues together produce far less lightening than their independent effects would suggest.

### 9. Multi-Cue Stacks (C1–C10)

Aligned-dark (C1: low-prestige + low income + village + slum) stays at MST 6.0 — no change from baseline. Aligned-light (C2: high-prestige + high income + city + gated community) reaches MST 5.0 — only 1 point lighter than baseline.

Contradiction prompts (C5, C6, C7, C8) all produce MST 6.0 regardless of which cues conflict.

C9 gradient (Income × Setting × Neighborhood): mean = 5.58, range 4–7.
C10 gradient (Gender × Income × Neighborhood): mean = 5.00, range 2–7.

**What this means:** GPT shows extreme resistance to lightening. Even with all five cues aligned toward "high status," the model only shifts 1 point lighter (6 → 5). In contrast, the dark-aligned prompts don't need to shift at all — the baseline is already dark. This is a profound asymmetry: the model's default is dark, and it resists any combination of cues that would push it lighter. Contradictory cues always resolve to the dark default.

### 10. Order-Effect Analysis

Four alternative cue orderings tested against the standard L5 ordering, using 6 occupations. ANOVA: F = 0.57, p = 0.68 (not significant). All orderings produce means within 0.17 MST of each other.

**What this means:** Cue order does not matter — the model responds to cue content, not cue position. This validates the experimental design and confirms that the observed colorism patterns are driven by semantic content, not syntactic position.

### 11. Income × Occupation Prestige Heatmap

2×3 grid of mean MST. Income effect (low − high) is largest for high-prestige occupations (d = 1.53) and smallest for low-prestige (d = 0.88). High-income + mid-prestige is the lightest cell (4.48); low-income + low-prestige is the darkest (6.06).

**What this means:** Income and prestige contribute independently but interact: the income gap is widest for high-prestige jobs. A high-income surgeon gets much lighter skin than a low-income surgeon, but a high-income cleaner still stays relatively dark. Prestige acts as a ceiling on how much income can lighten.

### 12. Effect Size Summary (Cohen's d)

Ranked from largest to smallest:
- Income (low vs high): d = +1.203 (large)
- Setting (village vs city): d = +0.713 (medium)
- Neighborhood (slum vs gated): d = +0.607 (medium)
- Occupation prestige (low vs high): d = +0.548 (medium)
- Gender (woman vs man): d = −0.190 (small)

**What this means:** The colorism hierarchy in GPT is: Income >> Setting > Neighborhood > Occupation >> Gender. Income alone explains more skin tone variation than all other cues combined. Gender is the weakest signal by a wide margin.

---

## Part 2 — Extended Analysis (E1–E10)

### E1. L-track vs N-track: Gender Moderation

At every layer, the gendered (L) and neutral (N) tracks produce nearly identical MST means. L2 vs N2: Δ = −0.20, d = −0.18, p = 0.46. L5 vs N5: Δ = −0.05, d = −0.05, p = 0.21. None significant.

**What this means:** Removing gender from the prompt changes nothing about how socioeconomic cues affect skin tone. Gender neither amplifies nor dampens the colorism effects. The bias is entirely socioeconomic.

### E2. X5: Occupation × Income — Amplification

Pure income effect alone (S3): Δ = +2.0. When paired with specific occupations, the income gap ranges from 0 (domestic worker, maid, rickshaw driver — locked at MST 6 regardless of income) to +4.0 (journalist, police officer, security guard).

**What this means:** Income amplification is occupation-dependent. Some occupations are so strongly coded as "dark" (domestic worker, rickshaw driver) that even high-income framing cannot lighten them. Others (journalist, police officer) are more responsive to income cues, producing dramatic 4-point swings between high and low income.

### E3. X8: Income × Setting Compounding

High+city = MST 4, low+village = MST 6. Maximum compounding: Δ = +2.0. But high+village also produces MST 6, suggesting village overrides high income.

**What this means:** Geographic context can override income. A high-income person in a village is rendered as dark as a low-income person — the village cue dominates. City setting allows income to lighten, but village locks the model at MST 6.

### E4. X9: Income × Neighborhood Compounding

High+gated = MST 3, low+slum = MST 6. Maximum compounding: Δ = +3.0. High+slum = MST 6 (slum overrides income).

**What this means:** The largest two-cue compounding gap in the study. High income + gated community produces the lightest output in any two-cue combination (MST 3), while poverty-coded neighborhoods override income signals entirely.

### E5. X10: Setting × Neighborhood — Which Dominates?

City+slum = MST 6, village+gated = MST 6. Contradiction gap: Δ = 0.0.

**What this means:** When setting and neighborhood conflict, neither dominates — they resolve to the dark default (MST 6). This differs from Grok where setting showed some moderation. GPT's default is to render dark skin whenever any poverty cue is present.

### E6. C5 vs C6: Occupation vs All Other Signals

C5 (surgeon + low-income context): MST 6. C6 (cleaner + high-income context): MST 6. Gap = 0.

**What this means:** In GPT, neither occupation prestige nor income context can overcome the other when contradicted — both resolve to the dark default. The model appears to have a strong MST 6 attractor that dominates in ambiguous or contradictory prompts.

### E7. X2: Gender × Occupation Stereotypes

Women are rendered lighter than men for most occupations (17 out of 30 show woman < man). Largest gaps: chef, factory worker, nurse, police officer, social worker, teacher — all show women 2 MST points lighter than men. Some low-status occupations (construction laborer, security guard, waste collector) show the reverse: women 1 point darker.

**What this means:** GPT shows gendered occupation stereotypes: women in traditionally "caring" or "intellectual" roles are lightened more than men, while women in manual labor roles are darkened relative to men. This is a gender × occupation interaction that doesn't appear in the aggregate gender statistics.

### E8. X6 & X7: Occupation × Geography

X6 (Occupation × Setting): High-prestige + city = MST 4.0, Low-prestige + village = MST 6.0. X7 (Occupation × Neighborhood): Mid-prestige + suburb = MST 3.2 (lightest cell), Low-prestige + slum = MST 6.3 (darkest).

**What this means:** Occupation prestige and geography compound strongly. The lightest combination (high-prestige + city) is 2 full MST points lighter than the darkest (low-prestige + village). Geographic context amplifies occupational colorism.

### E9. Per-Occupation Effect Size vs Baseline

Grand mean: 5.29, SD = 1.14. Largest negative effects (lightest): librarian (ES = −0.61), journalist (−0.41), university professor (−0.41). Largest positive effects (darkest): rickshaw driver (+0.62), domestic worker (+0.61), maid (+0.51).

**What this means:** The per-occupation effect sizes confirm the prestige-colorism mapping. The full spread from lightest (librarian, 4.60) to darkest (rickshaw driver, 6.00) is 1.4 MST points — equivalent to nearly 1.3 standard deviations. Intellectual occupations lighten; manual labor occupations darken.

### E10. C9 & C10: Socioeconomic Gradient Surfaces

C9 (Income × Setting × Neighborhood): Lightest = high+city+gated (MST 4) and high+city+suburb (MST 4). Darkest = low+village+gated (MST 7). Range: 4–7.

C10 (Gender × Income × Neighborhood): Lightest = woman+high+suburb (MST 2). Darkest = man+low+slum (MST 7). Range: 2–7.

**What this means:** The full gradient surface spans 5 MST points (2–7), showing the model is capable of large shifts when multiple cues align. The lightest achievable combination (woman + high income + suburb, MST 2) is 5 points away from the darkest (man + low income + slum, MST 7). But these extremes require specific multi-cue alignment — the model's default attractor (MST 6) dominates in most conditions.

---

## Part 3 — Distribution Analysis (D1–D12)

### D1. Overall MST Distribution Shape

Left-skewed distribution (skewness = −1.14, kurtosis = 0.26). 61.6% at MST 6 alone. 89.2% within MST 4–7.

**What this means:** GPT's skin tone output is not normally distributed — it has a massive spike at MST 6. The model has a strong "default dark" attractor that most prompts cannot overcome. This is qualitatively different from a uniform or bell-shaped distribution.

### D2. Income: Distribution Shift

High income: spread across MST 2–8 with 44% at MST 6. Low income: 87.1% concentrated at MST 6 alone. KS test: D = 0.462, p = 2.5e-111.

**What this means:** Low income nearly eliminates diversity — 87% of images are locked at a single skin tone. High income produces a broader (though still skewed) distribution. Income doesn't just shift the mean — it fundamentally changes the shape of the distribution.

### D3. Occupation Prestige: Distribution Shift

Low prestige: 74.4% at MST 6, concentrated and narrow. High prestige: 54.9% at MST 6, with more spread into MST 3–4. KS test (high vs low): D = 0.244, p = 5.1e-23.

**What this means:** Prestige affects distribution breadth. High-prestige occupations allow more skin tone diversity; low-prestige occupations collapse toward the dark default.

### D4. Neighborhood: 3-Way Distribution

Slum: 84.0% at MST 6, near-total concentration. Suburb: most diverse distribution (50.8% at MST 6, 17.3% at MST 4). Gated community: 64.3% at MST 6. KS test (gated vs slum): D = 0.217, p = 9.6e-13.

**What this means:** Slum produces the most collapsed distribution in the study — nearly 9 out of 10 images are MST 6. Suburb allows the most diversity, suggesting it's the most "neutral" neighborhood cue.

### D5. Setting: Village vs City Distribution

Village: 82.7% at MST 6, minimal spread. City: 53.6% at MST 6, with 17.3% at MST 4. KS test: D = 0.301, p = 1.1e-42.

**What this means:** Village behaves like "slum" — it locks the model at MST 6. City allows more variability. The model treats rural/poverty settings as strong constraints on skin tone, while urban/affluent settings allow more diversity.

### D6. Gender: Distribution Comparison

Man: 63.3% at MST 6, 18.7% at MST 4. Woman: 61.2% at MST 6, but 11.9% at MST 3 (lighter tail). KS test: D = 0.111, p = 3.1e-5.

**What this means:** Women have a slightly lighter tail distribution (more MST 2–3 representations) while men are more concentrated at MST 6. The gender effect is weak but shows women getting slightly more diverse (and lighter) skin tone representation.

### D7. Layer Progression: Distribution at Each Layer

L2 (occupation added): most diverse layer — 36.7% at MST 4, 21.7% at MST 3, only 15% at MST 6. By L5 (all cues): 65.3% at MST 6, distribution collapsed back.

**What this means:** Adding occupation initially diversifies skin tone output (away from the MST 6 default), but subsequent geographic cues push the distribution back toward the dark attractor. The model's diversity peaks at L2 and declines as more cues are added.

### D8. Top vs Bottom 5 Occupations: Distributions

Lightest 5 (librarian, journalist, professor, social worker, diplomat): spread across MST 2–6, no single value exceeds 45%. Darkest 5 (cleaner, street vendor, maid, domestic worker, rickshaw driver): rickshaw driver is 97.0% at MST 6, domestic worker is 95.6% at MST 6. KS test: D = 0.470, p = 1.6e-41.

**What this means:** The darkest occupations show near-zero diversity — the model has effectively hard-coded certain occupations to a single skin tone. The lightest occupations show genuine distributional variety. This is not just a mean shift; it's a diversity collapse for low-status occupations.

### D9. Income × Prestige: Distribution Grid (2×3)

Low-income + any prestige: 85–88% at MST 6, near-total collapse. High-income + high-prestige: only 32.9% at MST 6, with 31.3% at MST 4 — the most diverse cell. High-income + mid-prestige: 8.2% at MST 2, the highest extreme-light rate in any cell.

**What this means:** The combination of high income and high prestige is the only condition that genuinely breaks the MST 6 lock. Low income overrides prestige entirely — whether surgeon or cleaner, low income → MST 6.

### D10. Cumulative Distribution Functions (CDF)

Income: KS D = 0.462 (largest gap). Setting: KS D = 0.301. Prestige: KS D = 0.244. Neighborhood: KS D = 0.217. All p < 1e-12.

**What this means:** CDFs confirm the hierarchy: income creates the largest distributional gap, followed by setting, prestige, and neighborhood. All are highly significant.

### D11. Entropy: Which Cue Constrains MST Most?

Overall entropy: 1.74 bits (max possible: 3.32). Information gain by cue: Income = 0.350 bits (most constraining), Setting = 0.279, Neighborhood = 0.222, Prestige = 0.073, Gender = 0.069.

Low-income entropy: 0.73 bits (extreme constraint). High-income entropy: 2.05 bits (more freedom). Village entropy: 0.99 bits. City entropy: 1.94 bits.

**What this means:** Income removes the most uncertainty about skin tone — knowing someone's income level tells you more about what skin tone GPT will generate than any other single cue. Low-income prompts reduce entropy to 0.73 bits (the model is nearly deterministic), while high-income prompts leave 2.05 bits of uncertainty (genuine diversity).

### D12. Per-Occupation Distribution Heatmap (30 × MST)

30-occupation × MST heatmap showing the full distribution for each occupation. Clear gradient from intellectual occupations (spread across MST 2–6) to manual labor (concentrated at MST 6).

**What this means:** Visual confirmation that the occupational colorism hierarchy operates not just on means but on entire distributions. The heatmap reveals which occupations the model treats as "determined" (one dominant MST) versus "flexible" (spread across multiple values).

---

## Part 4 — Advanced Analysis (A1–A11)

### A1. OLS Regression: MST ~ Cues

Using L5 data (n = 720): R² = 0.434, Adj R² = 0.430, RMSE = 0.805. All five predictors are significant.

Standardized betas: Income (β_std = 0.542) >> Setting (0.353) > Neighborhood (0.202) > Prestige (0.179) > Gender (−0.083).

**What this means:** The five cues together explain 43.4% of skin tone variance — higher than Grok's 35%. Income alone accounts for the largest share. The model's colorism is substantially predictable from socioeconomic text cues.

### A2. OLS with Interaction Terms

R² = 0.475, Adj R² = 0.468 (up 4 points from main effects only). Three significant interactions: income × prestige (β = −0.254, p < 0.001), income × neighborhood (β = −0.379, p < 1e-7), setting × neighborhood (β = −0.262, p < 0.001). Gender × income is not significant.

**What this means:** Interactions add meaningful explanatory power. The negative interaction signs mean these cues partially substitute for each other — if income already darkened the person, neighborhood adds less incremental darkening. This confirms the bounded/ceiling effect observed in the two-cue analyses.

### A3. Darkening vs Lightening Asymmetry

Baseline (L0) = MST 6.0. Lightening magnitude vs darkening magnitude for each cue:
- Income: light = 4.79, dark = 5.94 (lightening 20× stronger than darkening)
- Setting: light = 5.10, dark = 5.80 (lightening 4.5× stronger)
- Neighborhood: light = 5.38, dark = 5.90 (lightening 5.6× stronger)
- Prestige: light = 5.13, dark = 5.69 (lightening 2.8× stronger)

Multi-cue: C2 (aligned light) = 5.0, C1 (aligned dark) = 6.0.

**What this means:** Because GPT's baseline is already dark (MST 6), "darkening" cues have almost nowhere to push. All the action is in lightening — high-status cues can pull MST down from 6, but low-status cues can barely push it higher. This is a fundamentally asymmetric system: the model starts dark and resists further darkening while allowing selective lightening.

### A4. Tail Probabilities: P(MST ≥ 7) and P(MST ≤ 4)

P(≤4) overall = 24.9%, P(≥7) overall = 2.4%. High income: 42.8% at ≤4 vs 0.9% at ≥7. Low income: 1.0% at ≤4 vs 4.1% at ≥7. Suburb: 33.2% at ≤4 (highest light-tail). Slum: 4.8% at ≥7 (highest dark-tail). Women: 1.3% at ≥7 (lowest dark-tail).

**What this means:** High income produces 43× the light-tail probability compared to low income. Slum produces 5× the dark-tail compared to suburb. The extreme tails of the distribution are almost entirely determined by socioeconomic cues.

### A5. Cliff's Delta: Non-Parametric Effect Sizes

Income: δ = +0.507 (large). Setting: δ = +0.319 (small). Neighborhood: δ = +0.239 (small). Prestige: δ = +0.279 (small). Gender: δ = −0.072 (negligible).

**What this means:** Non-parametric effect sizes confirm the parametric findings. Income is the only cue reaching "large" effect territory. Gender is negligible by any measure.

### A6. Variance Analysis: Model Certainty Per Condition

Low income: SD = 0.41, Var = 0.16 (extremely narrow). High income: SD = 1.29, Var = 1.66 (wide). Slum: SD = 0.53, Var = 0.29 (narrow). Suburb: SD = 1.26, Var = 1.59 (wide). Village: SD = 0.66, Var = 0.44 (narrow).

**What this means:** The model is most "certain" (least variable) when generating poverty-coded prompts — it locks onto MST 6 with very low variance. Affluent cues produce higher variance, meaning the model is less deterministic. Poverty = certainty of dark skin; affluence = some possibility of lighter skin but no guarantee.

### A7. Sequential Effect Decomposition (L-track)

Total shift L0→L5: −0.597 MST. Decomposition: Gender (−1.500, 251% of total), Occupation (−0.300, 50%), Income (+0.533, −89%), Setting (+0.650, −109%), Neighborhood (+0.019, −3%).

N-track total shift: −0.544. Occupation (−1.600, 294%), Income (+0.283, −52%), Setting (+0.758, −139%), Neighborhood (+0.014, −3%).

**What this means:** Gender and occupation pull MST down (lighten), but income and setting push it back up (darken). The net effect is small because opposing forces nearly cancel. This differs from Grok where all cues pushed in the same (darkening) direction. GPT has a more complex internal dynamic where some cues lighten and others darken.

### A8. Marginal vs Conditional Effects

Income effect in different contexts: S3 (alone) Δ = +2.00, X5 (with occupation) Δ = +2.17, L3 (with gender+occupation) Δ = +1.97, L5 (full context) Δ = +1.08.

Setting effect: S4 (alone) Δ = 0.00, L4 (multi-cue) Δ = +1.12, L5 (full) Δ = +0.71.

**What this means:** Income's effect is robust across contexts but attenuates in full multi-cue prompts (2.0 → 1.1). Setting shows no effect alone but gains power in multi-cue contexts (0.0 → 0.7) — it's purely a contextual amplifier. This reveals that some cues only "activate" when combined with others.

### A9. Disparate Impact: Fairness Metrics (4/5 Rule)

Using threshold MST ≥ 6 as "dark":
- Low vs High income: DI ratio = 2.03 — **VIOLATION** (91.1% vs 44.9%)
- Village vs City: DI ratio = 1.54 — **VIOLATION** (86.0% vs 55.8%)
- Low vs High prestige: DI ratio = 1.44 — **VIOLATION** (80.1% vs 55.7%)
- Slum vs Gated: DI ratio = 1.32 — **VIOLATION** (88.8% vs 67.2%)
- Woman vs Man: DI ratio = 0.94 — OK (62.5% vs 66.7%)

**What this means:** Four out of five socioeconomic comparisons violate the EEOC's 4/5 rule for disparate impact. Income produces the most extreme violation: low-income prompts are twice as likely to produce dark-skinned images. Only gender passes the fairness threshold.

### A10. 3-Way: Gender × Income × Prestige

Income gap (low − high MST) by gender and prestige:
- Man + High prestige: Δ = +1.20
- Man + Mid prestige: Δ = +1.18
- Man + Low prestige: Δ = +0.64
- Woman + High prestige: Δ = +1.32
- Woman + Mid prestige: Δ = +1.38
- Woman + Low prestige: Δ = +0.78

**What this means:** Women show a slightly larger income gap than men across all prestige bands. The income effect is strongest for high/mid-prestige occupations (Δ ≈ 1.2–1.4) and weakest for low-prestige (Δ ≈ 0.6–0.8), confirming that low-prestige occupations are harder to lighten even with high income.

### A11. Chi-Squared: Distribution Independence

Income: χ² = 657.3, Cramér's V = 0.534 (large). Setting: χ² = 260.5, V = 0.351 (medium). Gender: χ² = 116.4, V = 0.254 (small-medium). Neighborhood: χ² = 219.6, V = 0.247 (small-medium). Prestige: χ² = 240.3, V = 0.215 (small). All p < 1e-22.

**What this means:** All cues produce statistically significant distributional shifts (all p essentially zero). Cramér's V confirms income has the strongest association with the MST distribution (V = 0.534), meaning income alone explains over a quarter of the variation in the categorical distribution.

---

## Summary of Key Findings (GPT / gpt-image-1-mini)

**The Colorism Hierarchy:** Income (d = 1.20) >> Setting (d = 0.71) > Neighborhood (d = 0.61) > Occupation Prestige (d = 0.55) >> Gender (d = 0.19). Income is the dominant driver, even more so than in Grok.

**The MST 6 Attractor:** 61.6% of all images land at MST 6. The model has a powerful dark default that most prompts cannot overcome. Low-status cues lock the model at MST 6 with near-zero variance.

**The Asymmetry Finding:** GPT's baseline is dark (MST 6), so "darkening" cues have minimal room to push further. All the variation is in lightening — high-status cues can pull MST from 6 down to 4–5, but low-status cues barely move it above 6. This is the opposite direction from what a symmetric system would produce.

**The Diversity Collapse:** Low income reduces entropy to 0.73 bits (near-deterministic). Rickshaw driver is 97% MST 6. Domestic worker is 95.6% MST 6. Poverty-coded prompts eliminate skin tone diversity entirely.

**The Compounding Finding:** Multi-cue stacks show massive subadditivity — two lightening cues produce far less lightening than predicted. The model resists moving far from its MST 6 attractor, even with multiple aligned high-status cues.

**Fairness Violations:** 4 of 5 socioeconomic comparisons violate the EEOC 4/5 rule. Income produces a 2:1 disparate impact ratio. Only gender passes.

**R² = 0.43:** Five text cues explain 43% of skin tone variance — a substantial and statistically significant fraction of the model's colorism is directly traceable to socioeconomic language.

**Gender is Weak:** d = 0.19, the smallest effect. Colorism in GPT is socioeconomic, not gendered — though GPT does show a small gender × occupation interaction where women in caring/intellectual roles are lightened relative to men.

**No Order Effects:** Cue position in the prompt is irrelevant (F = 0.57, p = 0.68). The bias is semantic, not syntactic.
