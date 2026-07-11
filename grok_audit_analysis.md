# Colorism in Text-to-Image Generation: Full Analysis Results

**Model:** Grok (grok-imagine-image, xAI)
**Auditor:** gpt-5-mini-2025-08-07 (Monk Skin Tone scale, 1–10)
**Dataset:** 2,660 images across 6 experimental sections
**Cues tested:** Gender, Occupation (30, grouped by prestige), Income (high/low), Setting (city/village), Neighborhood (gated community/suburb/slum)

---

## Part 1 — Base Analysis (12 analyses)

### 1. Descriptive Overview

MST histogram across all 2,660 images. Mean MST = 5.50 ± 1.14, median = 6, mode = 6. 77.7% of all images cluster in MST 5–6. The effective range spans only MST 2–9 out of a possible 1–10.

**What this means:** The model has a strong central tendency — it defaults to a narrow band of medium-to-medium-dark skin tones regardless of prompt. The full MST scale is never used; extreme light (1–2) and extreme dark (9–10) are virtually absent.

### 2. Incremental Layer Analysis (L-track and N-track)

L-track adds cues one at a time in a fixed order (Gender → Occupation → Income → Setting → Neighborhood) across layers L0–L5. Mean MST shifts from 5.0 (L0, bare prompt) to 5.6 (L5, all 5 cues). Each added cue nudges MST upward.

N-track does the same without gender (using "person" instead of "man"/"woman"). It follows a nearly identical trajectory: 5.0 → 5.6.

**What this means:** Adding socioeconomic context to a prompt consistently darkens the generated person. This happens whether or not gender is specified, which immediately suggests gender is not a driver of colorism — the other cues are.

### 3. Single-Cue Effects (S1–S5)

Each cue tested in isolation against the bare baseline (L0, MST = 5.0). These are single-image cells (n=1 per value), so they show direction only, not statistical robustness. Income and neighborhood show the largest marginal shifts; gender shows almost none.

**What this means:** Even one word change — adding "low income" or "slum" to the prompt — is enough to shift the model's skin tone output. This establishes that each cue independently carries a skin-tone signal.

### 4. Income Effect

Cohen's d = 0.76 (medium-to-large), Mann-Whitney U p = 2.6e-73. Low income produces MST 0.63 higher (darker) than high income across all sections where income appears. The effect is consistent across L-track layers: d = 0.70 at L3, 0.84 at L4, 0.97 at L5.

**What this means:** Income is the second-strongest driver of colorism. The model systematically generates darker-skinned people for "low income" prompts and lighter-skinned people for "high income" prompts. The effect actually strengthens as more cues stack on top of it.

### 5. Occupation by Prestige Band

One-way ANOVA across High/Mid/Low prestige: F = 159, p = 7.6e-66, η² = 0.11. Low-prestige occupations are 0.66 MST darker than high-prestige occupations on average. Ranking all 30 occupations: rickshaw driver is darkest (6.31), librarian is lightest (4.72) — a 1.6-point spread on a 10-point scale.

**What this means:** The model encodes an occupational hierarchy that maps directly onto skin tone. High-status jobs (surgeon, lawyer, architect) get lighter skin; low-status jobs (cleaner, waste collector, rickshaw driver) get darker skin. This is a direct manifestation of colorist stereotypes.

### 6. Gender Effect

Cohen's d = -0.08 (negligible). Man: mean = 5.50, Woman: mean = 5.43, Neutral ("person"): mean = 5.56. No meaningful difference.

**What this means:** Gender is not a significant driver of skin tone in this model. This is an important null result — the colorism bias is not gendered, it is socioeconomic.

### 7. Setting and Neighborhood Effects

Setting (village vs city): d = 0.47, village is 0.4 MST darker. Neighborhood (slum vs gated community): d = 1.16, slum is 1.0 MST darker — the largest single-cue effect size in the entire study.

**What this means:** Geographic cues are powerful colorism triggers. "Slum" alone produces a full MST point of darkening compared to "gated community." The model has learned a strong association between poverty-coded places and dark skin.

### 8. Two-Cue Interaction Effects

For each two-cue pair (X1–X10), the interaction is calculated as: observed MST minus the sum of each single-cue effect (additive expectation). Most interactions are negative (subadditive) — combining two darkening cues produces less darkening than their effects would predict if they were independent.

**What this means:** The cues don't fully stack. Two "darkening" signals together produce a weaker-than-expected combined effect — likely because the model hits a ceiling in how dark it renders people. This is evidence of a bounded response, not independent additive bias.

### 9. Multi-Cue Stacks (C1–C10)

Aligned-dark prompts (C1: low-prestige + low income + village + slum) reach MST 7. Aligned-light prompts (C2: high-prestige + high income + city + gated community) stay at MST 5 — barely different from the bare baseline.

**What this means:** This is the asymmetry finding. The model can darken substantially (5.0 → 7.0) when all cues signal low status, but it cannot lighten much (5.0 → 5.0) even when all cues signal high status. There is a "whitening ceiling" at the baseline, but no comparable "darkening ceiling."

### 10. Order-Effect Analysis

Four alternative cue orderings tested against the standard L5 ordering, using the same 6 occupations. ANOVA: F = 0.53, p = 0.71 (not significant).

**What this means:** The order in which cues appear in the prompt does not matter. "A low-income man from a village" produces the same skin tone as "a man from a village with low income." The model responds to cue content, not cue position.

### 11. Income × Occupation Prestige Heatmap

2×3 grid of mean MST. The income effect (low − high) is consistent across all three prestige bands: d = 0.74 for High prestige, d = 0.78 for Mid, d = 0.84 for Low.

**What this means:** Income and occupation prestige contribute independently. Being a low-income surgeon still darkens the person relative to a high-income surgeon, and vice versa. The two biases stack rather than one overriding the other.

### 12. Effect Size Summary (Cohen's d)

Ranked cue effect sizes:

- Neighborhood (slum vs gated): d = 1.16 — large
- Occupation (low vs high prestige): d = 0.80 — large
- Income (low vs high): d = 0.76 — medium-to-large
- Setting (village vs city): d = 0.47 — small-to-medium
- Gender (woman vs man): d = -0.08 — negligible

**What this means:** The hierarchy is clear. Socioeconomic and geographic cues drive colorism; gender does not. Neighborhood alone has nearly 15× the effect of gender.

---

## Part 2 — Extended Analysis (10 analyses)

### E1. L-track vs N-track Statistical Comparison

Mann-Whitney U tests at each matched layer (L2/N2 through L5/N5). All p-values > 0.23. No significant difference at any layer.

**What this means:** Statistically confirms that gender specification ("man"/"woman" vs "person") does not moderate how the model responds to occupation, income, setting, or neighborhood cues.

### E2. X5: Income Effect per Occupation

Income effect (low − high MST) computed for each of the 30 occupations in the X5 two-cue section. Range: Δ = +3.0 (cleaner, judge — low income dramatically darkens) to Δ = -1.0 (pharmacist — reversed effect).

**What this means:** The income effect is not uniform. For some occupations, saying "low income" has a massive darkening effect; for others, almost none or even a slight reversal. The model's stereotypes are occupation-specific, not a flat bias.

### E3. X8: Income × Setting Compounding

Four cells: low+village reaches MST 7.0, high+city stays at MST 5.0. Maximum compounding Δ = 2.0.

**What this means:** When income and setting are both "low-status," the effects compound to produce substantial darkening — 2 full MST points above the high-status combination.

### E4. X9: Income × Neighborhood Compounding

Six cells: low+slum reaches MST 7.0, high+gated reaches MST 4.0. Maximum compounding Δ = 3.0 — the largest two-cue gap in the entire study.

**What this means:** Income + neighborhood is the most powerful two-cue darkening combination. A person described as "low income, living in a slum" is rendered 3 MST points darker than "high income, living in a gated community" — a massive perceptual difference.

### E5. X10: Setting × Neighborhood Contradiction

When setting and neighborhood conflict (city+slum vs village+gated), city+slum (MST 7) > village+gated (MST 6). Neighborhood dominates.

**What this means:** When geographic cues contradict each other, the neighborhood cue wins. "Slum" overrides "city" to darken the output; "gated community" overrides "village" to lighten it. Neighborhood is the more potent geographic signal.

### E6. C5 vs C6: Occupation vs Context

C5 (surgeon + low income + village + slum) reaches MST 9. C6 (cleaner + high income + city + gated community) stays at MST 6. Context signals (income, setting, neighborhood) overpower occupation — and C5 actually exceeds the aligned-dark stack C1 (MST 7).

**What this means:** A surgeon placed in a slum with low income is rendered darker than a prompt where everything is aligned dark. Context doesn't just override occupation — it amplifies beyond the expected maximum. The model's colorism is driven more by environmental signals than by job title.

### E7. X2: Gender × Occupation Stereotypes

Gender gap (woman − man MST) per occupation. Mostly near zero. Notable exceptions: maid (woman rendered 2.0 lighter than man), airline pilot (woman rendered 1.0 darker than man).

**What this means:** Gender-occupation stereotypes exist but are sparse and inconsistent. When a woman is depicted in a gender-stereotyped role (maid), she's rendered lighter; when in a counter-stereotyped role (pilot), slightly darker. But most occupations show no gender gap at all.

### E8. X6/X7: Occupation × Geography

Prestige × setting and prestige × neighborhood heatmaps. Low-prestige + village hits MST 6.8; high-prestige + city hits MST 4.9.

**What this means:** Occupation prestige and geographic cues compound systematically. The darkest outputs come from combining low-status jobs with low-status locations — the biases reinforce each other across all prestige bands.

### E9. Per-Occupation Forest Plot

Bootstrap 95% CIs for all 30 occupations. Rickshaw driver CI [6.18, 6.46] does not overlap with librarian CI [4.42, 5.01]. Clear prestige-band clustering: green (high prestige) occupations cluster left (lighter), red (low prestige) cluster right (darker).

**What this means:** The occupation-based skin tone differences are statistically robust, not noise. The model has reliably different skin tone outputs for different occupations, and these differences align with real-world occupational prestige hierarchies.

### E10. Gradient Surfaces (C9, C10)

C9 (Income × Setting × Neighborhood, 12 cells): high+city+slum hits MST 8 — "slum" overrides even affluent signals. C10 (Gender × Income × Neighborhood, 12 cells): man+low+slum (MST 7) vs woman+high+gated (MST 5). Gender contributes minimally on top of income+neighborhood.

**What this means:** In three-way combinations, neighborhood remains the dominant force. Even "high income + city" cannot counteract "slum" — the model's association between slums and dark skin is the single strongest bias. Gender adds negligible moderation.

---

## Part 3 — Distribution Analysis (12 analyses)

### D1. Overall Distribution Shape

Skewness = -0.22 (roughly symmetric), kurtosis = 0.52 (slightly peaked), mode = 6. The distribution is not normal — it's concentrated in a narrow band.

**What this means:** The model doesn't produce a spread of skin tones; it produces a spike at MST 5–6. This is a representational harm: entire swathes of the skin tone spectrum are systematically underrepresented.

### D2. Income Distribution Shift

KS test: D = 0.373, p = 8.2e-72. The entire distribution shifts — the mode changes from MST 5 (high income) to MST 6 (low income). Not just a mean shift; the shape changes.

**What this means:** Income doesn't just nudge the average; it reshapes the entire distribution. High-income prompts produce a more spread distribution centered at MST 5; low-income prompts collapse the distribution toward MST 6. The model becomes more "certain" about skin tone for low-income prompts.

### D3. Prestige Distribution Shift

KS test: High vs Low D = 0.356, p = 4.4e-49. However, High vs Mid D = 0.037, p = 0.56 (not significant).

**What this means:** Mid-prestige occupations are treated identically to high-prestige occupations in terms of skin tone distribution. The real divide is between the bottom tier (low prestige) and everyone else. The bias specifically targets the lowest-status jobs.

### D4. Neighborhood Distribution

KS test (gated vs slum): D = 0.439, p = 1.7e-52. Slum collapses 70.5% of all images onto MST 6 alone.

**What this means:** "Slum" doesn't just darken on average — it collapses diversity. The model produces nearly identical skin tones for everyone in a slum. This is a distinct harm: not only are slum-associated people depicted darker, they're depicted as homogeneously dark.

### D5. Setting Distribution

KS test (village vs city): D = 0.231, p = 4.0e-25. Moderate shift.

**What this means:** Village prompts shift the distribution darker, but with less force than neighborhood. Setting is a real but secondary geographic signal.

### D6. Gender Distribution

KS test (man vs woman): D = 0.063, p = 0.055 — borderline non-significant.

**What this means:** Confirms at the distribution level what the mean analysis showed: gender has essentially no effect on skin tone output. Men and women are rendered with nearly identical MST distributions.

### D7. Layer Distribution Heatmap

Percentage of images at each MST value across L0–L5 layers. Mass visibly migrates from MST 4–5 at L0 toward MST 6 at L5 as cues accumulate.

**What this means:** Visualizes the progressive darkening. At L0 (bare prompt), MST 5 is the mode. By L5 (all cues present), MST 6 dominates. The distribution doesn't just shift — the lighter tail (MST 3–4) shrinks.

### D8. Top-5 vs Bottom-5 Occupation Distributions

KS test (lightest vs darkest 5): D = 0.526, p = 5.8e-43. Rickshaw driver: 67% MST 6, 25% MST 7 — tightly compressed. Librarian: spread across MST 3–6 — much more diverse.

**What this means:** Low-prestige occupations don't just have higher mean MST; they have lower diversity. The model is most "stereotypical" about the lowest-status jobs, producing a narrow, dark-clustered distribution with little variation.

### D9. Income × Prestige Grid (2×3)

Six distribution panels. Low income + low prestige has zero images below MST 5 — the floor rises. High income + high prestige spreads across MST 3–7.

**What this means:** When income and prestige both signal low status, the model eliminates light skin tones entirely. It's not just shifting the mean; it's raising the floor of the distribution, making light-skinned depictions structurally impossible for low-status prompts.

### D10. CDF Comparisons

Cumulative distribution functions for all four major cue pairs. Neighborhood shows the largest visual CDF gap; gender shows almost none. KS test statistics annotated on each panel.

**What this means:** CDFs provide the clearest visual evidence of distributional dominance. The neighborhood curves are cleanly separated at every MST value — not just at the tails but across the entire range.

### D11. Shannon Entropy and Information Gain

Neighborhood has 3× the information gain (0.316 bits) of any other cue. Slum entropy is only 1.28 bits — compared to the overall 2.81 bits.

**What this means:** Knowing a person's neighborhood tells you more about their predicted skin tone than any other single cue. "Slum" reduces MST uncertainty by the most — the model's output becomes highly predictable (mostly MST 6) once it sees that word.

### D12. Per-Occupation Distribution Heatmap (30 × MST)

30 rows (occupations sorted lightest to darkest) × MST columns. Visual gradient from spread distributions at top (lighter, high prestige) to compressed-dark distributions at bottom (darker, low prestige).

**What this means:** A single image that encodes the entire occupational colorism gradient. Clear clustering by prestige band, with high-prestige jobs showing diverse skin tones and low-prestige jobs showing concentrated dark tones.

---

## Part 4 — Advanced Analysis (11 analyses)

### A1. OLS Regression (MST ~ 5 cues)

Using L5 data (720 rows, all 5 cues present). R² = 0.35. Standardized β coefficients: neighborhood (0.293), income (0.269), prestige (0.213), setting (0.167), gender (-0.031, not significant).

**What this means:** Five text cues explain 35% of the variance in MST. This is substantial for a single-sentence prompt. Neighborhood is the strongest individual predictor, followed by income and occupation prestige. Gender is not a significant predictor.

### A2. OLS with Interaction Terms

Income × neighborhood interaction: β = -0.25, p < 0.001 (significant, subadditive). Setting × neighborhood: β = -0.125, p = 0.035 (significant). Gender × income: not significant.

**What this means:** The interactions are subadditive — combining two "darkening" cues produces less than the sum of their individual effects. This confirms the bounded-response pattern from the interaction analysis. The model has a ceiling on how dark it goes, causing diminishing returns when multiple cues stack.

### A3. Darkening vs Lightening Asymmetry

Every cue pair shows asymmetry: the "dark" cue value darkens the output more than the "light" cue value lightens it. Multi-cue extremes: aligned-dark shifts +2.0 from baseline; aligned-light shifts +0.0. There is a whitening ceiling at the baseline.

**What this means:** The model's default (baseline) is already relatively light. Low-status cues push skin tone darker, but high-status cues cannot push it lighter than the default. The bias is one-directional: the model can darken but cannot whiten. This means the harm falls entirely on people depicted in low-status contexts.

### A4. Tail Probabilities

P(MST ≥ 7) and P(MST ≤ 4) per condition. Slum: 24.5% dark-tail, 0.3% light-tail. High income: 4.9% dark-tail, 20.4% light-tail. The tails are starkly asymmetric.

**What this means:** For slum prompts, 1 in 4 images has dark skin (MST ≥ 7) but virtually none have light skin (MST ≤ 4). For high-income prompts, the reverse: 1 in 5 has light skin, few have dark skin. The extremes of the distribution are where the bias is most visible and most harmful.

### A5. Cliff's Delta (Non-Parametric Effect Sizes)

Appropriate for ordinal MST data. Neighborhood δ = 0.538 (large), prestige δ = 0.421 (medium), income δ = 0.403 (medium), setting δ = 0.257 (small), gender δ = -0.055 (negligible).

**What this means:** Confirms the same hierarchy as Cohen's d but using a method that doesn't assume interval-level data. Since MST is ordinal (the "distance" between MST 3 and 4 isn't necessarily the same as between 7 and 8), Cliff's delta is actually the more appropriate measure. The results are consistent.

### A6. Variance Analysis

SD per condition. Slum has the lowest SD (0.644); gated community has SD = 1.20. Low-prestige SD < High-prestige SD.

**What this means:** The model is most "certain" (least variable) about skin tone when it generates for low-status contexts. Slum prompts produce a tight cluster of similar skin tones; gated community prompts produce more diversity. The bias isn't just about the mean — it's about the model being more rigidly stereotypical for disadvantaged groups.

### A7. Sequential Effect Decomposition (Waterfall)

Layer-by-layer contribution to the total L0→L5 shift: Occupation contributes +91% of total darkening, Setting +68%, Neighborhood +28%, Income +83%, Gender -83% (lightens relative to subsequent cue darkening).

**What this means:** The waterfall shows how each cue's marginal contribution adds up. Occupation and income are the largest sequential contributors to darkening. Gender actually lightens slightly when added (because specifying "man" or "woman" produces slightly lighter output than the generic "person"), but this is swamped by subsequent darkening from other cues.

### A8. Marginal vs Conditional Effects

Income effect in different contexts: Δ = 1.0 when income is the only cue (S3), Δ = 0.54 when all 5 cues are present (L5). Setting effect similarly shrinks from 1.0 (alone) to 0.4 (full stack).

**What this means:** Cue effects are context-dependent. Each cue has a weaker marginal effect when other cues are already present, because the cues share overlapping signal. This is consistent with the subadditive interactions found in A2.

### A9. Disparate Impact Ratios

At MST ≥ 6 threshold: slum vs gated = 1.86×, low vs high income = 1.97×, low vs high prestige = 1.85×. All violate the 4/5 rule (threshold: ratio > 1.25 or < 0.80). At MST ≥ 7: slum vs gated reaches 7.35×.

**What this means:** Using the EEOC's four-fifths rule, the model's skin tone assignment constitutes adverse impact for every socioeconomic cue tested. Low-income prompts are nearly twice as likely to produce dark skin as high-income prompts. At more extreme thresholds, the disparity becomes even more severe — slum prompts are 7× more likely to produce MST ≥ 7 than gated community prompts.

### A10. Three-Way Interaction (Gender × Income × Prestige)

Income gap is approximately 0.57 MST across all 6 gender × prestige cells. Gender does not modulate the income × prestige interaction.

**What this means:** The income bias is remarkably stable. It doesn't matter whether the person is a man or woman, or whether their occupation is high or low prestige — saying "low income" adds roughly the same amount of darkening. Gender plays no moderating role, confirming it is orthogonal to the colorism mechanism.

### A11. Chi-Squared and Cramér's V

Neighborhood: χ² = 551.8, V = 0.392, p ≈ 0 (strongest). Income: χ² = 345.5, V = 0.387. Prestige: χ² = 349.0, V = 0.259. Setting: χ² = 126.9, V = 0.245. Gender: χ² = 13.9, V = 0.088, p = 0.03 (weakest, barely significant).

**What this means:** The association between neighborhood and MST distribution is the strongest of all cues (V = 0.392). Income is nearly as strong. Gender barely reaches statistical significance and has a negligible association strength. This is the final confirmation of the cue hierarchy.

---

## Summary of Key Findings

**The colorism hierarchy:** Neighborhood (d = 1.16) > Occupation prestige (d = 0.80) > Income (d = 0.76) > Setting (d = 0.47) >> Gender (d = -0.08). Socioeconomic and geographic cues drive colorism; gender does not.

**The asymmetry finding:** The model darkens easily (baseline 5.0 → 7.0 for aligned-dark) but cannot lighten (baseline 5.0 → 5.0 for aligned-light). The harm is one-directional.

**The compounding finding:** Two-cue compounding reaches Δ = 3.0 (low income + slum vs high income + gated). Context cues overpower occupation — a surgeon in a slum (MST 9) is darker than a maximally-dark-aligned prompt (MST 7).

**The diversity collapse finding:** Low-status cues don't just shift the mean; they collapse the distribution. Slum concentrates 70.5% of images at a single MST value. The model is most rigidly stereotypical about the most disadvantaged groups.

**The fairness finding:** Every socioeconomic cue violates the EEOC four-fifths rule for disparate impact. At extreme thresholds, disparities reach 7×.

**The null result:** Gender is not a driver of colorism in this model. This is substantively important — the bias is socioeconomic, not gendered.

**Five text cues explain 35% of MST variance.** A single-sentence prompt controls over a third of the model's skin tone output through learned stereotypical associations between socioeconomic status and skin color.
