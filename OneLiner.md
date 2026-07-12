**MST Scale** — A 10-point skin tone ruler from lightest (1) to darkest (10). In our study, it reveals that both models cluster generated faces around MST 4–6, almost never producing very light (1–2) or very dark (9–10) skin — a diversity collapse.

**MAE** — Average distance between the auditor's prediction and the true skin tone label. Ours is under 1 MST step, confirming the auditor is reliable enough that all downstream findings reflect real model behavior, not measurement noise.

**Acc₃** — Percentage of predictions within ±3 steps of truth. At 81%, it tells us the auditor never catastrophically misjudges skin tone — a light face won't be scored as dark or vice versa, so our bias measurements aren't artifacts.

**Cohen's κ** — Agreement score that strips out lucky guesses. At 0.714, it confirms our automated auditor agrees with human experts as much as humans agree with each other — so using it at scale is justified.

**Zeros** — Images the auditor couldn't score. Tracking them reveals no systematic dropout by skin tone — the auditor isn't silently ignoring dark or light faces, which would create a hidden selection bias in our results.

**Cohen's d** — How many standard deviations apart two groups' skin tones are. It reveals that saying "high income" vs "low income" shifts GPT's output by 1.20 SDs — a larger effect than most interventions measured in social science. The bias isn't subtle; it's massive.

**R²** — Fraction of skin tone variation explained by the five cues. At 0.35–0.43, it reveals that prompt words alone account for nearly half of all skin tone decisions — the models aren't generating diverse faces and then nudging them; the cues are deterministic drivers.

**ANOVA F & p** — Tests whether rearranging cue order changes the output. F < 0.6 and p > 0.68 reveal it doesn't — you can put "income" first or last and get identical bias. The models parse meaning, not position, so this can't be fixed by reordering prompts.

**Δ (Interaction Gaps)** — Skin tone range across all conditions of a cue pair. Grok's Δ = 3.0 reveals that combining "low income" + "slum" vs "high income" + "gated community" produces a 3-step skin tone gap — visually, the difference between a light-beige and medium-brown face from the same prompt template.

**Subadditive** — Combined cue effect is less than the sum of individual effects. This reveals the models have an internal ceiling — once a prompt is "dark enough," adding more poverty cues doesn't push darker. Bias saturates rather than compounds linearly.

**EEOC 4/5 Rule** — Legal test: if one group is selected less than 80% as often as another, it's discriminatory. Applied to our data, every socioeconomic cue fails this test — meaning if these generators were used for hiring visuals or ad targeting, they'd constitute actionable discrimination under U.S. law.

**Pearson's r** — Measures whether the relationship between cue and skin tone is a clean gradient or just a binary jump. It reveals the bias is graded — it's not just "rich = light, poor = dark" as a switch; the shift scales proportionally with how many socioeconomic signals you stack.