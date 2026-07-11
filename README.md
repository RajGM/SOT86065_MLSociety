# Colorism in Text-to-Image Generation

Investigates colorism bias in AI image generators by systematically varying socioeconomic cues in prompts and measuring skin tone shifts using the Monk Skin Tone (MST) scale.

**Models tested:** GPT Image 1 Mini (OpenAI) and Grok Imagine Image (xAI), audited by GPT-5-mini.

## Quick Start

### Prerequisites

```bash
npm install openai dotenv node-fetch
pip install pandas matplotlib seaborn scipy numpy
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
XAI_API_KEY=xai-...
```

### Run the Pipeline

```bash
# Smoke test — first 10 prompts, both providers, full pipeline
node index.js --test

# Full production run — all 2,659 prompts
node index.js --prod
```

### Options

```bash
node index.js --test                     # 10 prompts, both providers
node index.js --prod                     # All prompts, both providers
node index.js --test --provider openai   # 10 prompts, OpenAI only
node index.js --prod --provider xai      # All prompts, xAI only
node index.js --prod --skip-analysis     # Skip Python analysis (stage 5)
```

### Run Stages Individually

```bash
node 01_generate_prompts.js          # → prompts_5cue_matrix.csv (2,659 prompts)
node 02_build_generation_jobs.js     # → generation_jobs.json
node 03_gpt_run_generation.js        # → gpt-image-1-mini/ + manifest CSV
node 03_grok_run_generation.js       # → grok-imagine-image/ + manifest CSV
node 04_audit.js --provider openai   # → mst_audit_gpt-image-1-mini.csv
node 04_audit.js --provider xai      # → mst_audit_grok-imagine-image.csv
python 05_run_full_analysis.py       # → figures_all/ + summary_stats.csv
```

All generation and audit scripts are **resumable** — interrupted runs pick up where they left off.

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 1 | `01_generate_prompts.js` | `prompts_5cue_matrix.csv` — prompt matrix |
| 2 | `02_build_generation_jobs.js` | `generation_jobs.json` — structured job list |
| 3 | `03_gpt_run_generation.js` / `03_grok_run_generation.js` | PNG images + generation manifests |
| 4 | `04_audit.js` | `mst_audit_*.csv` — MST skin tone scores per image |
| 5 | `05_run_full_analysis.py` | 45 statistical analyses + figures |

The `index.js` orchestrator runs all five stages in sequence. In `--test` mode it truncates `generation_jobs.json` to the first 10 prompts, runs the pipeline, then restores the full file automatically.

## Experimental Design

Five cue dimensions varied across 2,659 prompts in six experimental sections:

- **Gender:** man, woman, neutral ("person")
- **Occupation:** 30 occupations across high/mid/low prestige bands (10 each)
- **Income:** high, low
- **Setting:** village, city
- **Neighborhood:** gated community, suburb, public housing project

Sections: incremental L-track (gendered), N-track (neutral), single-cue isolations (S1–S5), two-cue interactions (X1–X10), multi-cue stacks (C1–C10), and order-effect tests (4 alternative orderings).

## Key Findings

Income is the strongest colorism driver in GPT (d=1.20); neighborhood is strongest in Grok (d=1.16). Both models systematically assign darker skin tones to low-status socioeconomic cues. Gender has negligible effect in both (d<0.2). Five text cues explain 35–43% of skin tone variance. Every socioeconomic comparison violates the EEOC 4/5 rule for disparate impact. Prompt order does not matter — the bias is semantic, not syntactic.

## Documentation

- `PROJECT_DOCUMENTATION.md` — Full technical documentation (design, pipeline, all 45 analyses)
- `gpt_audit_analysis.md` — Detailed GPT analysis results with interpretation
- `grok_audit_analysis.md` — Detailed Grok analysis results with interpretation

## File Structure

```
├── index.js                          # Pipeline orchestrator (--test / --prod)
├── 01_generate_prompts.js            # Prompt matrix generator
├── 02_build_generation_jobs.js       # CSV → JSON converter
├── 03_gpt_run_generation.js          # OpenAI image generation (resumable)
├── 03_grok_run_generation.js         # xAI image generation (resumable)
├── 04_audit.js                       # MST skin tone auditor (resumable)
├── 05_run_full_analysis.py           # 45-analysis statistical suite
├── prompts_5cue_matrix.csv           # Generated prompt matrix
├── gpt-image-1-mini.csv              # GPT generation manifest
├── grok-imagine-image.csv            # Grok generation manifest
├── mst_audit_gpt-image-1-mini.csv    # GPT MST audit results
├── mst_audit_grok-imagine-image.csv  # Grok MST audit results
├── gpt_audit_analysis.md             # GPT analysis write-up
├── grok_audit_analysis.md            # Grok analysis write-up
└── PROJECT_DOCUMENTATION.md          # Full technical documentation
```
