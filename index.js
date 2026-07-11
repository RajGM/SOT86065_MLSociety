#!/usr/bin/env node

/**
 * index.js — Pipeline orchestrator
 * ==================================
 * Runs the full colorism experiment pipeline end-to-end.
 *
 * Usage:
 *   node index.js --test          # First 10 prompts only (smoke test)
 *   node index.js --prod          # Full pipeline (all ~2,659 prompts)
 *   node index.js --test --skip-analysis   # Skip the Python analysis step
 *   node index.js --prod --provider openai # Only run one provider
 *
 * Pipeline stages:
 *   1. Generate prompt matrix        (01_generate_prompts.js)
 *   2. Build generation jobs JSON    (02_build_generation_jobs.js)
 *   3. Generate images via APIs      (03_gpt_run_generation.js + 03_grok_run_generation.js)
 *   4. Audit skin tones via GPT-5    (04_audit.js)
 *   5. Run statistical analysis      (05_run_full_analysis.py)
 */

const { execSync } = require("child_process");
const fs = require("fs");
const path = require("path");

// ═══════════════════════════════════════════════════════════════
// CLI PARSING
// ═══════════════════════════════════════════════════════════════

const args = process.argv.slice(2);

const MODE_TEST = args.includes("--test");
const MODE_PROD = args.includes("--prod");
const SKIP_ANALYSIS = args.includes("--skip-analysis");
const PROVIDER_FLAG = (() => {
  const idx = args.indexOf("--provider");
  return idx !== -1 && args[idx + 1] ? args[idx + 1] : null;
})();

if (!MODE_TEST && !MODE_PROD) {
  console.error(`
Usage:
  node index.js --test              Run pipeline with first 10 prompts (smoke test)
  node index.js --prod              Run full pipeline (all prompts)

Options:
  --skip-analysis                   Skip the Python analysis step (stage 5)
  --provider <openai|xai>           Only run one provider instead of both
`);
  process.exit(1);
}

if (MODE_TEST && MODE_PROD) {
  console.error("Error: specify either --test or --prod, not both.");
  process.exit(1);
}

const PROMPT_LIMIT = MODE_TEST ? 10 : null;
const PROVIDERS = PROVIDER_FLAG ? [PROVIDER_FLAG] : ["openai", "xai"];
const LABEL = MODE_TEST ? "TEST (10 prompts)" : "PRODUCTION (all prompts)";

// ═══════════════════════════════════════════════════════════════
// PATHS
// ═══════════════════════════════════════════════════════════════

const ROOT = __dirname;
const PROMPTS_CSV = path.join(ROOT, "prompts_5cue_matrix.csv");
const JOBS_JSON = path.join(ROOT, "generation_jobs.json");

// ═══════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════

function banner(stage, title) {
  const line = "═".repeat(60);
  console.log(`\n${line}`);
  console.log(`  STAGE ${stage}: ${title}`);
  console.log(`  Mode: ${LABEL}`);
  console.log(line);
}

function run(cmd, label) {
  console.log(`\n  ▶ ${label}`);
  console.log(`    $ ${cmd}\n`);
  try {
    execSync(cmd, { cwd: ROOT, stdio: "inherit", env: { ...process.env } });
  } catch (err) {
    console.error(`\n  ✗ ${label} failed (exit code ${err.status})`);
    process.exit(err.status || 1);
  }
  console.log(`  ✓ ${label} complete`);
}

function elapsed(startMs) {
  const s = ((Date.now() - startMs) / 1000).toFixed(1);
  return s >= 60 ? `${(s / 60).toFixed(1)}m` : `${s}s`;
}

/**
 * Truncate generation_jobs.json to PROMPT_LIMIT entries (test mode).
 * Keeps the original as generation_jobs_full.json for restore.
 */
function truncateJobs(limit) {
  if (!limit) return;
  if (!fs.existsSync(JOBS_JSON)) {
    console.error("  ✗ generation_jobs.json not found — cannot truncate.");
    process.exit(1);
  }

  const jobs = JSON.parse(fs.readFileSync(JOBS_JSON, "utf8"));
  const fullBackup = path.join(ROOT, "generation_jobs_full.json");

  // Back up full file if not already backed up
  if (!fs.existsSync(fullBackup)) {
    fs.copyFileSync(JOBS_JSON, fullBackup);
  }

  const truncated = jobs.slice(0, limit);
  fs.writeFileSync(JOBS_JSON, JSON.stringify(truncated, null, 2));
  console.log(`  ℹ Truncated generation_jobs.json: ${jobs.length} → ${truncated.length} jobs`);
}

/**
 * Restore full generation_jobs.json from backup (after test run).
 */
function restoreJobs() {
  const fullBackup = path.join(ROOT, "generation_jobs_full.json");
  if (fs.existsSync(fullBackup)) {
    fs.copyFileSync(fullBackup, JOBS_JSON);
    fs.unlinkSync(fullBackup);
    console.log("  ℹ Restored full generation_jobs.json from backup");
  }
}

// ═══════════════════════════════════════════════════════════════
// PIPELINE
// ═══════════════════════════════════════════════════════════════

async function main() {
  const t0 = Date.now();

  console.log("\n" + "═".repeat(60));
  console.log("  COLORISM EXPERIMENT PIPELINE");
  console.log(`  Mode:      ${LABEL}`);
  console.log(`  Providers: ${PROVIDERS.join(", ")}`);
  console.log(`  Analysis:  ${SKIP_ANALYSIS ? "SKIPPED" : "enabled"}`);
  console.log("═".repeat(60));

  // ─── Stage 1: Generate prompts ───
  banner(1, "GENERATE PROMPT MATRIX");
  run("node 01_generate_prompts.js", "Generate prompts → prompts_5cue_matrix.csv");

  // ─── Stage 2: Build jobs ───
  banner(2, "BUILD GENERATION JOBS");
  run("node 02_build_generation_jobs.js", "Build jobs → generation_jobs.json");

  // Truncate for test mode
  if (PROMPT_LIMIT) {
    truncateJobs(PROMPT_LIMIT);
  }

  // ─── Stage 3: Generate images ───
  banner(3, "GENERATE IMAGES");
  for (const provider of PROVIDERS) {
    const script = provider === "openai"
      ? "03_gpt_run_generation.js"
      : "03_grok_run_generation.js";
    run(`node ${script} --no-resume`, `Generate images (${provider})`);
  }

  // ─── Stage 4: Audit skin tones ───
  banner(4, "MST SKIN TONE AUDIT");
  for (const provider of PROVIDERS) {
    run(`node 04_audit.js --provider ${provider}`, `Audit MST scores (${provider})`);
  }

  // ─── Stage 5: Analysis ───
  if (!SKIP_ANALYSIS) {
    banner(5, "STATISTICAL ANALYSIS");
    run("python 05_run_full_analysis.py", "Run 45 analyses → figures_all/");
  } else {
    console.log("\n  ⏭ Stage 5 skipped (--skip-analysis)");
  }

  // Restore full jobs file if we truncated
  if (PROMPT_LIMIT) {
    restoreJobs();
  }

  // ─── Done ───
  const totalTime = elapsed(t0);
  console.log("\n" + "═".repeat(60));
  console.log("  PIPELINE COMPLETE");
  console.log(`  Mode:      ${LABEL}`);
  console.log(`  Providers: ${PROVIDERS.join(", ")}`);
  console.log(`  Time:      ${totalTime}`);
  console.log("═".repeat(60) + "\n");
}

main().catch((err) => {
  console.error("Pipeline fatal error:", err);
  // Restore jobs on crash too
  restoreJobs();
  process.exit(1);
});
