// run_generation.js
// Resumable image generation for incremental prompt decomposition experiment.
// Reads generation_jobs.json, calls OpenAI or xAI (both OpenAI-compatible),
// saves images, tracks state in a durable model JSON.
//
// Usage:
//   node run_generation.js                  # resume from last state
//   node run_generation.js --rebuild        # rebuild task list from scratch
//   node run_generation.js --no-resume      # ignore previous state, start fresh
//
// IMPORTANT: Set API keys in .env file. Do NOT hardcode keys.

require("dotenv").config();
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const fetch = global.fetch || require("node-fetch");
const OpenAI = require("openai");

/* ═══════════════════════════════════════════════════════════════
   CONFIG — edit these values
   ═══════════════════════════════════════════════════════════════ */

const PROVIDER = "openai"; // "openai" or "xai"

const PROVIDER_CONFIG = {
  openai: {
    model: "gpt-image-1-mini",
    baseURL: "https://api.openai.com/v1",
    envKey: "OPENAI_API_KEY",
    responseFormat: "url", // gpt-image returns url
  },
  xai: {
    model: "grok-imagine-image",
    baseURL: "https://api.x.ai/v1",
    envKey: "XAI_API_KEY",
    responseFormat: "url",
  },
};

const CONFIG = PROVIDER_CONFIG[PROVIDER];
const MODEL = CONFIG.model;

const OUTDIR = path.resolve(`./${MODEL}`);
const MANIFEST = path.resolve(`./${MODEL}.csv`);
const MODEL_JSON = path.resolve(`./${MODEL}.json`);
const SECTION_LOG = path.resolve(`./${MODEL}_by_section.json`);
const JOBS_PATH = path.resolve("./generation_jobs.json");

const MAX_CONCURRENT = 15;
const MAX_RETRIES = 5;
const INITIAL_BACKOFF_MS = 500;
const RATE_LIMIT = 120;          // max calls per minute
const RATE_WINDOW_MS = 60_000;

/* ═══════════════════════════════════════════════════════════════
   API KEY CHECK
   ═══════════════════════════════════════════════════════════════ */

const API_KEY = process.env[CONFIG.envKey] || "";
if (!API_KEY) {
  console.error(`❌ ERROR: set ${CONFIG.envKey} in your .env file or environment`);
  process.exit(1);
}

/* ═══════════════════════════════════════════════════════════════
   CLIENT (OpenAI-compatible for both providers)
   ═══════════════════════════════════════════════════════════════ */

const client = new OpenAI({
  apiKey: API_KEY,
  baseURL: CONFIG.baseURL,
});

/* ═══════════════════════════════════════════════════════════════
   RATE LIMITER — sliding window
   ═══════════════════════════════════════════════════════════════ */

const rateBucket = [];

async function waitForRateSlot() {
  while (true) {
    const now = Date.now();
    while (rateBucket.length > 0 && now - rateBucket[0] >= RATE_WINDOW_MS) {
      rateBucket.shift();
    }
    if (rateBucket.length < RATE_LIMIT) {
      rateBucket.push(now);
      return;
    }
    const waitMs = RATE_WINDOW_MS - (now - rateBucket[0]) + 50;
    console.log(`⏳ Rate limit reached, waiting ${waitMs}ms`);
    await sleep(waitMs);
  }
}

/* ═══════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════ */

function sanitizeName(s) {
  return String(s)
    .replace(/\s+/g, "_")
    .replace(/[^a-zA-Z0-9_\-]/g, "")
    .replace(/_+/g, "_")
    .toLowerCase();
}

function nowISO() {
  return new Date().toISOString();
}

function csvEscape(s) {
  if (s == null) return "";
  const str = String(s);
  return /[,"\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function atomicWriteJson(filePath, obj) {
  const tmp = `${filePath}.tmp-${process.pid}-${crypto.randomBytes(6).toString("hex")}`;
  fs.writeFileSync(tmp, JSON.stringify(obj, null, 2));
  fs.renameSync(tmp, filePath);
}

/* ═══════════════════════════════════════════════════════════════
   TASK GENERATION — reads generation_jobs.json
   ═══════════════════════════════════════════════════════════════ */

function buildFilename(promptId) {
  return `${sanitizeName(MODEL)}_${promptId}.png`;
}

function generateAllTasks() {
  const jobs = JSON.parse(fs.readFileSync(JOBS_PATH, "utf8"));
  const tasks = [];

  for (const job of jobs) {
    const id = `${job.prompt_id}__${PROVIDER}`;
    tasks.push({
      id,
      prompt_id: job.prompt_id,
      model: MODEL,
      provider: PROVIDER,
      section: job.section,
      layer_id: job.layer_id,
      cues: job.cues,
      gender: job.gender,
      occupation: job.occupation,
      income: job.income,
      setting: job.setting,
      neighborhood: job.neighborhood,
      prompt: job.prompt,
      filename: buildFilename(job.prompt_id),
      attempts: 0,
      state: "pending",
      lastError: null,
      responseStatus: null,
      timestamps: { created: nowISO(), started: null, finished: null },
    });
  }

  return tasks;
}

/* ═══════════════════════════════════════════════════════════════
   STORAGE INIT
   ═══════════════════════════════════════════════════════════════ */

function ensureOutDir() {
  if (!fs.existsSync(OUTDIR)) fs.mkdirSync(OUTDIR, { recursive: true });
  if (!fs.existsSync(MANIFEST)) {
    fs.writeFileSync(
      MANIFEST,
      "filename,prompt_id,section,layer_id,cues,occupation,income,model,provider,response_url,timestamp,ok,error\n"
    );
  }
}

function loadOrCreateModelJson(rebuild = false) {
  if (rebuild || !fs.existsSync(MODEL_JSON)) {
    const tasks = generateAllTasks();
    const wrapper = {
      meta: { model: MODEL, provider: PROVIDER, created: nowISO(), totalTasks: tasks.length },
      tasks,
    };
    atomicWriteJson(MODEL_JSON, wrapper);
    writeSectionLog(wrapper);
    console.log(`📋 Created ${MODEL_JSON} with ${tasks.length} tasks`);
    return wrapper;
  }

  const raw = fs.readFileSync(MODEL_JSON, "utf8");
  try {
    const parsed = JSON.parse(raw);

    // Merge any new tasks from jobs file
    const expected = generateAllTasks();
    const map = new Map(parsed.tasks.map((t) => [t.id, t]));
    let added = 0;
    for (const t of expected) {
      if (!map.has(t.id)) {
        map.set(t.id, t);
        added++;
      }
    }

    if (added > 0) {
      const merged = {
        meta: { model: MODEL, provider: PROVIDER, loaded: nowISO(), totalTasks: map.size },
        tasks: Array.from(map.values()),
      };
      atomicWriteJson(MODEL_JSON, merged);
      writeSectionLog(merged);
      console.log(`🔄 Merged ${added} new tasks into ${MODEL_JSON}`);
      return merged;
    }

    return parsed;
  } catch (err) {
    console.error("Failed to parse model.json. Use --rebuild to recreate.", err);
    process.exit(1);
  }
}

/* ═══════════════════════════════════════════════════════════════
   API CALL
   ═══════════════════════════════════════════════════════════════ */

async function callImageAPI(prompt) {
  try {
    const response = await client.images.generate({
      model: MODEL,
      prompt,
      n: 1,
      size: "1024x1024",
    });

    const imageUrl = response?.data?.[0]?.url || null;
    const b64 = response?.data?.[0]?.b64_json || null;
    const responseId = response?.id || null;

    return { ok: true, httpStatus: 200, imageUrl, b64, responseId };
  } catch (err) {
    const message = err?.message || String(err);
    const httpStatus = err?.status || err?.response?.status || null;
    return { ok: false, httpStatus, error: message };
  }
}

async function downloadAndSaveImage(imageStringOrUrl, outpath) {
  // Base64 data URI
  if (typeof imageStringOrUrl === "string" && imageStringOrUrl.startsWith("data:image")) {
    const base64 = imageStringOrUrl.replace(/^data:image\/\w+;base64,/, "");
    fs.writeFileSync(outpath, Buffer.from(base64, "base64"));
    return;
  }
  // Raw base64 string (no prefix)
  if (typeof imageStringOrUrl === "string" && !imageStringOrUrl.startsWith("http")) {
    fs.writeFileSync(outpath, Buffer.from(imageStringOrUrl, "base64"));
    return;
  }
  // URL
  if (typeof imageStringOrUrl === "string" && /^https?:\/\//i.test(imageStringOrUrl)) {
    const r = await fetch(imageStringOrUrl);
    if (!r.ok) throw new Error(`Download failed HTTP ${r.status}`);
    const ab = await r.arrayBuffer();
    fs.writeFileSync(outpath, Buffer.from(ab));
    return;
  }
  throw new Error("Unknown image format");
}

/* ═══════════════════════════════════════════════════════════════
   SECTION LOG — per-section summary (like occupation log in EACL)
   ═══════════════════════════════════════════════════════════════ */

function buildSectionLog(wrapper) {
  const bySection = {};

  for (const t of wrapper.tasks) {
    if (!bySection[t.section]) {
      bySection[t.section] = {
        summary: { pending: 0, in_progress: 0, success: 0, failed: 0, totalAttempts: 0 },
        layers: {},
      };
    }

    const sec = bySection[t.section];
    sec.summary.totalAttempts += t.attempts || 0;

    if (t.state === "pending") sec.summary.pending++;
    else if (t.state === "in-progress") sec.summary.in_progress++;
    else if (t.state === "success") sec.summary.success++;
    else if (t.state === "failed") sec.summary.failed++;

    // Track by layer
    if (!sec.layers[t.layer_id]) sec.layers[t.layer_id] = { success: 0, failed: 0, pending: 0 };
    sec.layers[t.layer_id][t.state === "success" ? "success" : t.state === "failed" ? "failed" : "pending"]++;
  }

  return {
    meta: { model: MODEL, provider: PROVIDER, generated: nowISO(), sections: Object.keys(bySection).length },
    bySection,
  };
}

function writeSectionLog(wrapper) {
  atomicWriteJson(SECTION_LOG, buildSectionLog(wrapper));
}

/* ═══════════════════════════════════════════════════════════════
   TASK STATE — in-memory updates + durable persistence
   ═══════════════════════════════════════════════════════════════ */

function updateTask(wrapper, id, patch) {
  const idx = wrapper.tasks.findIndex((t) => t.id === id);
  if (idx === -1) return false;
  wrapper.tasks[idx] = { ...wrapper.tasks[idx], ...patch };
  return true;
}

function persist(wrapper) {
  atomicWriteJson(MODEL_JSON, wrapper);
  writeSectionLog(wrapper);
}

/* ═══════════════════════════════════════════════════════════════
   SINGLE TASK RUNNER — with retries + exponential backoff
   ═══════════════════════════════════════════════════════════════ */

async function generateImageForTask(wrapper, task) {
  if (task.state === "success") return { skipped: true };

  // Mark in-progress
  updateTask(wrapper, task.id, {
    state: "in-progress",
    timestamps: { ...task.timestamps, started: nowISO() },
  });
  persist(wrapper);

  let lastErr = null;

  for (let attempt = task.attempts || 0; attempt < MAX_RETRIES + 1; attempt++) {
    try {
      updateTask(wrapper, task.id, { attempts: attempt + 1 });
      persist(wrapper);

      // Rate limit
      await waitForRateSlot();

      // API call
      const res = await callImageAPI(task.prompt);

      if (!res.ok) throw new Error(res.error || `API error HTTP ${res.httpStatus}`);

      // Get image data (URL or base64)
      const imageData = res.imageUrl || res.b64;
      if (!imageData) throw new Error("No image data returned by API");

      // Save image
      ensureOutDir();
      const outpath = path.join(OUTDIR, task.filename);
      await downloadAndSaveImage(imageData, outpath);

      // Mark success
      const now = nowISO();
      updateTask(wrapper, task.id, {
        state: "success",
        responseStatus: { httpStatus: res.httpStatus || 200, responseId: res.responseId, url: res.imageUrl },
        lastError: null,
        timestamps: { ...task.timestamps, finished: now },
      });
      persist(wrapper);

      // CSV log
      const line = [
        task.filename,
        task.prompt_id,
        task.section,
        task.layer_id,
        csvEscape(task.cues),
        csvEscape(task.occupation),
        task.income,
        MODEL,
        PROVIDER,
        res.imageUrl || "",
        now,
        "1",
        "",
      ].join(",") + "\n";
      fs.appendFileSync(MANIFEST, line);

      console.log(`💾 ${task.filename} (${task.id})`);
      return { ok: true };
    } catch (err) {
      lastErr = err.message || String(err);
      console.warn(`⚠️  Attempt ${attempt + 1}/${MAX_RETRIES + 1} failed for ${task.id}: ${lastErr}`);

      updateTask(wrapper, task.id, { lastError: lastErr, attempts: attempt + 1 });
      persist(wrapper);

      if (attempt < MAX_RETRIES) {
        const delay = INITIAL_BACKOFF_MS * Math.pow(2, attempt);
        console.log(`   Retrying after ${delay}ms...`);
        await sleep(delay);
      } else {
        const now = nowISO();
        updateTask(wrapper, task.id, {
          state: "failed",
          lastError: lastErr,
          timestamps: { ...task.timestamps, finished: now },
        });
        persist(wrapper);

        const line = [
          "",
          task.prompt_id,
          task.section,
          task.layer_id,
          csvEscape(task.cues),
          csvEscape(task.occupation),
          task.income,
          MODEL,
          PROVIDER,
          "",
          now,
          "0",
          csvEscape(lastErr),
        ].join(",") + "\n";
        fs.appendFileSync(MANIFEST, line);

        console.error(`❌ ${task.id} failed permanently: ${lastErr}`);
        return { ok: false, error: lastErr };
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════
   CONCURRENCY RUNNER — Promise.race pool
   ═══════════════════════════════════════════════════════════════ */

async function runAll(resume = true) {
  ensureOutDir();
  const wrapper = loadOrCreateModelJson(!resume);

  const toRun = wrapper.tasks.filter((t) => t.state !== "success");
  const alreadyDone = wrapper.tasks.length - toRun.length;

  console.log(`\n${"=".repeat(55)}`);
  console.log(`🚀 GENERATION RUN — ${PROVIDER} / ${MODEL}`);
  console.log(`${"=".repeat(55)}`);
  console.log(`   Total tasks:  ${wrapper.tasks.length}`);
  console.log(`   Already done: ${alreadyDone}`);
  console.log(`   To process:   ${toRun.length}`);
  console.log(`   Concurrency:  ${MAX_CONCURRENT}`);
  console.log(`   Rate limit:   ${RATE_LIMIT}/min`);
  console.log(`   Max retries:  ${MAX_RETRIES}`);
  console.log(`   Output dir:   ${OUTDIR}`);
  console.log(`${"=".repeat(55)}\n`);

  if (toRun.length === 0) {
    console.log("✅ All tasks already completed. Nothing to do.");
    return;
  }

  let i = 0;
  const inFlight = new Set();

  async function startTask(task) {
    const p = generateImageForTask(wrapper, task)
      .catch((e) => console.error("Task fatal error:", e.message || e))
      .finally(() => inFlight.delete(p));
    inFlight.add(p);
  }

  while (i < toRun.length || inFlight.size > 0) {
    while (i < toRun.length && inFlight.size < MAX_CONCURRENT) {
      startTask(toRun[i++]);
    }
    if (inFlight.size > 0) await Promise.race(Array.from(inFlight));
  }

  // Final persist
  persist(wrapper);

  // Summary
  const done = wrapper.tasks.filter((t) => t.state === "success").length;
  const failed = wrapper.tasks.filter((t) => t.state === "failed").length;

  console.log(`\n${"=".repeat(55)}`);
  console.log(`✅ RUN COMPLETE`);
  console.log(`   Success: ${done}`);
  console.log(`   Failed:  ${failed}`);
  console.log(`   Total:   ${wrapper.tasks.length}`);
  console.log(`${"=".repeat(55)}`);
}

/* ═══════════════════════════════════════════════════════════════
   CLI & SIGNALS
   ═══════════════════════════════════════════════════════════════ */

const args = process.argv.slice(2);
const rebuild = args.includes("--rebuild");
const resume = !args.includes("--no-resume");

process.on("SIGINT", () => {
  console.log("\n⚠️  SIGINT — saving state and exiting.");
  try {
    if (fs.existsSync(MODEL_JSON)) {
      console.log(`   model.json last saved: ${fs.statSync(MODEL_JSON).mtime.toISOString()}`);
    }
    if (fs.existsSync(SECTION_LOG)) {
      console.log(`   section log last saved: ${fs.statSync(SECTION_LOG).mtime.toISOString()}`);
    }
  } catch (e) {
    console.error("Error during SIGINT save:", e);
  }
  process.exit(0);
});

process.on("SIGTERM", () => {
  console.log("\n⚠️  SIGTERM — saving state and exiting.");
  process.exit(0);
});

(async () => {
  try {
    await runAll(resume && !rebuild);
  } catch (err) {
    console.error("Fatal error:", err);
    process.exit(1);
  }
})();
