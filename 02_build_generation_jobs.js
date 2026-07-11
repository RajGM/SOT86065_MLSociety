const fs = require("fs");
const path = require("path");

const CSV_PATH = path.join(__dirname, "prompts_5cue_matrix.csv");
const OUT_PATH = path.join(__dirname, "generation_jobs.json");

// ═══════════════════════════════════════════════════════════════
// CSV PARSER (handles quoted fields with commas)
// ═══════════════════════════════════════════════════════════════

function parseCSVLine(line) {
  const fields = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        fields.push(current);
        current = "";
      } else {
        current += ch;
      }
    }
  }
  fields.push(current);
  return fields;
}

// ═══════════════════════════════════════════════════════════════
// READ CSV & BUILD JOBS
// ═══════════════════════════════════════════════════════════════

const csv = fs.readFileSync(CSV_PATH, "utf8");
const lines = csv.split("\n").filter((l) => l.trim());
const headers = parseCSVLine(lines[0]);

const jobs = [];

for (let i = 1; i < lines.length; i++) {
  const fields = parseCSVLine(lines[i]);
  const row = {};
  headers.forEach((h, idx) => {
    row[h] = fields[idx] || "";
  });

  jobs.push({
    prompt_id: row.prompt_id,
    section: row.section,
    layer_id: row.layer_id,
    cues: row.cues,
    gender: row.gender || null,
    occupation: row.occupation || null,
    income: row.income || null,
    setting: row.setting || null,
    neighborhood: row.neighborhood || null,
    prompt: row.prompt,
  });
}

fs.writeFileSync(OUT_PATH, JSON.stringify(jobs, null, 2));
console.log(`${jobs.length} jobs written to ${OUT_PATH}`);
