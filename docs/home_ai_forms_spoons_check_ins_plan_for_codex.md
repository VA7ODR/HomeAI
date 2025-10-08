# HomeAI Forms: Spoons Check‑ins (Plan for Codex)

This document proposes a **forms** system for HomeAI, beginning with a hard‑coded **Spoons Check‑in** and evolving to fully **data‑driven JSON forms**. It covers UX, storage, message wire‑format, parsing, indexing, analysis, and migration strategy. Deliverables are broken into clear phases so Codex can implement incrementally.

---

## Goals

> **Phase ordering note:** The project phases are intentionally sequential as **1 → 2 → 3 → 4 → 4.5 → 5 → 6 → 7**. Phase **4.5** (views for forms) sits between analytics (4) and APIs/filters (5).
1. **Fast input**: sliders + text fields for a daily Spoons check‑in.
2. **Consistent structure**: machine‑readable payloads for trend analysis.
3. **Composable prompts**: per‑form instructions (“how to respond”).
4. **Message interoperability**: each submission is a normal chat message **plus** structured form metadata (form_id + payload), so conversations can be filtered by form.
5. **Storage‑agnostic**: works with **Postgres JSONB** or **local JSON files**.

---

## High‑level Architecture

- **UI layer (Gradio)**: renders form controls and composes a submission.
- **Forms runtime**: validates payload against JSON Schema, assigns `form_id` & `form_version`, attaches default instructions, and emits a **Message** to the existing message bus/storage.
- **Storage**: extend message storage to include:
  - `form_id` (nullable; `"chat"` for plain free‑text)
  - `form_version`
  - `form_payload` (JSON)
- **Analytics**: separate read path for querying `form_payload` across time; optional embeddings for semantic notes.
- **Prompting**: prepend a short, deterministic instruction snippet when the LLM receives a form submission.

---

## Phase 1 — Hard‑coded Spoons Check‑in (MVP)

**Scope**
- Add a **Spoons** panel with:
  - `Energy Spoons` (slider 0–10, default 5)
  - `Mood Spoons` (slider 0–10, default 5)
  - `Gravity` (slider 0–3, labels: 0=None, 1=Light, 2=Moderate, 3=Heavy)
  - `Must Do's` (textbox, single line)
  - `Nice To's` (textbox, single line)
  - `Other Notes` (multiline)
- Submit button → creates a message with **both** human‑friendly text and machine‑readable `form_payload`.

**Message record shape**
```json
// API envelope (what the app passes around)
{
  "message_id": "uuid-or-snowflake-string",
  "thread_id": "string",
  "role": "user",
  "content": "Energy Spoons: 8
Mood Spoons: 4
Gravity: 1
Must Do's: Work
Nice To's: Play
Other Notes: Woke early with a headache.",
  "metadata": {
    "form_id": "spoons_checkin",
    "form_version": 1,
    "form_payload": {
      "energy": 8,
      "mood": 4,
      "gravity": 1,
      "must_dos": "Work",
      "nice_tos": "Play",
      "notes": "Woke early with a headache.",
      "timestamp": "2025-10-08T16:00:00Z"
    }
  }
}
```

**DB row mapping (to your existing `public.messages` schema)**
```sql
-- messages table columns (existing):
-- id BIGINT PK, message_id TEXT, thread_id TEXT, role TEXT, content TEXT,
-- embedding VECTOR(1024), metadata JSONB, created_at TIMESTAMPTZ, updated_at TIMESTAMPTZ
-- We store form details inside `metadata` (no new columns needed):
--   metadata.form_id        TEXT
--   metadata.form_version   INTEGER
--   metadata.form_payload   JSONB
```
```json
{
  "id": "uuid",
  "thread_id": "uuid",
  "role": "user",
  "content": "Energy Spoons: 8\nMood Spoons: 4\nGravity: 1\nMust Do's: Work\nNice To's: Play\nOther Notes: Woke early with a headache.",
  "form_id": "spoons_checkin",
  "form_version": 1,
  "form_payload": {
    "energy": 8,
    "mood": 4,
    "gravity": 1,
    "must_dos": "Work",
    "nice_tos": "Play",
    "notes": "Woke early with a headache.",
    "timestamp": "2025-10-08T16:00:00Z"
  },
  "created_at": "2025-10-08T16:00:00Z"
}
```

**Default instruction (prepended for LLM on this message)**
```
[Form: spoons_checkin.v1]
You are a pacing coach. Given energy, mood, and gravity, produce a brief pacing plan for today: 3–5 concrete tasks, timeboxes, rest ratios using a “10% less than I think I can” rule, red/yellow/green activities, and a one‑sentence reasoning. Keep it under 180 words and avoid medical claims.
```

**DB changes (Postgres)**
```sql
-- No new columns required; use existing `metadata JSONB` on `public.messages`.
-- Recommended supporting indexes:

-- 1) GIN on metadata for general JSONB containment/paths
CREATE INDEX IF NOT EXISTS messages_metadata_gin_idx
  ON public.messages USING GIN (metadata);

-- 2) Expression index to speed up filtering by form_id
CREATE INDEX IF NOT EXISTS messages_form_id_idx
  ON public.messages ((metadata->>'form_id'));

-- 3) Optional expression index for created_at by form
CREATE INDEX IF NOT EXISTS messages_form_created_idx
  ON public.messages ((metadata->>'form_id'), created_at);
```
```sql
ALTER TABLE messages
  ADD COLUMN form_id TEXT,
  ADD COLUMN form_version INTEGER,
  ADD COLUMN form_payload JSONB;

-- Helpful index for analytics
CREATE INDEX IF NOT EXISTS idx_messages_form ON messages (form_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_formpayload_gin ON messages USING GIN (form_payload);
```

**File‑mode changes**
- Extend message JSON lines with the same keys: `form_id`, `form_version`, `form_payload`.

**Embedding policy**
- Continue embedding the natural‑language `content` for RAG.
- Optionally embed `notes` only (extracted from `form_payload`) to reduce noise.

**Filtering**
- UI toggle: “Show only Spoons submissions in this thread” →
  ```sql
  SELECT * FROM public.messages
   WHERE thread_id = $1
     AND metadata->>'form_id' = 'spoons_checkin'
   ORDER BY created_at;
  ```

**LLM Input construction**
- When sending to the model, append the default instruction **immediately before** the user’s content for this turn.

**MVP acceptance**
- Can submit and store check‑ins.
- Can filter and view just Spoons submissions.
- Model replies with a pacing plan using the instruction.

---

## Phase 2 — Wire Format & Robust Parsing

To make submissions resilient and easily copyable into chat, use a **dual‑format** payload:

1) **Human‑friendly header block** (what Jim already types):
```
Energy Spoons: 8
Mood Spoons: 4
Gravity: 1
Must Do's: Work
Nice To's: Play
Other Notes: Woke early with a headache.
```

2) **Machine‑friendly JSON block** (YAML front‑matter style), appended after a marker so parsers can extract without guesswork:
```
---FORM JSON---
{"form_id":"spoons_checkin","form_version":1,"energy":8,"mood":4,"gravity":1,
 "must_dos":"Work","nice_tos":"Play","notes":"Woke early with a headache.",
 "timestamp":"2025-10-08T16:00:00Z"}
---END FORM JSON---
```

**Parser behavior**
- If the JSON block exists and validates against the schema → trust it.
- Else, fallback to header parsing with tolerant regex → construct payload and validate.

**JSON Schema (spoons_checkin.v1)
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "homeai://schemas/spoons_checkin.v1.json",
  "type": "object",
  "required": ["energy", "mood", "gravity"],
  "properties": {
    "energy": {"type": "integer", "minimum": 0, "maximum": 10},
    "mood":   {"type": "integer", "minimum": 0, "maximum": 10},
    "gravity": {"type": "integer", "minimum": 0, "maximum": 3},
    "must_dos": {"type": "string"},
    "nice_tos": {"type": "string"},
    "notes": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"}
  },
  "additionalProperties": false
}
```
json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "homeai://schemas/spoons_checkin.v1.json",
  "type": "object",
  "required": ["energy", "mood", "gravity"],
  "properties": {
    "form_id": {"const": "spoons_checkin"},
    "form_version": {"const": 1},
    "energy": {"type": "integer", "minimum": 0, "maximum": 10},
    "mood": {"type": "integer", "minimum": 0, "maximum": 10},
    "gravity": {"type": "integer", "minimum": 0, "maximum": 3},
    "must_dos": {"type": "string"},
    "nice_tos": {"type": "string"},
    "notes": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"}
  },
  "additionalProperties": false
}
```

---

## Phase 3 — Data‑Driven Forms

Introduce **Form Definitions** stored as JSON and loaded at startup (or hot‑reloaded).

**Form Definition schema (draft)**
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "homeai://schemas/form_definition.v1.json",
  "type": "object",
  "required": ["id", "version", "title", "fields", "instruction"],
  "properties": {
    "id": {"type": "string"},
    "version": {"type": "integer"},
    "title": {"type": "string"},
    "instruction": {"type": "string"},
    "fields": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["key", "label", "type"],
        "properties": {
          "key": {"type": "string"},
          "label": {"type": "string"},
          "type": {"enum": ["slider","text","textarea","select","multiselect","checkbox","date","time","number"]},
          "min": {"type": "number"},
          "max": {"type": "number"},
          "step": {"type": "number"},
          "placeholder": {"type": "string"},
          "options": {"type": "array", "items": {"type": "string"}},
          "required": {"type": "boolean", "default": false}
        },
        "additionalProperties": false
      }
    }
  },
  "additionalProperties": false
}
```

**Example: spoons_checkin.form.json**
```json
{
  "id": "spoons_checkin",
  "version": 2,
  "title": "Daily Spoons Check‑in",
  "instruction": "[Form: spoons_checkin.v2] You are a pacing coach... (same constraints as v1 with any refinements)",
  "fields": [
    {"key": "energy", "label": "Energy Spoons", "type": "slider", "min": 0, "max": 10, "step": 1, "required": true},
    {"key": "mood",   "label": "Mood Spoons",   "type": "slider", "min": 0, "max": 10, "step": 1, "required": true},
    {"key": "gravity", "label": "Gravity",        "type": "slider", "min": 0, "max": 3,  "step": 1, "required": true},
    {"key": "must_dos","label": "Must Do's",      "type": "text",   "placeholder": "Work, meds, appointments"},
    {"key": "nice_tos","label": "Nice To's",      "type": "text",   "placeholder": "Game, art, walk"},
    {"key": "notes",   "label": "Other Notes",     "type": "textarea", "placeholder": "Anything else"}
  ]
}
```

**Runtime behavior**
- Load all form definitions from `/forms/*.json`.
- Render UI dynamically based on `fields`.
- On submit, build `form_payload` from keys, stamp `form_id` & `form_version`, validate against the schema, store.
- Attach the form’s `instruction` to the LLM call.

## Phase 4 — Analytics & Charts & Charts

**Queries**
- Time‑series of `energy`, `mood`, `gravity` per day.
- Correlate `gravity` vs. `next‑day energy` (lagged join on dates).
- Keyword search over `notes`.

**Example SQL (Postgres)**
```sql
-- Daily series for a thread
SELECT date_trunc('day', created_at) AS day,
       AVG( (metadata->'form_payload'->>'energy')::int ) AS energy_avg,
       AVG( (metadata->'form_payload'->>'mood')::int )   AS mood_avg,
       AVG( (metadata->'form_payload'->>'gravity')::int ) AS gravity_avg
FROM public.messages
WHERE thread_id = $1
  AND metadata->>'form_id' = 'spoons_checkin'
GROUP BY 1
ORDER BY 1;

-- Filter notes by keyword
SELECT * FROM public.messages
WHERE metadata->>'form_id' = 'spoons_checkin'
  AND (metadata->'form_payload'->>'notes') ILIKE '%' || $kw || '%'
ORDER BY created_at DESC;
```

**Endpoints / UI**
- `/forms/spoons/summary?thread_id=...&from=...&to=...`
- Charts: moving averages (7‑day), min/max bands, heatmaps by weekday.
- Export CSV/Parquet.

---

## Phase 5 — API & Filtering

**Filter by form**
- Thread view: checkbox “Only show forms → [select: spoons_checkin, …]”.
- Global search: `form:spoons_checkin energy>6 gravity<=1` (simple query parser) → translates to SQL JSONB ops.

**Programmatic**
- `GET /threads/{id}/forms/{form_id}` → paged list of submissions (id, created_at, payload excerpt).
- `POST /forms/{form_id}/submit` → accepts JSON payload with optional `content_hint` (free‑text summary), returns stored message.

---

## Phase 6 — Instruction Packs

Provide composable instruction snippets per form. For `spoons_checkin`, wire these by default when a message carries `metadata.form_id = 'spoons_checkin'`.

### instruction (single check‑in → pacing plan)
```
[Form: spoons_checkin.v1]

You are a supportive pacing coach. The user submitted a structured “Spoons” check-in with fields:
- energy (0–10), mood (0–10), gravity (0–3), must_dos, nice_tos, notes.

Goal: produce a SAME-DAY pacing plan that prevents post-exertional malaise (PEM) using a “10% less than I think I can” rule. Be concise, practical, and non-medical.

Rules
- Tone: calm, collaborative, non-judgmental. No medical advice or diagnoses.
- Output < 180 words. No emojis. No bullet spam; keep it tight.
- Prefer concrete actions, short timeboxes, and visible stopping points.
- If energy ≤ 3 or gravity ≥ 2, bias toward micro-tasks, longer rests, and fewer commitments.
- Always include “red / yellow / green” activity gates based on today’s spoons.
- Convert must_dos into the smallest viable steps; at most 3–5 tasks total.
- Use the “10% less” rule explicitly when sizing blocks.

Required Output Format (markdown)
### Today’s Pacing Plan
**Spoons**: Energy {{energy}} | Mood {{mood}} | Gravity {{gravity}}

**Focus (3–5 tasks)**  
1) … (timebox, success criteria)  
2) …  
3) …

**Rest Rhythm**  
- Work:Rest ≈ X:Y (e.g., 20:10). Insert 1–2 longer resets.

**Red / Yellow / Green**  
- Red (avoid): …  
- Yellow (limit): …  
- Green (safe/short): …

**Why this works (1 sentence)**  
…

Use must_dos/nice_tos/notes to personalize, but keep it brief.
```

### analysis_instruction (weekly/monthly trends)
```
[Form: spoons_checkin.analysis]

Given a set of Spoons entries (energy, mood, gravity, notes) over a time window, summarize trends in < 180 words:
- 7-day moving tendencies (up/down/flat) for energy, mood, gravity.
- Two strongest correlations you notice (e.g., high gravity → next-day energy dips).
- One experiment to try next week (change only one variable).
Avoid medical claims. Keep it actionable and kind.
```

### chart_instruction (when returning plots)
```
[Form: spoons_checkin.chart]

Briefly annotate the chart: call out peaks, troughs, plateaus, and any weekday patterns. Offer one cautious hypothesis and one next step to confirm or falsify it. ≤ 120 words.
```

## Phase 7 — Migrations & Repos
 Migrations & Repos

Your core tables already exist:
- `public.messages` stores chat + embeddings + `metadata JSONB`.
- `public.doc_chunks` stores file chunk embeddings.
- `public.homeai_memory` stores structured memories with `plain_text` for FTS.

We layer forms atop `messages.metadata` and keep migrations minimal (indexes above).

**Optional: FTS fix for `homeai_memory` (syntax correction)**
```sql
-- If you want a simple FTS index on plain_text using the 'simple' config:
CREATE INDEX IF NOT EXISTS homeai_memory_plain_text_fts_idx
  ON public.homeai_memory
  USING GIN (to_tsvector('simple', coalesce(plain_text, '')));
```

**PgRepo**
- `save_message(message: Message)` → sets `metadata.form_id`, `metadata.form_version`, `metadata.form_payload`.
- `list_forms(thread_id, form_id, from=None, to=None)` → filters via `metadata->>'form_id'`.
- `aggregate_forms(form_id, window='7 days')` → uses expressions on `metadata->'form_payload'`.

---

## UX Details (Gradio)
- Modular panel: `Forms ▸ Spoons Check‑in`.
- Controls: 3 sliders, 3 text fields.
- Submit → shows a preview card: human text + hidden JSON block.
- Buttons: `Send`, `Send + Pin Today`, `Copy as Text`, `Copy JSON`.
- Thread filter toggle: `Only forms in view`.
- Minimal latency: local validate → store → render LLM reply.

---

## Error Handling & Validation
- Client‑side: range checks for sliders; empty optional strings become `""`.
- Server‑side: JSON Schema validation; reject unknown keys when `additionalProperties=false`.
- Versioning: increment `form_version` when definition changes; runtime stores the version used for each submission.

---

## Phasing Strategy (Recommended)

**Stage A (quick win)**
- Implement Phase 1 fully (hard‑coded spoons form) + storage columns + filtering + instruction prepend.

**Stage B**
- Add Phase 2 dual‑format wire and robust parser; backfill `form_payload` for prior header‑only entries when possible.

**Stage C**
- Implement Phase 3 data‑driven forms with JSON definitions; keep spoons as the first dynamic form.

**Stage D**
- Add Phase 4 analytics views and charts.

**Stage E**
- Implement Phase 5 API endpoints and global filters.

**Stage F**
- Add Phase 6 instruction packs and Phase 7 migrations/rollups.

---

## Notes on the Uploaded Release Archives
You mentioned compressed release files (HomeAI‑0.3.1). This plan integrates without requiring direct archive access. Codex can wire the new forms code into the current repo boundaries (UI, server, storage) using the incremental migrations above.

---

## Ready‑to‑Build Checklist for Codex
- [ ] Add columns & indexes to `messages` (or file‑mode keys)
- [ ] Implement `FormRuntime` (validate, version, build `form_payload`, attach instruction)
- [ ] Hard‑code spoons UI + submit flow
- [ ] LLM call adapter: prepend instruction when `form_id` present
- [ ] Filter: per‑thread, `form_id = 'spoons_checkin'`
- [ ] Parser: JSON block → header fallback
- [ ] Data‑driven loader for `/forms/*.json`
- [ ] Analytics endpoints + charts (later stages)
- [ ] Tests: unit for parser/validation; integration for store/query; snapshot for LLM prompt envelope

---

**That’s the spine.** Start with Stage A; you’ll get immediate value (fast check‑ins + consistent replies), then layer in the fancy footwork (definitions, analytics) without breaking compatibility.


---

## Phase 4.5 — Messages View for Forms (Postgres)

Add convenient views to make analytics and dashboards painless while keeping the base `public.messages` table unchanged.

```sql
-- 1) Generic messages+forms view (keeps content + exposes metadata)
CREATE OR REPLACE VIEW public.messages_forms_v AS
SELECT
  m.id,
  m.message_id,
  m.thread_id,
  m.role,
  m.content,
  m.created_at,
  (m.metadata->>'form_id')::text                     AS form_id,
  NULLIF(m.metadata->>'form_version','')::int        AS form_version,
  m.metadata->'form_payload'                         AS form_payload,
  -- Spoons-specific typed projections (NULL for non-spoons rows)
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN NULLIF(m.metadata->'form_payload'->>'energy','')::int END   AS spoons_energy,
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN NULLIF(m.metadata->'form_payload'->>'mood','')::int END     AS spoons_mood,
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN NULLIF(m.metadata->'form_payload'->>'gravity','')::int END  AS spoons_gravity,
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN m.metadata->'form_payload'->>'must_dos' END                 AS spoons_must_dos,
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN m.metadata->'form_payload'->>'nice_tos' END                 AS spoons_nice_tos,
  CASE WHEN m.metadata->>'form_id' = 'spoons_checkin'
       THEN m.metadata->'form_payload'->>'notes' END                    AS spoons_notes
FROM public.messages m;

-- 2) Spoons-only view with helpful time fields
CREATE OR REPLACE VIEW public.messages_forms_spoons_v AS
SELECT
  mf.*, 
  -- prefer a client-supplied payload timestamp; fall back to created_at
  COALESCE((mf.form_payload->>'timestamp')::timestamptz, mf.created_at)        AS ts,
  (date_trunc('day', COALESCE((mf.form_payload->>'timestamp')::timestamptz, mf.created_at)))::date AS day,
  extract(dow from COALESCE((mf.form_payload->>'timestamp')::timestamptz, mf.created_at))::int     AS dow
FROM public.messages_forms_v mf
WHERE mf.form_id = 'spoons_checkin';
```

**Notes**
- These are **plain views** (not materialized), so they auto-reflect new data.
- The _spoons_ columns are `NULL` for non-spoons rows, keeping the generic view broadly useful.
- If you later add new forms, you can append additional `CASE WHEN form_id = 'x' THEN ... END AS x_field` columns, or keep them polymorphic and only type-project in form-specific views.

**Handy queries**
```sql
-- Latest 14 spoons entries for a thread
SELECT day, spoons_energy, spoons_mood, spoons_gravity, spoons_must_dos, spoons_notes
FROM public.messages_forms_spoons_v
WHERE thread_id = $1
ORDER BY ts DESC
LIMIT 14;

-- Daily summary across a date range
SELECT day,
       AVG(spoons_energy)  AS energy_avg,
       AVG(spoons_mood)    AS mood_avg,
       AVG(spoons_gravity) AS gravity_avg
FROM public.messages_forms_spoons_v
WHERE thread_id = $1 AND day BETWEEN $from::date AND $to::date
GROUP BY day
ORDER BY day;

-- Keyword search in notes
SELECT ts, spoons_energy, spoons_mood, spoons_gravity, spoons_notes
FROM public.messages_forms_spoons_v
WHERE thread_id = $1 AND spoons_notes ILIKE '%' || $kw || '%'
ORDER BY ts DESC;
```

