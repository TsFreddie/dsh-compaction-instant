/**
 * Same-session recall core for `dsh-compaction-instant`.
 *
 * The lossless counterpart of the VCC-style compiler: compaction checkpoints
 * mark every elision with a `(seq N)` / `(seqs A-B)` pointer into the durable
 * session log; this module expands those pointers back into the exact
 * original content. Because the log is append-only, every token the compiler
 * ever truncated or elided is still present and recoverable — recall closes
 * the loop that makes the compaction engine near-lossless.
 *
 * Shared by the model-facing `recall` tool (`./tool`). Reads any session-shaped
 * value with `events` and `deriveEventMessage`, so it is unit-testable without
 * a running harness.
 *
 * @module dsh-compaction-instant/recall
 */
import { estimateEntryTokens, isCheckpointSource, projectToolResultText, sanitize, truncateTokens } from "./compiler.js";
import { sessionEventAt, sessionEvents } from "./session-compat.js";

/** Default total budget for one recall operation, in density-aware tokens. */
export const DEFAULT_MAX_RECALL_TOKENS = 16000;
/** Widest single range accepted from one selection, bounding expansion work. */
export const MAX_RECALL_SPAN = 1000;

/**
 * Parse one seq selection string into ordered inclusive ranges.
 *
 * Accepted forms per comma-separated part: `12`, `3-7`, `seq 12`,
 * `seqs 3-7`, with optional surrounding parentheses — so a model or human
 * can paste a checkpoint marker verbatim.
 * @param input - raw selection text.
 * @returns `{ selections, errors }`; errors are stable human-readable strings.
 */
export function parseSeqSpec(input) {
  const selections = [];
  const errors = [];
  const raw = String(input ?? "").trim();
  if (raw.length === 0) return { selections, errors: ["missing seq selection"] };
  for (const part of raw.split(",")) {
    let token = part.trim();
    if (token.length === 0) continue;
    token = token.replace(/^\((.*)\)$/u, "$1").trim();
    const match = /^(?:seqs?\s+)?(\d+)(?:\s*-\s*(\d+))?$/iu.exec(token);
    if (match === null) {
      errors.push(`invalid seq selection "${part}"`);
      continue;
    }
    const start = Number(match[1]);
    const end = match[2] === undefined ? start : Number(match[2]);
    if (end < start) {
      errors.push(`invalid seq range "${part}" (end before start)`);
      continue;
    }
    if (end - start + 1 > MAX_RECALL_SPAN) {
      errors.push(`seq range "${part}" is wider than the ${MAX_RECALL_SPAN}-seq limit`);
      continue;
    }
    selections.push({ start, end });
  }
  return { selections, errors };
}

/**
 * Project one derived message into plain text, keeping everything: text,
 * reasoning, raw tool-call arguments, and nested tool-result content.
 * Recall is the lossless layer, so nothing is elided here except media,
 * which renders as labels (their bytes live in the attachment service).
 * @param message - derived message (`Session.deriveEventMessage` output).
 * @returns full plain-text projection.
 */
export function projectMessageText(message) {
  const parts = [];
  for (const block of message.content) {
    switch (block.type) {
      case "text":
        if (block.text !== undefined && block.text.length > 0) parts.push(sanitize(block.text));
        break;
      case "reasoning":
        if (block.text !== undefined && block.text.length > 0) parts.push(`[reasoning]\n${sanitize(block.text)}`);
        break;
      case "tool-call":
        parts.push(`[tool-call ${block.name ?? "unknown"}]\n${sanitize(block.arguments ?? "")}`);
        break;
      case "tool-result":
        parts.push(`[tool-result]\n${projectToolResultText(block.content ?? [])}`);
        break;
      case "image":
        parts.push("[image]");
        break;
      case "document":
        parts.push("[document]");
        break;
      default:
        parts.push(`[${String(block.type)}]`);
    }
  }
  return parts.join("\n");
}

/**
 * Expand ordered selections into a deduplicated ordered seq list.
 * @param selections - parsed inclusive ranges.
 * @returns unique seqs in first-appearance order.
 */
export function expandSelections(selections) {
  const seqs = [];
  const seen = new Set();
  for (const { start, end } of selections) {
    for (let seq = start; seq <= end; seq += 1) {
      if (seen.has(seq)) continue;
      seen.add(seq);
      seqs.push(seq);
    }
  }
  return seqs;
}

/**
 * Collect the durable seqs of every landed compaction checkpoint node in one
 * session, oldest first. Each compaction replaces its span with exactly one
 * checkpoint node, so the ordinal (`1` = oldest) counts compactions.
 * @param session - session-shaped value with `eventAt()`/`snapshotEvents()` (or the legacy `events` array).
 * @returns checkpoint node seqs in chronological order.
 */
export function findCheckpointSeqs(session) {
  const seqs = [];
  for (const event of sessionEvents(session)) {
    if (event.type === "user/message" && isCheckpointSource(event.data?.source)) seqs.push(event.seq);
  }
  return seqs;
}

/**
 * Resolve one typed recall reference into inclusive seq ranges.
 *
 *   - `type: "seq"` — `id` is a seq selection: numbers/ranges with the
 *     checkpoint marker forms (`12`, `3-7,15`, `seq 12`, `seqs 3-7`).
 *   - `type: "result"` — `id` names a tool result: the `result N` pointer
 *     from a tool-call one-liner (`result 3`, `seq 3`, or bare `3`); the
 *     resolved seq must be a `tool/result` event.
 *   - `type: "checkpoint"` — `id` is a checkpoint ordinal (`1` = oldest,
 *     as in the `[checkpoint N]` elision marker) or a `seq N` pointer to a
 *     checkpoint node.
 *
 * @param session - session-shaped value with `events` and `deriveEventMessage`.
 * @param type - reference type: `seq`, `result`, or `checkpoint`.
 * @param id - type-dependent reference text.
 * @returns `{ selections, errors }`; errors are stable human-readable strings.
 */
export function resolveRecallReference(session, type, id) {
  const raw = String(id ?? "").trim();
  if (type === "seq") return parseSeqSpec(raw);
  if (type === "result") {
    const token = raw.replace(/^\((.*)\)$/u, "$1").trim().replace(/^result\s+/iu, "");
    const match = /^(\d+)$/u.exec(token);
    if (match === null) return { selections: [], errors: [`invalid result reference "${id}" (expected a seq like "3" or "result 3")`] };
    const seq = Number(match[1]);
    const event = sessionEventAt(session, seq);
    if (event === undefined || event.seq !== seq) return { selections: [], errors: [`result seq ${seq} not found in this session`] };
    if (event.type !== "tool/result") return { selections: [], errors: [`seq ${seq} is not a tool result (it is ${event.type})`] };
    return { selections: [{ start: seq, end: seq }], errors: [] };
  }
  if (type === "checkpoint") {
    const token = raw.replace(/^\((.*)\)$/u, "$1").trim();
    const bySeq = /^seqs?\s+(\d+)$/iu.exec(token);
    if (bySeq !== null) {
      const seq = Number(bySeq[1]);
      const event = sessionEventAt(session, seq);
      if (event === undefined || event.seq !== seq) return { selections: [], errors: [`checkpoint seq ${seq} not found in this session`] };
      if (!isCheckpointSource(event.data?.source)) return { selections: [], errors: [`seq ${seq} is not a checkpoint node`] };
      return { selections: [{ start: seq, end: seq }], errors: [] };
    }
    const ordinal = token.replace(/^checkpoint\s+/iu, "");
    const match = /^(\d+)$/u.exec(ordinal);
    if (match === null) return { selections: [], errors: [`invalid checkpoint reference "${id}" (expected an ordinal like "1" or "checkpoint 1", or a "seq N" pointer)`] };
    const index = Number(match[1]);
    if (!Number.isSafeInteger(index) || index < 1) return { selections: [], errors: [`invalid checkpoint ordinal "${id}"`] };
    const checkpointSeqs = findCheckpointSeqs(session);
    const seq = checkpointSeqs[index - 1];
    if (seq === undefined) return { selections: [], errors: [`checkpoint ${index} not found (this session has ${checkpointSeqs.length} checkpoint(s))`] };
    return { selections: [{ start: seq, end: seq }], errors: [] };
  }
  return { selections: [], errors: [`invalid recall type "${String(type)}" (expected "seq", "result", or "checkpoint")`] };
}

/**
 * Recall the full original content of the requested seqs from one session's
 * durable log. Message events project through `deriveEventMessage`; log-only
 * events render as a labeled data dump. The total output is bounded by
 * `config.maxRecallTokens`; a cut appends a provenance marker and remaining
 * requested seqs are accounted in `skipped`.
 * @param session - session-shaped value with `events` and `deriveEventMessage`.
 * @param selections - parsed inclusive ranges.
 * @param config - `{ maxRecallTokens }`.
 * @returns `{ text, entries, seqs, recalled, missing, skipped, truncated, tokens }`.
 */
export function recallSession(session, selections, config) {
  const maxRecallTokens = config.maxRecallTokens;
  const requested = expandSelections(selections);
  const entries = [];
  const seqs = [];
  let budget = maxRecallTokens;
  let truncated = false;
  let missing = 0;
  let skipped = 0;
  for (let index = 0; index < requested.length; index += 1) {
    const seq = requested[index];
    if (budget <= 0) {
      skipped = requested.length - index;
      truncated = true;
      break;
    }
    const event = sessionEventAt(session, seq);
    if (event === undefined || event.seq !== seq) {
      missing += 1;
      entries.push({ seq, text: `[seq ${seq}: not found in this session]` });
      continue;
    }
    const message = typeof session.deriveEventMessage === "function" ? session.deriveEventMessage(event) : null;
    const body = message !== null
      ? `[seq ${seq}: ${message.role}]\n${projectMessageText(message)}`
      : `[seq ${seq}: ${event.type}]\n${JSON.stringify(event.data).slice(0, maxRecallTokens * 4)}`;
    const kept = truncateTokens(body, budget, "recall budget");
    budget -= estimateEntryTokens(kept.text);
    if (kept.truncated) truncated = true;
    entries.push({ seq, text: kept.text, truncated: kept.truncated });
    seqs.push(seq);
  }
  if (skipped > 0) {
    entries.push({ seq: requested[requested.length - skipped], text: `[recall budget exhausted: ${skipped} further requested seq(s) not included]` });
  }
  return {
    text: entries.map((entry) => entry.text).join("\n\n"),
    entries,
    seqs,
    recalled: seqs.length,
    missing,
    skipped,
    truncated,
    tokens: entries.reduce((total, entry) => total + estimateEntryTokens(entry.text), 0)
  };
}
