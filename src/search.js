/**
 * Keyword/regex search over the durable session log for `dsh-compaction-instant`.
 *
 * The retrieval counterpart of the VCC-style compiler's `--grep` view: the
 * append-only session log is the full transcript, so this module scans every
 * event's projected content and returns the matching events with their
 * `(seq N)` pointers. An agent (or the `/recall` command) can then call
 * `recall` with any pointer to restore the exact original tokens.
 *
 * Nothing is modified, nothing is elided — only the rendered hit lines are
 * bounded, by the shared `maxRecallTokens` budget and a per-hit line cap.
 *
 * @module dsh-compaction-instant/search
 */
import { estimateEntryTokens, sanitize, truncateTokens } from "./compiler.js";
import { projectMessageText } from "./recall.js";
import { sessionEventAt, sessionEvents } from "./session-compat.js";

/** Default cap on the number of matching events shown in one result. */
export const DEFAULT_MAX_SEARCH_HITS = 50;
/** Default cap on shown matching lines per hit event. */
export const MAX_LINES_PER_HIT = 10;

/** A search pattern that cannot be compiled into a regular expression. */
export class InvalidSearchPatternError extends Error {
  /**
   * @param source - the raw pattern the user supplied.
   * @param reason - the regex compiler's message.
   */
  constructor(source, reason) {
    super(`invalid search pattern ${JSON.stringify(String(source))}: ${reason}`);
    this.source = String(source);
  }
}

/**
 * Compile one search pattern: case-insensitive, Unicode-aware, treated as a
 * regular expression (VCC `--grep` semantics) so keywords work as-is and
 * regexes work when escaped.
 * @param source - raw pattern text.
 * @returns the compiled pattern.
 * @throws InvalidSearchPatternError when empty or not compilable.
 */
export function compileSearchPattern(source) {
  const trimmed = String(source ?? "").trim();
  if (trimmed.length === 0) throw new InvalidSearchPatternError(source, "pattern is empty");
  try {
    return new RegExp(trimmed, "iu");
  } catch (error) {
    throw new InvalidSearchPatternError(source, error instanceof Error ? error.message : String(error));
  }
}

/** One search hit: the durable seq, its kind, and the rendered matched lines. */
export function searchSession(session, patternSource, config) {
  const pattern = compileSearchPattern(patternSource);
  const maxHits = config.maxSearchHits;
  const maxTokens = config.maxRecallTokens;
  const events = sessionEvents(session);
  const hits = [];
  let totalMatches = 0;
  let budget = maxTokens;
  let truncated = false;
  for (let seq = 0; seq < events.length; seq += 1) {
    const event = events[seq];
    if (event === undefined || event.seq !== seq) continue;
    const message = typeof session.deriveEventMessage === "function" ? session.deriveEventMessage(event) : null;
    const body = message !== null
      ? projectMessageText(message)
      : `[${event.type}]\n${JSON.stringify(event.data ?? null)}`;
    const lines = body.split("\n");
    const matched = [];
    for (let index = 0; index < lines.length; index += 1) {
      if (pattern.test(lines[index])) matched.push({ line: index + 1, text: sanitize(lines[index]) });
    }
    if (matched.length === 0) continue;
    totalMatches += 1;
    if (hits.length >= maxHits || budget <= 0) {
      truncated = true;
      continue;
    }
    const header = `[seq ${seq}: ${message !== null ? message.role : event.type}]`;
    const shown = matched.slice(0, MAX_LINES_PER_HIT);
    const more = matched.length - shown.length;
    const block = [
      header,
      ...shown.map((match) => `  ${match.line}: ${match.text}`),
      ...(more > 0 ? [`  ...(${more} more matching lines in this event)`] : [])
    ].join("\n");
    const kept = truncateTokens(block, budget, "search budget");
    budget -= estimateEntryTokens(kept.text);
    if (kept.truncated) truncated = true;
    hits.push({ seq, kind: message !== null ? message.role : event.type, text: kept.text });
  }
  const omitted = totalMatches - hits.length;
  const lines = [
    `[search "${pattern.source}": ${totalMatches} matching event(s)]`,
    ...hits.map((hit) => hit.text),
    ...(omitted > 0 ? [`[${omitted} more matching event(s) omitted — narrow the pattern or use recall]`] : []),
    ...(truncated ? ["[search budget exhausted; some hits were cut]"] : [])
  ];
  if (hits.length > 0) lines.push("[use recall with a (seq N) pointer to restore any hit's full original content]");
  const text = lines.join("\n\n");
  return {
    pattern: pattern.source,
    totalMatches,
    hits,
    omitted,
    truncated,
    tokens: hits.reduce((total, hit) => total + estimateEntryTokens(hit.text), 0),
    text
  };
}
