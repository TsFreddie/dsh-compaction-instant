/**
 * Same-session recall tests: seq parsing, log expansion, keyword search, the
 * model-facing tools (`recall` typed restore + `search` grep),
 * and the `/recall` command, exercised against a real detached Session.
 * @module dsh-compaction-instant/test/recall
 */
import assert from "node:assert/strict";
import test from "node:test";
import { Context } from "@deepseek-ai/cordis";
import { createAssistantMessage, createToolResultMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { Session } from "@deepseek-ai/dsh-session";
import { sessionEventAt, sessionEvents } from "../src/session-compat.js";
import { apply as applyCommand, defineRecallCommand, resolveConfig as resolveCommandConfig } from "../src/command.js";
import { DEFAULT_MAX_RECALL_TOKENS, findCheckpointSeqs, parseSeqSpec, projectMessageText, recallSession, resolveRecallReference } from "../src/recall.js";
import { compileSearchPattern, InvalidSearchPatternError, searchSession } from "../src/search.js";
import { apply as applyTool, defineRecallTool, defineSearchTool, resolveConfig as resolveToolConfig } from "../src/tool.js";
import { InstantCompactionEngine } from "../src/index.js";
import { compactSurfaceRegion } from "../src/region.js";

/** Build one detached session with a complete (idle) turn bracket. */
function makeIdleSession() {
  const user = createUserMessage({
    content: [{ type: "text", text: "please fix the bug" }],
    source: { kind: "user" }
  });
  const assistant = createAssistantMessage({
    content: [
      { type: "text", text: "on it" },
      { type: "reasoning", text: "the counter overflows here" },
      { type: "tool-call", id: "call-1", name: "read", arguments: '{"file_path":"a.js","offset":1}' }
    ],
    source: { provider: "p", model: "m" }
  });
  const result = createToolResultMessage({
    callId: "call-1",
    content: [{ type: "text", text: "const counter = 0;" }],
    isError: false
  });
  const assistant2 = createAssistantMessage({
    content: [{ type: "text", text: "done" }],
    source: { provider: "p", model: "m" }
  });
  const seed = [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: user, surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { turn: 1, step: 0, stream: [], message: assistant }, surfaceOp: "append" },
    { type: "tool/result", seq: 3, time: 4, data: { message: result }, surfaceOp: "append" },
    { type: "assistant/message", seq: 4, time: 5, data: { turn: 1, step: 0, stream: [], message: assistant2 }, surfaceOp: "append" },
    { type: "turn/end", seq: 5, time: 6, data: { turn: 1 } }
  ];
  return Session.create("session-recall", seed);
}

const idleAgent = (session) => ({
  session,
  runMaintenance: (task) => task(new AbortController().signal)
});

const SEARCH_CONFIG = { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS, maxSearchHits: 50 };

// ── parseSeqSpec ────────────────────────────────────────────────────────────

test("parseSeqSpec accepts numbers, ranges, and checkpoint marker forms", () => {
  assert.deepEqual(parseSeqSpec("12").selections, [{ start: 12, end: 12 }]);
  assert.deepEqual(parseSeqSpec("3-7,15").selections, [{ start: 3, end: 7 }, { start: 15, end: 15 }]);
  assert.deepEqual(parseSeqSpec("seq 12").selections, [{ start: 12, end: 12 }]);
  assert.deepEqual(parseSeqSpec("seqs 3-7").selections, [{ start: 3, end: 7 }]);
  assert.deepEqual(parseSeqSpec("(seq 12), (seqs 3-7)").selections, [{ start: 12, end: 12 }, { start: 3, end: 7 }]);
  assert.deepEqual(parseSeqSpec("  4 - 6 , 8 ").selections, [{ start: 4, end: 6 }, { start: 8, end: 8 }]);
});

test("parseSeqSpec rejects malformed, reversed, and over-wide selections", () => {
  assert.ok(parseSeqSpec("").errors.length > 0);
  assert.ok(parseSeqSpec("abc").errors.length > 0);
  assert.ok(parseSeqSpec("7-3").errors.length > 0);
  assert.ok(parseSeqSpec("1,xyz,2").errors.length === 1);
  assert.ok(parseSeqSpec("1-2000").errors.length > 0);
});

// ── recallSession ───────────────────────────────────────────────────────────

test("recallSession restores exact original content with role headers", () => {
  const session = makeIdleSession();
  const result = recallSession(session, [{ start: 1, end: 4 }], { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS });
  assert.equal(result.recalled, 4);
  assert.equal(result.missing, 0);
  assert.equal(result.skipped, 0);
  assert.equal(result.truncated, false);
  assert.match(result.text, /\[seq 1: user\]\nplease fix the bug/);
  assert.match(result.text, /\[seq 2: assistant\]\non it/);
  assert.match(result.text, /\[reasoning\]\nthe counter overflows here/);
  assert.match(result.text, /\[tool-call read\]\n\{"file_path":"a\.js","offset":1\}/);
  assert.match(result.text, /\[seq 3: user\]\n\[tool-result\]\nconst counter = 0;/);
  assert.match(result.text, /\[seq 4: assistant\]\ndone/);
});

test("recallSession reports missing seqs and deduplicates overlaps", () => {
  const session = makeIdleSession();
  const result = recallSession(session, [{ start: 1, end: 2 }, { start: 2, end: 99 }], { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS });
  // Seq 5 (turn/end) and seq 6 (session/end-seed) are log-only events and
  // are recalled as labeled data dumps.
  assert.equal(result.recalled, 6);
  assert.equal(result.missing, 93);
  const seqs = result.entries.map((entry) => entry.seq);
  assert.deepEqual(seqs.slice(0, 6), [1, 2, 3, 4, 5, 6]);
  assert.equal(seqs[6], 7);
  assert.match(result.entries[6].text, /not found in this session/);
});

test("recallSession truncates at the budget and accounts skipped seqs", () => {
  const session = makeIdleSession();
  const result = recallSession(session, [{ start: 1, end: 4 }], { maxRecallTokens: 24 });
  assert.equal(result.recalled + result.missing + result.skipped, 4);
  assert.ok(result.skipped > 0);
  assert.equal(result.truncated, true);
  assert.match(result.text, /recall budget exhausted/);
  // The first entry always survives intact.
  assert.match(result.text, /please fix the bug/);
});

test("recallSession renders log-only events as a labeled data dump", () => {
  const session = makeIdleSession();
  const result = recallSession(session, [{ start: 0, end: 0 }], { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS });
  assert.equal(result.recalled, 1);
  assert.match(result.text, /\[seq 0: turn\/start\]\n\{"turn":1\}/);
});

test("projectMessageText keeps reasoning and raw tool arguments", () => {
  const message = {
    role: "assistant",
    content: [
      { type: "reasoning", text: "think" },
      { type: "tool-call", id: "c", name: "bash", arguments: '{"command":"ls -la"}' },
      { type: "image", attachment: { id: "i" } }
    ]
  };
  const text = projectMessageText(message);
  assert.match(text, /\[reasoning\]\nthink/);
  assert.match(text, /\[tool-call bash\]\n\{"command":"ls -la"\}/);
  assert.match(text, /\[image\]/);
});

// ── search core ─────────────────────────────────────────────────────────────

test("compileSearchPattern accepts keywords and regexes, case-insensitively", () => {
  assert.equal(compileSearchPattern("VCC").test("vcc compiler"), true);
  assert.equal(compileSearchPattern("a\\.js").test("read a.js"), true);
  assert.equal(compileSearchPattern("（seq）").test("前缀（seq）后缀"), true);
  assert.throws(() => compileSearchPattern(""), InvalidSearchPatternError);
  assert.throws(() => compileSearchPattern("   "), InvalidSearchPatternError);
  assert.throws(() => compileSearchPattern("(unclosed"), InvalidSearchPatternError);
});

test("searchSession finds matching events with seq pointers and matched lines", () => {
  const session = makeIdleSession();
  const result = searchSession(session, "counter", SEARCH_CONFIG);
  assert.equal(result.totalMatches, 2); // reasoning + tool result
  assert.ok(result.hits.length === 2);
  assert.deepEqual(result.hits.map((hit) => hit.seq), [2, 3]);
  assert.match(result.text, /\[search "counter": 2 matching event\(s\)\]/);
  assert.match(result.text, /\[seq 2: assistant\]/);
  assert.match(result.text, /the counter overflows here/);
  assert.match(result.text, /\[seq 3: user\]/);
  assert.match(result.text, /const counter = 0;/);
  assert.match(result.text, /recall with a \(seq N\) pointer/);
});

test("searchSession matches raw tool arguments and log-only event data", () => {
  const session = makeIdleSession();
  const inArgs = searchSession(session, "a\\.js", SEARCH_CONFIG);
  assert.equal(inArgs.totalMatches, 1);
  assert.equal(inArgs.hits[0].seq, 2);
  assert.match(inArgs.hits[0].text, /\{"file_path":"a\.js","offset":1\}/);
  const logOnly = searchSession(session, "turn/start", SEARCH_CONFIG);
  assert.equal(logOnly.totalMatches, 1);
  assert.match(logOnly.hits[0].text, /\[seq 0: turn\/start\]/);
});

test("searchSession reports no hits, caps hits, and bounds output", () => {
  const session = makeIdleSession();
  const none = searchSession(session, "nothing-here", SEARCH_CONFIG);
  assert.equal(none.totalMatches, 0);
  assert.equal(none.hits.length, 0);
  assert.match(none.text, /0 matching event\(s\)/);
  const capped = searchSession(session, "e", { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS, maxSearchHits: 1 });
  assert.ok(capped.totalMatches > 1);
  assert.equal(capped.hits.length, 1);
  assert.ok(capped.omitted > 0);
  assert.match(capped.text, /more matching event\(s\) omitted/);
  const budgeted = searchSession(session, "e", { maxRecallTokens: 10, maxSearchHits: 50 });
  assert.equal(budgeted.truncated, true);
  assert.match(budgeted.text, /search budget exhausted/);
});

// ── recall tool (typed references) ──────────────────────────────────────────

test("recall tool executes with type seq against the calling agent's session", async () => {
  const session = makeIdleSession();
  const tool = defineRecallTool(resolveToolConfig({}));
  assert.equal(tool.name, "recall");
  const value = await tool.execute({ type: "seq", id: "1-2" }, { agent: { session } });
  assert.equal(value.recalled, 2);
  assert.equal(value.missing, 0);
  assert.match(value.text, /please fix the bug/);
  const blocks = tool.output.render({ type: "seq", id: "1-2" }, value);
  assert.deepEqual(blocks, [{ type: "text", text: value.text }]);
});

test("recall tool rejects missing agents and invalid references", async () => {
  const tool = defineRecallTool(resolveToolConfig({}));
  await assert.rejects(() => tool.execute({ type: "seq", id: "1" }, {}), (error) => error.code === "RECALL_AGENT_REQUIRED");
  await assert.rejects(() => tool.execute({ type: "seq", id: "nope" }, { agent: { session: makeIdleSession() } }), (error) => error.code === "RECALL_INVALID_SELECTION");
});

// ── recall tool typed references (result / checkpoint) ────────────────────────

/** Build a session with a tool step and two landed checkpoint nodes. */
function makeCheckpointSession() {
  const session = makeIdleSession();
  const checkpoint = (text) => createUserMessage({
    content: [{ type: "text", text }],
    source: { kind: "plugin", plugin: "compact", compactionId: "test" }
  });
  session.append("user/message", checkpoint("first checkpoint body " + "word ".repeat(50)), { surfaceOp: "append" });
  session.append("user/message", checkpoint("second checkpoint body"), { surfaceOp: "append" });
  return session;
}

test("findCheckpointSeqs returns checkpoint node seqs in chronological order", () => {
  const session = makeCheckpointSession();
  assert.deepEqual(findCheckpointSeqs(session), [7, 8]);
});

test("resolveRecallReference resolves seq, result, and checkpoint references", () => {
  const session = makeCheckpointSession();
  // seq: marker forms pass through
  const seq = resolveRecallReference(session, "seq", "2, seq 4");
  assert.deepEqual(seq.selections, [{ start: 2, end: 2 }, { start: 4, end: 4 }]);
  assert.deepEqual(seq.errors, []);
  // result: the `result N` pointer resolves to the tool/result event
  assert.deepEqual(resolveRecallReference(session, "result", "result 3").selections, [{ start: 3, end: 3 }]);
  assert.deepEqual(resolveRecallReference(session, "result", "3").selections, [{ start: 3, end: 3 }]);
  assert.ok(resolveRecallReference(session, "result", "1").errors[0].includes("not a tool result"));
  assert.ok(resolveRecallReference(session, "result", "bogus").errors[0].includes("invalid result reference"));
  // checkpoint: ordinal, "checkpoint N" form, and "seq N" pointer
  assert.deepEqual(resolveRecallReference(session, "checkpoint", "1").selections, [{ start: 7, end: 7 }]);
  assert.deepEqual(resolveRecallReference(session, "checkpoint", "checkpoint 2").selections, [{ start: 8, end: 8 }]);
  assert.deepEqual(resolveRecallReference(session, "checkpoint", "seq 7").selections, [{ start: 7, end: 7 }]);
  assert.ok(resolveRecallReference(session, "checkpoint", "3").errors[0].includes("has 2 checkpoint(s)"));
  assert.ok(resolveRecallReference(session, "checkpoint", "bogus").errors[0].includes("invalid checkpoint reference"));
  // unknown type
  assert.ok(resolveRecallReference(session, "bogus", "1").errors[0].includes("invalid recall type"));
});

test("recall tool restores by type seq, result, and checkpoint", async () => {
  const session = makeCheckpointSession();
  const tool = defineRecallTool(resolveToolConfig({}));
  assert.equal(tool.name, "recall");
  const seq = await tool.execute({ type: "seq", id: "1-2" }, { agent: { session } });
  assert.equal(seq.recalled, 2);
  assert.match(seq.text, /please fix the bug/);
  const result = await tool.execute({ type: "result", id: "result 3" }, { agent: { session } });
  assert.equal(result.recalled, 1);
  assert.match(result.text, /\[seq 3: user\]/);
  assert.match(result.text, /const counter = 0;/);
  const checkpoint = await tool.execute({ type: "checkpoint", id: "1" }, { agent: { session } });
  assert.equal(checkpoint.recalled, 1);
  assert.match(checkpoint.text, /first checkpoint body/);
  assert.doesNotMatch(checkpoint.text, /second checkpoint body/);
});

test("recall tool rejects missing agents and invalid references", async () => {
  const tool = defineRecallTool(resolveToolConfig({}));
  const session = makeCheckpointSession();
  await assert.rejects(() => tool.execute({ type: "seq", id: "1" }, {}), (error) => error.code === "RECALL_AGENT_REQUIRED");
  await assert.rejects(() => tool.execute({ type: "seq", id: "nope" }, { agent: { session } }), (error) => error.code === "RECALL_INVALID_SELECTION");
  await assert.rejects(() => tool.execute({ type: "result", id: "1" }, { agent: { session } }), (error) => error.code === "RECALL_INVALID_SELECTION");
  await assert.rejects(() => tool.execute({ type: "checkpoint", id: "9" }, { agent: { session } }), (error) => error.code === "RECALL_INVALID_SELECTION");
  await assert.rejects(() => tool.execute({ type: "bogus", id: "1" }, { agent: { session } })); // enum rejects before execute
});

// ── recall (grep) tool ──────────────────────────────────────────────────────

test("search tool searches the session and points at recall", async () => {
  const session = makeIdleSession();
  const tool = defineSearchTool(resolveToolConfig({}));
  assert.equal(tool.name, "search");
  const value = await tool.execute({ pattern: "counter" }, { agent: { session } });
  assert.equal(value.totalMatches, 2);
  assert.equal(value.pattern, "counter");
  assert.match(value.text, /\[seq 2: assistant\]/);
  const blocks = tool.output.render({ pattern: "counter" }, value);
  assert.deepEqual(blocks, [{ type: "text", text: value.text }]);
});

test("search tool rejects missing agents and invalid patterns", async () => {
  const tool = defineSearchTool(resolveToolConfig({}));
  await assert.rejects(() => tool.execute({ pattern: "x" }, {}), (error) => error.code === "RECALL_AGENT_REQUIRED");
  await assert.rejects(() => tool.execute({ pattern: "(unclosed" }, { agent: { session: makeIdleSession() } }), (error) => error.code === "SEARCH_INVALID_PATTERN");
});

test("recall tools config validates the budget and hit cap", () => {
  assert.equal(resolveToolConfig({}).maxRecallTokens, DEFAULT_MAX_RECALL_TOKENS);
  assert.equal(resolveToolConfig({}).maxSearchHits, 50);
  assert.throws(() => resolveToolConfig({ maxRecallTokens: 0 }), /maxRecallTokens/);
  assert.throws(() => resolveToolConfig({ maxSearchHits: 0 }), /maxSearchHits/);
});

// ── /recall command (grep-based) ────────────────────────────────────────────

test("recall command appends a form:recall user message with search hits", async () => {
  const session = makeIdleSession();
  const command = defineRecallCommand(resolveCommandConfig({}));
  assert.equal(command.name, "recall");
  const result = await command.handler({
    agent: idleAgent(session),
    rawInput: "counter",
    signal: new AbortController().signal,
    commandId: "cmd-1"
  });
  assert.equal(result.kind, "success");
  assert.match(result.text, /Found 2 matching event\(s\)/);
  const appended = sessionEventAt(session, result.sourceEventSeq);
  assert.equal(appended.type, "user/message");
  assert.equal(appended.data.source.kind, "plugin");
  assert.equal(appended.data.source.plugin, "recall");
  assert.equal(appended.data.source.form, "recall");
  assert.deepEqual(appended.surfaceOp, "append");
  assert.deepEqual(appended.sourceEventSeqs, [2, 3]);
  assert.match(appended.data.content[0].text, /\[seq 2: assistant\]/);
  // The search result joined the model-visible surface.
  assert.deepEqual(session.surface.nodes, [1, 2, 3, 4, appended.seq]);
});

test("recall command rejects invalid input and no-hit searches without appending", async () => {
  const session = makeIdleSession();
  const command = defineRecallCommand(resolveCommandConfig({}));
  const before = sessionEvents(session).length;
  const empty = await command.handler({
    agent: idleAgent(session),
    rawInput: "",
    signal: new AbortController().signal,
    commandId: "cmd-2"
  });
  assert.equal(empty.kind, "error");
  assert.match(empty.text, /Usage: \/recall/);
  const bad = await command.handler({
    agent: idleAgent(session),
    rawInput: "(unclosed",
    signal: new AbortController().signal,
    commandId: "cmd-3"
  });
  assert.equal(bad.kind, "error");
  assert.match(bad.text, /invalid search pattern/);
  const none = await command.handler({
    agent: idleAgent(session),
    rawInput: "nothing-here",
    signal: new AbortController().signal,
    commandId: "cmd-4"
  });
  assert.equal(none.kind, "error");
  assert.match(none.text, /No matching events/);
  assert.equal(sessionEvents(session).length, before);
});

test("recall command maps maintenance conflicts to an error result", async () => {
  const session = makeIdleSession();
  const command = defineRecallCommand(resolveCommandConfig({}));
  const result = await command.handler({
    agent: {
      session,
      runMaintenance: () => {
        throw new Error("another maintenance task owns the agent");
      }
    },
    rawInput: "counter",
    signal: new AbortController().signal,
    commandId: "cmd-5"
  });
  assert.equal(result.kind, "error");
  assert.match(result.text, /another maintenance task owns the agent/);
});

test("recall command config validates the budget and hit cap", () => {
  assert.equal(resolveCommandConfig({}).maxRecallTokens, DEFAULT_MAX_RECALL_TOKENS);
  assert.equal(resolveCommandConfig({}).maxSearchHits, 50);
  assert.throws(() => resolveCommandConfig({ maxRecallTokens: -1 }), /maxRecallTokens/);
  assert.throws(() => resolveCommandConfig({ maxSearchHits: -1 }), /maxSearchHits/);
});

// ── plugin registration through a real cordis fiber ─────────────────────────

test("recall tool and command apply plugins register and dispose through cordis", async () => {
  const root = new Context();
  const registrations = { tools: [], commands: [] };
  const disposals = { tools: 0, commands: 0 };
  root.provide("tools", {
    register: (definition) => {
      registrations.tools.push(definition);
      return () => {
        disposals.tools += 1;
      };
    }
  });
  root.provide("commands", {
    register: (definition) => {
      registrations.commands.push(definition);
      return () => {
        disposals.commands += 1;
      };
    }
  });
  const fiber = root.plugin((ctx) => {
    applyTool(ctx, {});
    applyCommand(ctx, {});
  });
  await fiber;
  assert.deepEqual(registrations.tools.map((tool) => tool.name).sort(), ["recall", "search"]);
  assert.equal(registrations.commands.length, 1);
  assert.equal(registrations.commands[0].name, "recall");
  await fiber.dispose();
  assert.equal(disposals.tools, 2);
  assert.equal(disposals.commands, 1);
});

// ── the near-lossless loop: compact, then recall/search the shadowed originals ─

test("recall and search restore the exact originals after compaction", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: (target) => ({
      logRevision: 0,
      baseline: { kind: "none", tokens: 0 },
      surfaceDeltaTokens: 0,
      totalTokens: 0,
      surfaceTokens: 0,
      nodes: target.surface.nodes.map((seq) => ({ seq, tokens: 100 }))
    }),
    estimateMessage: (message) => message.content.reduce((total, block) => total + Math.ceil((block.text?.length ?? 0) / 4), 0)
  });
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    textTokens: 64,
    userTextTokens: 64,
    toolCallTokens: 32,
    toolResultExcerptTokens: 32,
    maxTokens: 512
  });
  const compaction = await compactSurfaceRegion(engine.regionDependencies(), session, 1, 3, undefined, {
    owner: null,
    stability: "selected-span"
  }, undefined);
  assert.deepEqual(compaction.shadowedSeqs, [1, 2, 3]);
  // The originals left the surface but remain in the append-only log.
  assert.deepEqual(session.surface.nodes, [compaction.summarySeq + 1, 4]);
  const recalled = recallSession(session, [{ start: 1, end: 3 }], { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS });
  assert.equal(recalled.recalled, 3);
  assert.equal(recalled.truncated, false);
  assert.match(recalled.text, /\[seq 1: user\]\nplease fix the bug/);
  assert.match(recalled.text, /\[tool-call read\]\n\{"file_path":"a\.js","offset":1\}/);
  assert.match(recalled.text, /\[seq 3: user\]\n\[tool-result\]\nconst counter = 0;/);
  // The grep view finds the shadowed content without knowing its seq.
  // Reasoning was elided from the checkpoint, so it is only in the log.
  const found = searchSession(session, "overflows", SEARCH_CONFIG);
  assert.equal(found.totalMatches, 1);
  assert.deepEqual(found.hits.map((hit) => hit.seq), [2]);
  assert.match(found.text, /the counter overflows here/);
  // The tool result survives in the checkpoint excerpt AND the durable log.
  const broad = searchSession(session, "counter", SEARCH_CONFIG);
  assert.ok(broad.totalMatches >= 2, `found ${broad.totalMatches} matches`);
  assert.ok(broad.hits.some((hit) => hit.seq === 3), "original tool result hit");
});
