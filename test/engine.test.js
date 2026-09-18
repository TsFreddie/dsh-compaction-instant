/**
 * Engine-level test: the real `InstantCompactionEngine.compile` hook runs the
 * deterministic compiler over a real detached session without any model call.
 * @module dsh-compaction-instant/test/engine
 */
import assert from "node:assert/strict";
import test from "node:test";
import { Context } from "@deepseek-ai/cordis";
import { createAssistantMessage, createToolResultMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { Session } from "@deepseek-ai/dsh-session";
import { sessionEventAt, sessionEvents } from "../src/session-compat.js";
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
      { type: "tool-call", id: "call-1", name: "read", arguments: '{"file_path":"a.js"}' }
    ],
    source: { provider: "p", model: "m" }
  });
  const result = createToolResultMessage({
    callId: "call-1",
    content: [{ type: "text", text: "file content" }],
    isError: false
  });
  const assistant2 = createAssistantMessage({
    content: [{ type: "text", text: "done" }],
    source: { provider: "p", model: "m" }
  });
  const user2 = createUserMessage({
    content: [{ type: "text", text: "thank you" }],
    source: { kind: "user" }
  });
  const seed = [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: user, surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { turn: 1, step: 0, stream: [], message: assistant }, surfaceOp: "append" },
    { type: "tool/result", seq: 3, time: 4, data: { message: result }, surfaceOp: "append" },
    { type: "assistant/message", seq: 4, time: 5, data: { turn: 1, step: 0, stream: [], message: assistant2 }, surfaceOp: "append" },
    { type: "user/message", seq: 5, time: 6, data: user2, surfaceOp: "append" },
    { type: "turn/end", seq: 6, time: 7, data: { turn: 1 } }
  ];
  return Session.create("session-1", seed);
}

test("compile() compiles a real session deterministically with seq provenance", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: session.surface.nodes.map((seq) => ({ seq, tokens: 100 })) }),
    estimateMessage: () => 10
  });
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    textTokens: 64,
    userTextTokens: 64,
    toolCallTokens: 32,
    toolResultExcerptTokens: 32,
    maxTokens: 512
  });
  const prepared = {
    shadowedSeqs: [1, 2, 3, 4],
    session,
    start: 1,
    end: 4,
    startIdx: 0,
    endIdx: 3,
    measurement: { nodes: [] },
    selectedNodes: [],
    shadowedTokenCount: 400
  };
  const result = await engine.compile(prepared, undefined, undefined);
  assert.equal(result.provider, "dsh-compaction-instant");
  assert.equal(result.model, "vcc-compiler");
  const text = result.entries.map((entry) => entry.text).join("\n");
  assert.match(text, /please fix the bug/);
  assert.match(text, /on it/);
  assert.match(text, /\* read "a\.js" \(seq 2 -> result 3\)/);
  assert.doesNotMatch(text, /-> read: ok/);
  assert.doesNotMatch(text, /file content/);
  assert.match(text, /done/);
  // Deterministic: compiling twice yields identical output.
  const again = await engine.compile(prepared, undefined, undefined);
  assert.deepEqual(result.entries, again.entries);
  assert.deepEqual(result.stats, again.stats);
});

test("compile() aborts on a cancelled signal", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: [] }),
    estimateMessage: () => 10
  });
  const engine = new InstantCompactionEngine(ctx, { auto: false });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    () => engine.compile({ shadowedSeqs: [], session }, undefined, controller.signal),
    (error) => error.name === "AbortError"
  );
});

test("compile() caps the checkpoint at checkpointCap", async () => {
  const session = makeIdleSession();
  const ctx = new Context();
  ctx.provide("tokenMeter", {
    measure: () => ({ nodes: [] }),
    estimateMessage: () => 10
  });
  // The total budget for one checkpoint is exactly `checkpointCap` — no
  // proportional scaling, no floor — so a small span compiles near-losslessly
  // under the cap and a huge span elides down to it.
  const engine = new InstantCompactionEngine(ctx, {
    auto: false,
    checkpointCap: 4096,
    textTokens: 512,
    userTextTokens: 1024,
    toolCallTokens: 128,
    toolResultExcerptTokens: 256
  });
  const long = `long text ${"word ".repeat(300)}`;
  const seeded = Session.create("session-2", [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: createUserMessage({ content: [{ type: "text", text: long }], source: { kind: "user" } }), surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { turn: 1, step: 0, stream: [], message: createAssistantMessage({ content: [{ type: "text", text: "ok" }], source: { provider: "p", model: "m" } }) }, surfaceOp: "append" },
    { type: "turn/end", seq: 3, time: 4, data: { turn: 1 } }
  ]);
  const prepared = {
    shadowedSeqs: [1, 2],
    session: seeded,
    start: 1,
    end: 2,
    startIdx: 0,
    endIdx: 1,
    measurement: { nodes: [] },
    selectedNodes: [],
    shadowedTokenCount: 2000
  };
  // The budget is the cap regardless of the shadowed span size.
  assert.equal(engine.effectiveMaxTokens(2000), 4096);
  assert.equal(engine.effectiveMaxTokens(1_000_000), 4096);
  const result = await engine.compile(prepared, undefined, undefined);
  // 2000 priced tokens fit under the 4096 cap, so the long user text survives
  // untruncated.
  const text = result.entries.map((entry) => entry.text).join("\n");
  assert.match(text, /word word word/);
  assert.ok(text.length > 400, `long text survived (${text.length} chars)`);
  assert.equal(result.capped, false);
});

test("engine regionDependencies drive a real manual transaction end-to-end", async () => {
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
  const result = await compactSurfaceRegion(engine.regionDependencies(), session, 1, 4, undefined, {
    owner: null,
    stability: "selected-span"
  }, undefined);
  assert.deepEqual(result.shadowedSeqs, [1, 2, 3, 4]);
  const summaryEvent = sessionEventAt(session, result.summarySeq);
  assert.equal(summaryEvent.data.provider, "dsh-compaction-instant");
  assert.equal(summaryEvent.data.model, "vcc-compiler");
  const checkpoint = sessionEventAt(session, result.summarySeq + 1);
  assert.equal(checkpoint.data.source.plugin, "compact");
  const text = checkpoint.data.content.map((block) => block.text).join("\n");
  assert.match(text, /<compacted-checkpoint>/);
  assert.match(text, /自动压缩: 将 4 个节点 \/ ~400 tokens 编译为 \d+ 条目 \/ ~\d+ tokens/);
  assert.match(text, /## Compiled checkpoint: 4 nodes \(seqs 1-4/);
  assert.match(text, /尾部原文保留: 1 节点 \/ ~100 tokens/);
  assert.match(text, /RECALL: /);
  assert.match(text, /recall/);
  assert.match(text, /search/);
  assert.match(text, /\* read "a\.js" \(seq 2 -> result 3\)/);
  assert.doesNotMatch(text, /-> read: ok/);
  assert.deepEqual(session.surface.nodes, [checkpoint.seq, 5]);
});


