/**
 * Integration tests for the durable compaction transaction against a real
 * detached Session, a fake token meter, and a fake compile hook.
 * @module dsh-compaction-instant/test/region
 */
import assert from "node:assert/strict";
import test from "node:test";
import { ManualCompactionError } from "@deepseek-ai/dsh-compaction";
import { createAssistantMessage, createToolResultMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { Session } from "@deepseek-ai/dsh-session";
import { sessionEventAt, sessionEvents } from "../src/session-compat.js";
import { compactSurfaceRegion, fenceCode, selectCompactableRange, SurfaceChangedError } from "../src/region.js";

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

/** Static fake meter pricing 100 tokens per surface node and 10 for any message. */
function makeFakeMeter() {
  return {
    measure: (session) => {
      const nodes = session.surface.nodes.map((seq) => ({ seq, tokens: 100 }));
      return {
        logRevision: 0,
        baseline: { kind: "none", tokens: 0 },
        surfaceDeltaTokens: 0,
        totalTokens: nodes.length * 100,
        surfaceTokens: nodes.length * 100,
        nodes
      };
    },
    estimateMessage: () => 10
  };
}

const fakeCompile = async () => ({
  entries: [{ seq: 1, text: "[user]\ncompiled body" }],
  stats: { tokens: 3 },
  capped: false,
  provider: "test-provider",
  model: "test-compiler"
});

/**
 * Build one detached three-turn session:
 * turn 1 (no tools): user + assistant;
 * turn 2 (tools): user + assistant(tool-call) + tool/result + assistant;
 * turn 3 (no tools): user + assistant.
 * Every surface node prices at 100 tokens through the fake meter.
 */
function makeMultiTurnSession() {
  const user = (text) => createUserMessage({
    content: [{ type: "text", text }],
    source: { kind: "user" }
  });
  const assistant = (text, content = [{ type: "text", text }]) => createAssistantMessage({
    content,
    source: { provider: "p", model: "m" }
  });
  const seed = [];
  let seq = 0;
  const push = (type, data, surfaceOp) => seed.push({
    type,
    seq: seq++,
    time: seq,
    data,
    ...surfaceOp === undefined ? {} : { surfaceOp }
  });
  // Turn 1: seqs 1-2
  push("turn/start", { turn: 1 });
  push("user/message", user("please fix the bug"), "append");
  push("assistant/message", { turn: 1, step: 0, stream: [], message: assistant("on it") }, "append");
  push("turn/end", { turn: 1 });
  // Turn 2: seqs 5-8
  push("turn/start", { turn: 2 });
  push("user/message", user("show me the file"), "append");
  push("assistant/message", { turn: 2, step: 0, stream: [], message: assistant("reading", [
    { type: "text", text: "reading" },
    { type: "tool-call", id: "call-1", name: "read", arguments: '{"file_path":"a.js"}' }
  ]) }, "append");
  push("tool/result", { message: createToolResultMessage({
    callId: "call-1",
    content: [{ type: "text", text: "file content" }],
    isError: false
  }) }, "append");
  push("assistant/message", { turn: 2, step: 0, stream: [], message: assistant("done") }, "append");
  push("turn/end", { turn: 2 });
  // Turn 3: seqs 10-11
  push("turn/start", { turn: 3 });
  push("user/message", user("thank you"), "append");
  push("assistant/message", { turn: 3, step: 0, stream: [], message: assistant("welcome") }, "append");
  push("turn/end", { turn: 3 });
  return Session.create("session-multi", seed);
}

test("selectCompactableRange retains complete turns and never exceeds the token ceiling", () => {
  const session = makeMultiTurnSession();
  const measurement = makeFakeMeter().measure(session); // 8 nodes × 100 tokens
  // Default semantics: 1 mandatory turn (200 tokens), no ceiling → only turn 3 stays.
  const one = selectCompactableRange(session, measurement, 1, 0);
  assert.deepEqual([one.start, one.end], [1, 8]);
  // Two mandatory turns → turns 2+3 stay, only turn 1 compiles.
  const two = selectCompactableRange(session, measurement, 2, 0);
  assert.deepEqual([two.start, two.end], [1, 2]);
  // Ceiling 600 = exactly last two turns (turn 3: 200 + turn 2: 400) → both stay.
  const fits = selectCompactableRange(session, measurement, 1, 600);
  assert.deepEqual([fits.start, fits.end], [1, 2]);
  // Ceiling 599: turn 3 is 200 < 599, but adding turn 2 (400) would overshoot
  // to 600 → never exceeds the ceiling → only turn 3 stays.
  const capped = selectCompactableRange(session, measurement, 1, 599);
  assert.deepEqual([capped.start, capped.end], [1, 8]);
  // A huge ceiling retains everything compilable → nothing left to compile.
  const huge = selectCompactableRange(session, measurement, 1, 1_000_000);
  assert.equal(huge, null);
  // Ceiling 150 < the latest turn alone (200): the ceiling is absolute even
  // against the mandated turn — only the node suffix that fits stays
  // (turn 3's last node, seq 12; the cut before it is balanced).
  const partial = selectCompactableRange(session, measurement, 1, 150);
  assert.deepEqual([partial.start, partial.end], [1, 11]);
  // The preferred-turn rule never overrides the ceiling either (retainTurns=3
  // with ceiling 100 still keeps only the last node).
  const partialPreferred = selectCompactableRange(session, measurement, 3, 100);
  assert.deepEqual([partialPreferred.start, partialPreferred.end], [1, 11]);
  // Selecting with no compilable predecessor → null (single-turn session with
  // the mandatory turn covering the whole surface).
  const whole = selectCompactableRange(makeIdleSession(), makeFakeMeter().measure(makeIdleSession()), 1, 0);
  assert.equal(whole, null);
});

test("compactSurfaceRegion runs a complete manual transaction with a flush", async () => {
  const session = makeIdleSession();
  const meter = makeFakeMeter();
  let flushed = 0;
  const result = await compactSurfaceRegion({ meter, compile: fakeCompile }, session, 1, 4, undefined, {
    owner: null,
    stability: "selected-span",
    flush: async () => {
      flushed += 1;
    }
  }, undefined);
  assert.equal(flushed, 1);
  assert.deepEqual(result.shadowedSeqs, [1, 2, 3, 4]);
  assert.equal(result.shadowedTokenCount, 400);
  assert.deepEqual(result.shadowedRange, { start: 1, end: 4 });
  // The UI-facing summary is the compiled body in one adaptive code fence,
  // opened by a compaction intro line and closed by the retention footer.
  const summaryText = result.summary[0].text;
  assert.match(summaryText, /自动压缩: 将 4 个节点 \/ ~400 tokens 编译为 1 条目 \/ ~3 tokens/);
  assert.match(summaryText, /## Compiled checkpoint: 4 nodes \(seqs 1-4, ~400 tokens\)/);
  assert.match(summaryText, /\[user\]\ncompiled body/);
  assert.match(summaryText, /尾部原文保留: 1 节点 \/ ~100 tokens/);
  const events = sessionEvents(session);
  const startEvent = events[result.startSeq];
  const summaryEvent = events[result.summarySeq];
  const endEvent = events[result.endSeq];
  assert.equal(startEvent.type, "compaction/start");
  assert.equal(summaryEvent.type, "compaction/summary");
  assert.equal(endEvent.type, "compaction/end");
  assert.equal(summaryEvent.data.compactionId, startEvent.data.compactionId);
  assert.equal(summaryEvent.data.provider, "test-provider");
  assert.equal(summaryEvent.data.model, "test-compiler");
  assert.deepEqual(summaryEvent.data.shadowedSeqs, [1, 2, 3, 4]);
  assert.deepEqual(summaryEvent.data.shadowedRange, { start: 1, end: 4 });
  const checkpoint = events[result.summarySeq + 1];
  assert.equal(checkpoint.type, "user/message");
  assert.equal(checkpoint.data.source.plugin, "compact");
  assert.equal(checkpoint.data.source.compactionId, startEvent.data.compactionId);
  assert.deepEqual(session.surface.nodes, [checkpoint.seq, 5]);
});

test("compactSurfaceRegion rejects a checkpoint that does not shrink the surface", async () => {
  const session = makeIdleSession();
  const meter = makeFakeMeter();
  meter.estimateMessage = () => 500;
  await assert.rejects(
    () => compactSurfaceRegion({ meter, compile: fakeCompile }, session, 1, 4, undefined, { owner: null, stability: "selected-span" }, undefined),
    (error) => error instanceof ManualCompactionError && error.code === "summary"
  );
  // The failed manual attempt still closed its bracket with an error marker.
  const events = sessionEvents(session);
  const last = events[events.length - 1];
  assert.equal(last.type, "compaction/end");
  assert.ok(last.data.error.length > 0);
  // Surface untouched.
  assert.deepEqual(session.surface.nodes, [1, 2, 3, 4, 5]);
});

test("compactSurfaceRegion reports busy on an unmatched compaction start", async () => {
  const session = makeIdleSession();
  session.append("compaction/start", { compactionId: "stale", turn: null });
  await assert.rejects(
    () => compactSurfaceRegion({ meter: makeFakeMeter(), compile: fakeCompile }, session, 1, 4, undefined, { owner: null, stability: "selected-span" }, undefined),
    (error) => error instanceof ManualCompactionError && error.code === "busy"
  );
});

test("compactSurfaceRegion reports busy for manual compaction inside an open turn", async () => {
  const session = makeIdleSession();
  session.append("turn/start", { turn: 2 });
  await assert.rejects(
    () => compactSurfaceRegion({ meter: makeFakeMeter(), compile: fakeCompile }, session, 1, 4, undefined, { owner: null, stability: "selected-span" }, undefined),
    (error) => error instanceof ManualCompactionError && error.code === "busy"
  );
});

test("compactSurfaceRegion reports changed when the surface moves during compilation", async () => {
  const session = makeIdleSession();
  const meter = makeFakeMeter();
  let calls = 0;
  const original = meter.measure;
  meter.measure = (target) => {
    calls += 1;
    const measured = original(target);
    if (calls > 1) measured.nodes = measured.nodes.map((node) => ({ ...node, tokens: 999 }));
    return measured;
  };
  await assert.rejects(
    () => compactSurfaceRegion({ meter, compile: fakeCompile }, session, 1, 4, undefined, { owner: null, stability: "selected-span" }, undefined),
    (error) => error instanceof ManualCompactionError && error.code === "changed"
  );
});

test("compactSurfaceRegion runs an automatic in-turn transaction", async () => {
  const session = makeIdleSession();
  session.append("turn/start", { turn: 2 });
  const result = await compactSurfaceRegion({ meter: makeFakeMeter(), compile: fakeCompile }, session, 1, 4, undefined, {
    owner: "current-turn",
    stability: "whole-surface"
  }, undefined);
  assert.deepEqual(result.shadowedSeqs, [1, 2, 3, 4]);
  assert.equal(sessionEventAt(session, result.startSeq).data.turn, 2);
});

test("SurfaceChangedError is exported for callers to distinguish", () => {
  assert.equal(new SurfaceChangedError("x") instanceof Error, true);
});

test("fenceCode wraps text and adapts to embedded fences", () => {
  assert.equal(fenceCode("a\nb"), "```\na\nb\n```");
  // A 3-backtick run inside forces a longer fence; content bytes untouched.
  assert.equal(fenceCode("x\n```\ncode\n```\ny"), "````\nx\n```\ncode\n```\ny\n````");
  // Longer runs escalate further.
  assert.equal(fenceCode("````"), "`````\n````\n`````");
  assert.equal(fenceCode(""), "```\n\n```");
  assert.equal(fenceCode("no backticks"), "```\nno backticks\n```");
});
