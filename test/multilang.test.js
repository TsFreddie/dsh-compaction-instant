/**
 * Multilingual tests for the compiler and recall layers.
 *
 * The VCC tokenizer is ASCII-class-based: `[a-zA-Z]+` runs, `[0-9]+` runs,
 * and every other non-whitespace code unit as its own token. These tests pin
 * the deterministic behavior for CJK, Cyrillic, Arabic, accented Latin, and
 * emoji, and — most importantly — that every truncation/excerpt boundary
 * respects UTF-16 surrogate pairs, so no cut ever emits a lone half-emoji.
 * @module dsh-compaction-instant/test/multilang
 */
import assert from "node:assert/strict";
import test from "node:test";
import { createAssistantMessage, createToolResultMessage, createUserMessage } from "@deepseek-ai/dsh-llm";
import { Session } from "@deepseek-ai/dsh-session";
import { sessionEventAt, sessionEvents } from "../src/session-compat.js";
import {
  compileNoisePatterns,
  compileNodes,
  compileRegion,
  countTokens,
  DEFAULT_NOISE_PATTERNS,
  estimateEntryTokens,
  excerptToolResult,
  sanitize,
  stripNoiseXml,
  tokenize,
  truncateTokens
} from "../src/compiler.js";
import { DEFAULT_MAX_RECALL_TOKENS, projectMessageText, recallSession } from "../src/recall.js";

const CONFIG = {
  maxTokens: 8192,
  textTokens: 512,
  userTextTokens: 1024,
  toolCallTokens: 128,
  toolResultExcerptTokens: 256,
  includeReasoning: false,
  stripNoiseXml: true,
  noisePatterns: compileNoisePatterns(DEFAULT_NOISE_PATTERNS),
  toolKeyFields: {}
};

/** Assert a string contains no unpaired UTF-16 surrogates. */
function assertNoLoneSurrogates(text) {
  for (let index = 0; index < text.length; index += 1) {
    const code = text.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      assert.ok(index + 1 < text.length, `lone high surrogate at end of ${JSON.stringify(text)}`);
      const next = text.charCodeAt(index + 1);
      assert.ok(next >= 0xdc00 && next <= 0xdfff, `lone high surrogate at ${index} in ${JSON.stringify(text)}`);
      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      assert.fail(`lone low surrogate at ${index} in ${JSON.stringify(text)}`);
    }
  }
}

// ── tokenization across scripts (VCC-parity, deterministic) ─────────────────

test("CJK text tokenizes one code point per token", () => {
  assert.deepEqual(tokenize("你好，世界！"), ["你", "好", "，", "世", "界", "！"]);
  assert.equal(countTokens("你好，世界！"), 6);
  assert.equal(countTokens("日本語のテキストです"), 10);
  assert.equal(countTokens("한국어 텍스트"), 6);
});

test("Cyrillic and Arabic tokenize per non-ASCII code unit", () => {
  assert.deepEqual(tokenize("Привет мир"), ["П", "р", "и", "в", "е", "т", " ", "м", "и", "р"]);
  assert.equal(countTokens("Привет мир"), 9);
  assert.equal(countTokens("مرحبا بالعالم"), 12);
});

test("accented Latin splits at non-ASCII boundaries", () => {
  assert.deepEqual(tokenize("café"), ["caf", "é"]);
  assert.equal(countTokens("café"), 2);
  assert.deepEqual(tokenize("naïve"), ["na", "ï", "ve"]);
  assert.equal(countTokens("naïve"), 3);
});

test("emoji tokenize as two code units each", () => {
  assert.equal(countTokens("😀"), 2);
  assert.equal(countTokens("a😀b"), 4);
});

// ── truncation and excerpts never split code points ─────────────────────────

test("truncateTokens keeps exact CJK tokens and adds the provenance note", () => {
  const out = truncateTokens("你好世界这是一个很长的句子", 5, "seq 9");
  assert.equal(out.text, "你好世界这...(truncated from seq 9)");
  assert.equal(out.truncated, true);
});

test("truncateTokens never splits a surrogate pair (mixed ASCII + emoji)", () => {
  const out = truncateTokens("a" + "😀".repeat(50) + "Z", 8, "seq 1");
  assert.equal(out.truncated, true);
  assertNoLoneSurrogates(out.text);
  assert.match(out.text, /\.\.\.\(truncated from seq 1\)$/);
});

test("excerptToolResult anchors the multilingual tail and splits no pairs", () => {
  const text = "😀".repeat(300) + "重要结尾TAIL";
  const out = excerptToolResult(text, 64, "seq 3");
  assert.match(out, /重要结尾TAIL$/);
  assertNoLoneSurrogates(out);
});

test("estimateEntryTokens uses the max of token count and char density", () => {
  assert.equal(estimateEntryTokens("你好世界"), 4);
  assert.equal(estimateEntryTokens("a".repeat(100)), 30);
  assert.equal(estimateEntryTokens("😀"), 2);
});

// ── sanitize and noise stripping on multilingual content ────────────────────

test("sanitize leaves multibyte text intact and normalizes line endings", () => {
  assert.equal(sanitize("你好\r\n世界\t"), "你好\n世界\t");
  assert.equal(sanitize("مرحبا"), "مرحبا");
});

test("stripNoiseXml removes noise wrappers around multilingual content", () => {
  const text = "请保留这句话 <system-reminder>中文噪声内容</system-reminder> 结尾";
  assert.equal(stripNoiseXml(text, CONFIG.noisePatterns), "请保留这句话  结尾");
});

// ── compileNodes across languages ───────────────────────────────────────────

test("compileNodes compiles a multilingual conversation without splitting code points", () => {
  const nodes = [
    { seq: 1, message: { role: "user", content: [{ type: "text", text: "请修复这个 bug。" }], source: { kind: "user" } } },
    { seq: 2, message: { role: "assistant", content: [
      { type: "text", text: "わかりました。" },
      { type: "tool-call", id: "c1", name: "read", arguments: '{"file_path":"源文件.js"}' }
    ] } },
    { seq: 3, message: { role: "user", content: [{ type: "tool-result", toolCallId: "c1", content: [{ type: "text", text: "مرحبا بالعالم" }] }] } },
    { seq: 4, message: { role: "assistant", content: [{ type: "text", text: "수정 완료" }] } }
  ];
  const { entries } = compileNodes(nodes, CONFIG);
  const text = entries.map((entry) => entry.text).join("\n");
  assert.match(text, /\[user\]\n请修复这个 bug。/);
  assert.match(text, /\[assistant\]\nわかりました。/);
  assert.match(text, /\* read "源文件\.js" \(seq 2 -> result 3\)/);
  assert.doesNotMatch(text, /-> read: ok/);
  assert.doesNotMatch(text, /مرحبا بالعالم/);
  assert.match(text, /수정 완료/);
  for (const entry of entries) assertNoLoneSurrogates(entry.text);
});

test("compileNodes truncates a long CJK block at the token budget with exact characters", () => {
  const long = "字".repeat(3000);
  const nodes = [{ seq: 1, message: { role: "user", content: [{ type: "text", text: long }], source: { kind: "user" } } }];
  const { entries } = compileNodes(nodes, CONFIG);
  const kept = entries[0].text;
  const body = kept.replace(/^\[user\]\n/u, "").replace(/\.\.\.\(truncated from seq 1\)$/u, "");
  assert.equal(body, "字".repeat(1024));
  assert.match(kept, /\.\.\.\(truncated from seq 1\)$/);
  assertNoLoneSurrogates(kept);
});

test("compileRegion cap enforcement emits valid text for emoji-heavy input", () => {
  const nodes = [];
  for (let seq = 1; seq <= 30; seq += 1) {
    nodes.push({ seq, message: { role: "user", content: [{ type: "text", text: `😀😀${seq}号消息😀😀` }], source: { kind: "user" } } });
  }
  const { entries, stats } = compileRegion(nodes, { ...CONFIG, maxTokens: 120 });
  assert.ok(stats.tokens <= 120 + 64, `tokens ${stats.tokens} within cap`);
  for (const entry of entries) assertNoLoneSurrogates(entry.text);
});

// ── recall across languages ─────────────────────────────────────────────────

test("recallSession restores multilingual originals byte-exact", () => {
  const user = createUserMessage({
    content: [{ type: "text", text: "请修复这个 bug。😀" }],
    source: { kind: "user" }
  });
  const assistant = createAssistantMessage({
    content: [
      { type: "text", text: "Привет, мир!" },
      { type: "tool-call", id: "c1", name: "read", arguments: '{"file_path":"源文件.js"}' }
    ],
    source: { provider: "p", model: "m" }
  });
  const result = createToolResultMessage({
    callId: "c1",
    content: [{ type: "text", text: "مرحبا بالعالم — 수정 완료" }],
    isError: false
  });
  const seed = [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: user, surfaceOp: "append" },
    { type: "assistant/message", seq: 2, time: 3, data: { turn: 1, step: 0, stream: [], message: assistant }, surfaceOp: "append" },
    { type: "tool/result", seq: 3, time: 4, data: { message: result }, surfaceOp: "append" },
    { type: "turn/end", seq: 4, time: 5, data: { turn: 1 } }
  ];
  const session = Session.create("session-multilang", seed);
  const recalled = recallSession(session, [{ start: 1, end: 3 }], { maxRecallTokens: DEFAULT_MAX_RECALL_TOKENS });
  assert.equal(recalled.recalled, 3);
  assert.equal(recalled.truncated, false);
  assert.match(recalled.text, /\[seq 1: user\]\n请修复这个 bug。😀/);
  assert.match(recalled.text, /Привет, мир!/);
  assert.match(recalled.text, /\{"file_path":"源文件\.js"\}/);
  assert.match(recalled.text, /مرحبا بالعالم — 수정 완료/);
  assertNoLoneSurrogates(recalled.text);
});

test("projectMessageText keeps multilingual reasoning and arguments intact", () => {
  const text = projectMessageText({
    role: "assistant",
    content: [
      { type: "reasoning", text: "这里需要修复。😀" },
      { type: "tool-call", id: "c", name: "bash", arguments: '{"command":"echo 你好"}' }
    ]
  });
  assert.match(text, /\[reasoning\]\n这里需要修复。😀/);
  assert.match(text, /\{"command":"echo 你好"\}/);
  assertNoLoneSurrogates(text);
});

test("recall budget truncation keeps multibyte content intact", () => {
  const user = createUserMessage({
    content: [{ type: "text", text: "字".repeat(500) }],
    source: { kind: "user" }
  });
  const seed = [
    { type: "turn/start", seq: 0, time: 1, data: { turn: 1 } },
    { type: "user/message", seq: 1, time: 2, data: user, surfaceOp: "append" },
    { type: "turn/end", seq: 2, time: 3, data: { turn: 1 } }
  ];
  const session = Session.create("session-multilang-budget", seed);
  const recalled = recallSession(session, [{ start: 1, end: 1 }], { maxRecallTokens: 40 });
  assert.equal(recalled.truncated, true);
  assert.match(recalled.text, /\.\.\.\(truncated from recall budget\)$/);
  assertNoLoneSurrogates(recalled.text);
});
