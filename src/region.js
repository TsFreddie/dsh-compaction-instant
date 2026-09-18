/**
 * Surface retention selection and the shared log-recorded compaction
 * transaction for `dsh-compaction-instant`.
 *
 * This module ports the durable protocol of `@deepseek-ai/dsh-compaction-basic`
 * verbatim — lock entry assertions, `compaction/start|summary|end` bracketing,
 * checkpoint replacement provenance, surface-stability checks, flush handling,
 * and the `busy`/`changed`/`summary`/`commit`/`persistence` failure vocabulary —
 * so the instant backend remains a contract-exact drop-in replacement for the
 * `/compact` command and every other `ctx.compaction` consumer.
 *
 * The ONLY replaced step is summarization: where basic awaits an LLM replay,
 * this backend runs the deterministic VCC-style compiler (see compiler.js),
 * which never awaits a model and therefore completes "instantly".
 *
 * @module dsh-compaction-instant/region
 */
import { CompactionId, ManualCompactionError, compactCheckpointSource, toolPairingBalancedAfter, toolPairingBalancedBefore } from "@deepseek-ai/dsh-compaction";
import { createUserMessage, errorChain } from "@deepseek-ai/dsh-llm";
import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import { frameCheckpoint, joinCompiledEntries } from "./compiler.js";
import { sessionEventAt, sessionEvents } from "./session-compat.js";

/**
 * Rejects a compiled checkpoint whose replacement boundaries are no longer
 * the ones it was built from, distinguished from compile and shrink failures
 * so a manual caller can report the two causes differently.
 */
export class SurfaceChangedError extends Error {}

/**
 * Turn index of every surface node, read from the log's `turn/start` events.
 * Nodes before the first turn boundary get turn 0; sessions with no turn
 * events at all produce an all-zero map (selection then keeps everything).
 * @param session - session supplying the event log.
 * @param surfaceNodes - current surface sequences.
 * @returns the per-node turn numbers, in surface order.
 */
function surfaceTurns(session, surfaceNodes) {
  const turnOfSeq = new Map();
  let turn = 0;
  for (const event of sessionEvents(session)) {
    if (event.type === "turn/start") turn = event.data.turn;
    turnOfSeq.set(event.seq, turn);
  }
  return surfaceNodes.map((seq) => turnOfSeq.get(seq) ?? 0);
}

/** First surface index of the turn that contains `index`. */
function turnStartIndex(turns, index) {
  const turn = turns[index];
  let start = index;
  while (start > 0 && turns[start - 1] === turn) start -= 1;
  return start;
}

/**
 * Resolve the next head-anchored range while retaining a recent tail and
 * never splitting an assistant tool-call/result pair.
 *
 * The retained region is selected backward from the latest turn:
 *
 * - `retainTokens` (`> 0`) is an **absolute ceiling**: complete turns are
 *   kept while their total fits, and when even the latest turn alone exceeds
 *   the ceiling the region collapses to the latest node suffix that fits
 *   (then the tool-pair guard may recede one node further). The retained
 *   total never exceeds the ceiling by a whole turn; the balance guard is the
 *   only permitted overshoot. `retainTurns` is the preferred number of whole
 *   turns — it never overrides the ceiling.
 * - `retainTokens` (`0`) keeps exactly `retainTurns` complete recent turns.
 * @param session - session supplying authoritative current surface positions.
 * @param measurement - unified pressure and surface measurement from the conversation meter.
 * @param retainTurns - preferred complete recent turns kept verbatim (>= 1).
 * @param retainTokens - hard retained-region token ceiling; 0 keeps only the preferred turns.
 * @returns the inclusive positional seq range to compact, or `null`.
 */
export function selectCompactableRange(session, measurement, retainTurns, retainTokens) {
  const pricedNodes = measurement.nodes;
  if (pricedNodes.length === 0) return null;
  const surfaceNodes = session.surface.nodes;
  if (surfaceNodes.length !== pricedNodes.length || surfaceNodes.some((seq, index) => seq !== pricedNodes[index]?.seq)) throw new Error("compaction: token-meter surface does not match the current session surface");
  const turns = surfaceTurns(session, surfaceNodes);
  const lastTurn = turns[turns.length - 1];
  if (lastTurn === 0) return null;
  const preferredTurns = Math.max(1, retainTurns);
  let keepFromIdx;
  if (retainTokens > 0) {
    // Absolute ceiling: roll whole turns backward while they fit.
    let scanIdx = turns.length;
    let total = 0;
    let latestWholeIdx;
    while (scanIdx > 0) {
      const startIdx = turnStartIndex(turns, scanIdx - 1);
      let added = 0;
      for (let index = startIdx; index < scanIdx; index += 1) added += pricedNodes[index].tokens;
      if (total + added > retainTokens) break;
      total += added;
      scanIdx = startIdx;
    }
    latestWholeIdx = scanIdx; // first kept node index under the whole-turn rule
    if (scanIdx === turns.length) {
      // Even the latest turn exceeds the ceiling: keep the node suffix that
      // fits inside it (then the balance guard recedes as usual).
      let nodeTotal = 0;
      keepFromIdx = turns.length;
      const latestStart = turnStartIndex(turns, turns.length - 1);
      for (let index = turns.length - 1; index >= latestStart; index -= 1) {
        if (nodeTotal + pricedNodes[index].tokens > retainTokens) break;
        nodeTotal += pricedNodes[index].tokens;
        keepFromIdx = index;
      }
    } else {
      keepFromIdx = latestWholeIdx;
    }
    // Preferred-turn rule: when the ceiling still has room and fewer than
    // preferredTurns whole turns were kept, additional whole turns already
    // cannot fit (loop above broke) — nothing more to do.
    void preferredTurns;
  } else {
    const mandatoryTurnFloor = Math.max(0, lastTurn - preferredTurns + 1);
    keepFromIdx = 0;
    while (keepFromIdx < turns.length && turns[keepFromIdx] < mandatoryTurnFloor) keepFromIdx += 1;
  }
  while (keepFromIdx > 0) {
    if (toolPairingBalancedBefore(session, surfaceNodes[keepFromIdx])) break;
    keepFromIdx -= 1;
  }
  // A `system/message` at surface node 0 may only be rewritten by a
  // system/message over exactly that node, so the range never starts there:
  // the prompt head stays outside every compacted span.
  const firstIdx = sessionEventAt(session, surfaceNodes[0])?.type === "system/message" ? 1 : 0;
  if (keepFromIdx <= firstIdx) return null;
  return {
    start: surfaceNodes[firstIdx],
    end: surfaceNodes[keepFromIdx - 1]
  };
}

/**
 * Run the single compaction transaction over one selected positional span.
 * Selection and validation are read-only. Idle/log validation and
 * `compaction/start` are synchronously adjacent, so the durable opening marker
 * is the compaction lock before the compile yields. Every later failure makes
 * exactly one `compaction/end` attempt; a failed close deliberately leaves the
 * unmatched start detectable.
 * @param dependencies - conversation meter and the deterministic compile hook.
 * @param session - session whose surface is mutated.
 * @param start - inclusive first surface-node seq.
 * @param end - inclusive last surface-node seq.
 * @param agent - retained for signature parity with other backends; the
 *   instant compiler never routes a model call through it.
 * @param options - bracket owner, stability rule, and optional durability checkpoint.
 * @param signal - optional cancellation signal checked at the safe boundaries.
 * @returns the successful durable compaction result.
 */
export async function compactSurfaceRegion(dependencies, session, start, end, agent, options, signal) {
  if (options.owner === null) signal?.throwIfAborted();
  const selection = validateSurfaceRegion(session, start, end);
  const entryState = inspectCompactionEntryState(sessionEvents(session));
  assertCompactionInactive(entryState.unmatchedCompactionStart, entryState.latestEndSeedSeq, "compaction");
  let owner;
  if (options.owner === null) {
    if (entryState.openTurn !== null) throw new ManualCompactionError("busy", "manual compaction: the session already has an open turn");
    owner = null;
  } else {
    if (entryState.openTurn === null) throw new Error("compactRegion: no open turn — automatic compaction events must be enclosed in a turn");
    owner = entryState.openTurn;
  }
  const compactionId = CompactionId(randomUUID());
  const lifecycle = {
    compactionId,
    ...options.sourceCommandId === undefined ? {} : { sourceCommandId: options.sourceCommandId },
    turn: owner
  };
  const startEvent = session.append("compaction/start", lifecycle);
  const assertStable = options.stability === "whole-surface" ? assertWholeSurfaceUnchanged : assertSelectedSpanStable;
  let failure;
  let flushFailure;
  let result;
  let closed = false;
  let closing = false;
  let stage = "summary";
  try {
    const prepared = prepareCompaction(dependencies, session, selection);
    const compiled = await compileCompaction(dependencies, prepared, agent, compactionId, options.sourceCommandId, signal);
    if (options.owner === null) signal?.throwIfAborted();
    assertStable(dependencies, session, compiled);
    stage = "commit";
    const pending = commitCompactionBody(session, startEvent, compiled);
    closing = true;
    const endEvent = session.append("compaction/end", lifecycle);
    closed = true;
    result = completeCompaction(pending, endEvent);
  } catch (error) {
    failure = {
      error,
      stage: closing ? "commit" : stage
    };
    if (!closing) {
      closing = true;
      try {
        session.append("compaction/end", {
          ...lifecycle,
          error: errorChain(error)
        });
        closed = true;
      } catch (closeError) {
        failure = {
          error: closeError,
          stage: "commit"
        };
      }
    }
  }
  if (closed && options.flush !== undefined) try {
    await options.flush();
  } catch (error) {
    flushFailure = error;
  }
  if (options.owner === null) signal?.throwIfAborted();
  if (failure !== undefined) {
    if (options.owner === null) throwManualFailure(failure);
    throw failure.error;
  }
  if (flushFailure !== undefined) throw new ManualCompactionError("persistence", "manual compaction durability checkpoint failed", { cause: flushFailure });
  /* v8 ignore next -- every path without a result records and throws a failure above. */
  if (result === undefined) throw new Error("compaction committed without a result");
  return result;
}

/** Classify one closed manual attempt without weakening cancellation precedence. */
function throwManualFailure(failure) {
  if (failure.stage === "commit") throw new ManualCompactionError("commit", "manual compaction did not commit cleanly", { cause: failure.error });
  if (failure.error instanceof SurfaceChangedError) throw new ManualCompactionError("changed", "the compacted history changed during manual compaction", { cause: failure.error });
  throw new ManualCompactionError("summary", "manual compaction could not produce a smaller checkpoint", { cause: failure.error });
}

/**
 * Reject a durable unmatched compaction marker unless a later constructor-seed
 * boundary proves that its owner belongs to an earlier session lifecycle.
 * @param unmatchedCompactionStart - latest unmatched opening marker, if any.
 * @param latestEndSeedSeq - newest constructor-seed boundary, if any.
 * @param stage - operation label included in the busy diagnostic.
 */
function assertCompactionInactive(unmatchedCompactionStart, latestEndSeedSeq, stage) {
  if (unmatchedCompactionStart === undefined || latestEndSeedSeq !== undefined && latestEndSeedSeq > unmatchedCompactionStart.seq) return;
  throw new ManualCompactionError("busy", `${stage}: compaction already in progress; the session compaction lock is already active`);
}

/**
 * Recheck the durable compaction lock after an asynchronous policy decision.
 * @param session - session whose latest marker state is inspected.
 * @param stage - operation label included in the busy diagnostic.
 */
export function assertNoActiveCompaction(session, stage) {
  const entryState = inspectCompactionEntryState(sessionEvents(session));
  assertCompactionInactive(entryState.unmatchedCompactionStart, entryState.latestEndSeedSeq, stage);
}

/** Validate one requested surface-position span before any work begins. */
export function validateSurfaceRegion(session, start, end) {
  const nodes = session.surface.nodes;
  const startIdx = nodes.indexOf(start);
  const endIdx = nodes.indexOf(end);
  if (startIdx === -1) throw new Error(`compactRegion: start seq ${start} not found in surface`);
  if (endIdx === -1) throw new Error(`compactRegion: end seq ${end} not found in surface`);
  if (startIdx > endIdx) throw new Error(`compactRegion: start seq ${start} (position ${startIdx}) is after end seq ${end} (position ${endIdx}) on the surface`);
  if (!toolPairingBalancedBefore(session, nodes[startIdx])) throw new Error(`compactRegion: start seq ${start} is not a balanced boundary (would split a step's tool-call/result pair)`);
  if (!toolPairingBalancedAfter(session, nodes[endIdx])) throw new Error(`compactRegion: end seq ${end} is not a balanced boundary (would split a step, or the step is still open)`);
  return {
    start,
    end,
    startIdx,
    endIdx,
    shadowedSeqs: nodes.slice(startIdx, endIdx + 1)
  };
}

/** Snapshot pricing for a validated surface range. */
export function prepareCompaction(dependencies, session, selection) {
  const measurement = dependencies.meter.measure(session);
  const selectedNodes = measurement.nodes.slice(selection.startIdx, selection.endIdx + 1);
  if (selectedNodes.length !== selection.shadowedSeqs.length || selectedNodes.some((node, index) => node.seq !== selection.shadowedSeqs[index])) throw new SurfaceChangedError("compaction: selected surface changed before compilation began");
  return {
    ...selection,
    session,
    measurement,
    selectedNodes,
    shadowedTokenCount: selectedNodes.reduce((total, node) => total + node.tokens, 0)
  };
}

/**
 * Run the deterministic region compiler, frame its checkpoint, and price the
 * replacement under the singleton token meter. Mirrors basic's shrink
 * guarantee: a checkpoint that would not reduce the surface is rejected.
 * @param dependencies - conversation meter and the compile hook.
 * @param prepared - priced selection snapshot.
 * @param agent - retained for signature parity; the compiler never routes it.
 * @param compactionId - owning transaction identity for checkpoint provenance.
 * @param sourceCommandId - initiating manual command, when present.
 * @param signal - optional cancellation signal.
 * @returns the compiled summary, provenance, and framed checkpoint message.
 */
async function compileCompaction(dependencies, prepared, agent, compactionId, sourceCommandId, signal) {
  const compiled = await dependencies.compile(prepared, agent, signal);
  // The UI-facing summary IS the compiled body: the checkpoint row expands to
  // exactly the entries the model sees. The body is joined with separators
  // and wrapped in an adaptive Markdown fence, so the UI renders the whole
  // expansion as one tidy code block even when messages contain markdown.
  const verb = sourceCommandId === undefined ? "自动压缩" : "手动 /compact";
  const introLine = `${verb}: 将 ${prepared.shadowedSeqs.length} 个节点 / ~${prepared.shadowedTokenCount} tokens 编译为 ${compiled.entries.length} 条目 / ~${compiled.stats.tokens} tokens`;
  const headerLine = `## Compiled checkpoint: ${prepared.shadowedSeqs.length} nodes (seqs ${prepared.start}-${prepared.end}, ~${prepared.shadowedTokenCount} tokens) — ${compiled.entries.length} entries, ~${compiled.stats.tokens} tokens compiled`;
  // Verbatim retention footer: nodes after the compiled span were never
  // compiled, so they stay in the live surface as original text.
  const retainedNodes = prepared.measurement.nodes.slice(prepared.endIdx + 1);
  const retainedTokenCount = retainedNodes.reduce((total, node) => total + node.tokens, 0);
  const footerLine = retainedNodes.length === 0 ? undefined
    : `尾部原文保留: ${retainedNodes.length} 节点 / ~${retainedTokenCount} tokens（未被压缩,仍在对话中）`;
  const bodyEntries = [
    introLine,
    headerLine,
    ...compiled.entries,
    ...(footerLine === undefined ? [] : [footerLine])
  ];
  const summary = [{ type: "text", text: fenceCode(joinCompiledEntries(bodyEntries)) }];
  const checkpointMessage = createUserMessage({
    content: frameCheckpoint(compiled.entries, headerLine, introLine, footerLine),
    source: compactCheckpointSource(compactionId, sourceCommandId)
  });
  const framedTokenCount = dependencies.meter.estimateMessage(checkpointMessage);
  if (framedTokenCount >= prepared.shadowedTokenCount) throw new Error(`compiled checkpoint is not smaller than the shadowed content (${framedTokenCount} estimated framed tokens >= ${prepared.shadowedTokenCount})`);
  return {
    ...prepared,
    summary,
    provider: compiled.provider,
    model: compiled.model,
    checkpointMessage,
    framedTokenCount
  };
}

/**
 * Wrap text in a Markdown code fence for tidy UI rendering. The fence length
 * adapts to the longest backtick run inside the text — the Markdown-correct
 * way to "escape" embedded ``` blocks (a 3-backtick fence would terminate at
 * them), keeping the content bytes untouched.
 * @param text - body text to fence.
 * @returns the fenced text.
 */
export function fenceCode(text) {
  const runs = text.match(/`+/gu);
  const longest = runs === null ? 0 : Math.max(...runs.map((run) => run.length));
  const fence = "`".repeat(Math.max(3, longest + 1));
  return `${fence}\n${text}\n${fence}`;
}

/** Reject a checkpoint prepared against any earlier surface generation. */
export function assertWholeSurfaceUnchanged(dependencies, session, prepared) {
  if (!isDeepStrictEqual(dependencies.meter.measure(session).nodes, prepared.measurement.nodes)) throw new SurfaceChangedError("compaction: session surface changed during compilation");
}

/**
 * Require only that the selected span remain the same present, contiguous,
 * equally priced, balanced replacement target. Nodes added outside it remain
 * visible and do not invalidate the checkpoint.
 */
export function assertSelectedSpanStable(dependencies, session, prepared) {
  let current;
  try {
    current = validateSurfaceRegion(session, prepared.start, prepared.end);
  } catch (error) {
    throw new SurfaceChangedError("compaction: the selected span is no longer a valid replacement target", { cause: error });
  }
  if (!isDeepStrictEqual([...current.shadowedSeqs], [...prepared.shadowedSeqs])) throw new SurfaceChangedError("compaction: the selected span changed during compilation");
  if (!isDeepStrictEqual(dependencies.meter.measure(session).nodes.slice(current.startIdx, current.endIdx + 1), prepared.selectedNodes)) throw new SurfaceChangedError("compaction: the selected span was rewritten during compilation");
}

/** Append one completed checkpoint record and replacement body without yielding. */
function commitCompactionBody(session, startEvent, compiled) {
  const { start, end, shadowedSeqs, shadowedTokenCount, summary, provider, model, checkpointMessage } = compiled;
  const summaryEvent = session.append("compaction/summary", {
    compactionId: startEvent.data.compactionId,
    ...startEvent.data.sourceCommandId === undefined ? {} : { sourceCommandId: startEvent.data.sourceCommandId },
    summary,
    shadowedRange: {
      start,
      end
    },
    shadowedSeqs: [...shadowedSeqs],
    shadowedTokenCount,
    provider,
    model
  });
  session.append("user/message", checkpointMessage, {
    surfaceOp: {
      op: "replace",
      startSeq: start,
      endSeq: end
    },
    sourceEventSeqs: [
      startEvent.seq,
      summaryEvent.seq,
      ...shadowedSeqs
    ]
  });
  return {
    compactionId: startEvent.data.compactionId,
    ...startEvent.data.sourceCommandId === undefined ? {} : { sourceCommandId: startEvent.data.sourceCommandId },
    startSeq: startEvent.seq,
    summarySeq: summaryEvent.seq,
    summary,
    shadowedRange: {
      start,
      end
    },
    shadowedSeqs: [...shadowedSeqs],
    shadowedTokenCount
  };
}

/** Attach the successfully appended close event to a pending result. */
function completeCompaction(pending, endEvent) {
  return {
    ...pending,
    endSeq: endEvent.seq
  };
}

/** Inspect open-turn, unmatched-compaction, and latest seed-boundary state independently. */
export function inspectCompactionEntryState(events) {
  let openTurn = null;
  let openTurnStateKnown = false;
  let unmatchedCompactionStart;
  let compactionEntryStateKnown = false;
  let latestEndSeedSeq;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (latestEndSeedSeq === undefined && event.type === "session/end-seed") latestEndSeedSeq = event.seq;
    if (!compactionEntryStateKnown) {
      if (event.type === "compaction/start") {
        unmatchedCompactionStart = event;
        compactionEntryStateKnown = true;
      } else if (event.type === "compaction/end") compactionEntryStateKnown = true;
    }
    if (!openTurnStateKnown) {
      if (event.type === "turn/start") {
        openTurn = event.data.turn;
        openTurnStateKnown = true;
      } else if (event.type === "turn/end") openTurnStateKnown = true;
    }
    if (openTurnStateKnown && compactionEntryStateKnown && latestEndSeedSeq !== undefined) break;
  }
  return {
    openTurn,
    unmatchedCompactionStart,
    latestEndSeedSeq
  };
}
