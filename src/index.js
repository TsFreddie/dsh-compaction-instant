/**
 * Instant replay-aware compaction backend: VCC-style deterministic context
 * compilation for the DeepSeek Harness.
 *
 * A contract-exact drop-in replacement for `@deepseek-ai/dsh-compaction-basic`:
 * identical `ctx.compaction` seam (trigger policy, retention, durable
 * transaction protocol, `busy`/`changed`/`summary`/`commit`/`persistence`
 * failures), but the summarization step is replaced by the offline
 * conversation compiler ported from https://github.com/lllyasviel/VCC
 * (see compiler.js). No model call is made, so one compaction completes in
 * milliseconds with original tokens only — instant and near-lossless.
 *
 * @module dsh-compaction-instant
 */
import z from "@deepseek-ai/schemastery";
import { appendFileSync } from "node:fs";
import { CompactionEngine, ManualCompactionError } from "@deepseek-ai/dsh-compaction";
import { CONTEXT_WINDOW_EXCEEDED_CODE } from "@deepseek-ai/dsh-llm";
import { assertNever, deepFreeze } from "@deepseek-ai/dsh-util-values";
import { compileNoisePatterns, compileRegion, COMPILER_REV, DEFAULT_ARG_TOOLS, DEFAULT_NOISE_PATTERNS, isCheckpointSource } from "./compiler.js";
import { assertNoActiveCompaction, compactSurfaceRegion, selectCompactableRange } from "./region.js";
import { sessionEventAt, sessionEvents } from "./session-compat.js";

// ── configuration resolution ───────────────────────────────────────────────

/** Default request-pressure fraction for every routed model. */
const DEFAULT_THRESHOLD_RATIO = 0.5;
/** Default number of complete recent turns kept verbatim. */
const DEFAULT_RETAIN_TURNS = 1;
/** Default retained-region token ceiling (never exceeded by turn extension). */
const DEFAULT_RETAIN_TOKENS = 5120;
/** Default total cap for one compiled checkpoint, in compiler tokens. */
const DEFAULT_MAX_TOKENS = 8192;
/** Default scaled-cap fraction of the shadowed token count. */
const DEFAULT_CHECKPOINT_SCALE = 0.1;
/** Default absolute ceiling for the scaled checkpoint cap. */
const DEFAULT_CHECKPOINT_CAP = 65536;
/** Default per-block budgets for the compiled view (compiler tokens). */
const DEFAULT_TEXT_TOKENS = 512;
const DEFAULT_USER_TEXT_TOKENS = 1024;
const DEFAULT_TOOL_CALL_TOKENS = 128;
const DEFAULT_TOOL_RESULT_EXCERPT_TOKENS = 256;
/** Backend provenance recorded on the `compaction/summary` event. */
const COMPILER_PROVIDER = "dsh-compaction-instant";
const COMPILER_MODEL = "vcc-compiler";

/** Fields shared by top-level defaults and exact-target overrides. */
const POLICY_CONFIG_KEYS = [
  "thresholdRatio",
  "retainTurns",
  "retainTokens",
  // Accepted for drop-in configuration compatibility with compaction-basic;
  // the instant backend never routes a model, so these are inert.
  "summarizationProvider",
  "summarizationModel",
  "maxTokens",
  "compactionRetries",
  "maxOverflowRetries"
];
/** Compiler-specific tuning keys. */
const COMPILER_CONFIG_KEYS = [
  "textTokens",
  "userTextTokens",
  "toolCallTokens",
  "toolResultExcerptTokens",
  "includeReasoning",
  "stripNoiseXml",
  "noisePatterns",
  "toolKeyFields",
  "toolArgTools",
  "hideTools",
  "debug",
  "debugLogPath"
];
/** Remaining checkpoint-cap keys (top-level only). `checkpointScale` is deprecated-inert. */
const MANUAL_CONFIG_KEYS = [
  "checkpointScale",
  "checkpointCap"
];
/** Complete public top-level configuration key set. */
const INSTANT_COMPACT_CONFIG_KEYS = new Set([
  ...POLICY_CONFIG_KEYS,
  ...COMPILER_CONFIG_KEYS,
  ...MANUAL_CONFIG_KEYS,
  "modelPolicies",
  "auto"
]);
/** Complete exact-target override key set. */
const MODEL_POLICY_KEYS = new Set([
  "provider",
  "model",
  ...POLICY_CONFIG_KEYS
]);

/** Target-specific pressure configuration failure eligible for warning suppression. */
export class TargetPressureConfigError extends Error {
  /**
   * @param targetKey - exact provider/model route used as the warning key.
   * @param message - actionable configuration failure detail.
   */
  constructor(targetKey, message) {
    super(message);
    this.targetKey = targetKey;
  }
}

/**
 * Pick the settings-exposed subset out of a composition entry config, so the
 * namespace's `base` layer carries exactly the fields its schema declares.
 */
function pickSettingsFields(config) {
  return {
    ...config.checkpointCap !== undefined ? { checkpointCap: config.checkpointCap } : {},
    ...config.auto !== undefined ? { auto: config.auto } : {},
    ...config.thresholdRatio !== undefined ? { thresholdRatio: config.thresholdRatio } : {},
    ...config.retainTurns !== undefined ? { retainTurns: config.retainTurns } : {},
    ...config.retainTokens !== undefined ? { retainTokens: config.retainTokens } : {}
  };
}

/**
 * Resolve and validate service defaults plus exact-target partial overrides.
 * @param config - untrusted plugin configuration after Loader normalization.
 * @returns detached immutable defaults and validated exact-target overrides.
 */
export function resolveConfig(config = {}) {
  validateKeys(config, INSTANT_COMPACT_CONFIG_KEYS, "InstantCompactionConfig");
  validatePolicy(config, "InstantCompactionConfig");
  if (config.auto !== undefined && typeof config.auto !== "boolean") throw new Error("InstantCompactionConfig: auto must be a boolean");
  const thresholdRatio = config.thresholdRatio ?? DEFAULT_THRESHOLD_RATIO;
  const retainTurns = config.retainTurns ?? DEFAULT_RETAIN_TURNS;
  const retainTokens = config.retainTokens ?? DEFAULT_RETAIN_TOKENS;
  const modelPolicies = resolveModelPolicies(config.modelPolicies);
  const debug = config.debug === true || (typeof process !== "undefined" && process.env?.DSH_COMPACTION_DEBUG === "1");
  const debugLogPath = config.debugLogPath ?? (typeof process !== "undefined" && process.env?.DSH_HOME ? `${process.env.DSH_HOME}/compaction-debug.log` : "/tmp/dsh-compaction-debug.log");
  return deepFreeze({
    thresholdRatio,
    retainTurns,
    retainTokens,
    maxTokens: config.maxTokens ?? DEFAULT_MAX_TOKENS,
    checkpointScale: config.checkpointScale ?? DEFAULT_CHECKPOINT_SCALE,
    checkpointCap: config.checkpointCap ?? DEFAULT_CHECKPOINT_CAP,
    compactionRetries: config.compactionRetries ?? 1,
    maxOverflowRetries: config.maxOverflowRetries ?? 1,
    modelPolicies,
    auto: config.auto ?? true,
    textTokens: config.textTokens ?? DEFAULT_TEXT_TOKENS,
    userTextTokens: config.userTextTokens ?? DEFAULT_USER_TEXT_TOKENS,
    toolCallTokens: config.toolCallTokens ?? DEFAULT_TOOL_CALL_TOKENS,
    toolResultExcerptTokens: config.toolResultExcerptTokens ?? DEFAULT_TOOL_RESULT_EXCERPT_TOKENS,
    includeReasoning: config.includeReasoning ?? false,
    stripNoiseXml: config.stripNoiseXml ?? true,
    noisePatterns: compileNoisePatterns(config.noisePatterns !== undefined && config.noisePatterns.length > 0 ? config.noisePatterns : DEFAULT_NOISE_PATTERNS),
    toolKeyFields: resolveToolKeyFields(config.toolKeyFields),
    toolArgTools: resolveToolNameList(config.toolArgTools, DEFAULT_ARG_TOOLS, "toolArgTools"),
    hideTools: resolveToolNameList(config.hideTools, [], "hideTools"),
    debug,
    debugLogPath,
    ...debug ? {
      debugSink: (line) => {
        try {
          appendFileSync(debugLogPath, `${line}\n`, "utf8");
        } catch (error) {
          console.error(`[dsh-compaction-instant] debug sink failed (${String(error)})`);
        }
      }
    } : {}
  });
}

/**
 * Merge the exact provider/model override over the validated default policy.
 * @param config - validated service defaults and override table.
 * @param target - exact durable provider/model route to match.
 * @returns detached immutable policy before model-capacity scaling.
 */
export function resolveTargetPolicy(config, target) {
  const override = config.modelPolicies.find((policy) => policy.provider === target.provider && policy.model === target.model);
  return deepFreeze({
    target: {
      provider: target.provider,
      model: target.model
    },
    thresholdRatio: override?.thresholdRatio ?? config.thresholdRatio,
    retainTurns: override?.retainTurns ?? config.retainTurns,
    retainTokens: override?.retainTokens ?? config.retainTokens,
    summarizationProvider: override?.summarizationProvider ?? config.summarizationProvider ?? "",
    summarizationModel: override?.summarizationModel ?? config.summarizationModel ?? "",
    maxTokens: override?.maxTokens ?? config.maxTokens,
    compactionRetries: override?.compactionRetries ?? config.compactionRetries,
    maxOverflowRetries: override?.maxOverflowRetries ?? config.maxOverflowRetries
  });
}

/**
 * Scale one routed policy into concrete token budgets for its model capacity.
 * @param policy - merged policy for the exact routed target.
 * @param contextWindow - positive adapter-owned capacity for that target.
 * @returns detached immutable pressure and retention budgets.
 */
export function resolveCompactSpec(policy, contextWindow) {
  const targetKey = `${policy.target.provider}/${policy.target.model}`;
  if (!Number.isInteger(contextWindow) || contextWindow <= 0) throw new TargetPressureConfigError(targetKey, `InstantCompactionConfig: contextWindow (${contextWindow}) must be a positive integer`);
  const thresholdTokens = Math.floor(contextWindow * policy.thresholdRatio);
  return deepFreeze({
    target: { ...policy.target },
    contextWindow,
    thresholdRatio: policy.thresholdRatio,
    thresholdTokens,
    retainTurns: policy.retainTurns,
    retainTokens: policy.retainTokens,
    maxTokens: policy.maxTokens,
    compactionRetries: policy.compactionRetries,
    maxOverflowRetries: policy.maxOverflowRetries
  });
}

/** Validate, detach, and reject duplicate exact-target policies. */
function resolveModelPolicies(configured) {
  if (configured === undefined) return [];
  if (!Array.isArray(configured)) throw new Error("InstantCompactionConfig: modelPolicies must be an array");
  const seen = new Set();
  return configured.map((source, index) => {
    assertModelPolicy(source, `InstantCompactionConfig: modelPolicies[${index}]`);
    const key = `${source.provider}\u0000${source.model}`;
    if (seen.has(key)) throw new Error(`InstantCompactionConfig: duplicate model policy for ${source.provider}/${source.model}`);
    seen.add(key);
    return { ...source };
  });
}

/** Validate one untrusted exact-target override and narrow its public type. */
function assertModelPolicy(source, name) {
  if (!isUnknownRecord(source)) throw new Error(`${name} must be an object`);
  validateKeys(source, MODEL_POLICY_KEYS, name);
  assertNonEmptyString(`${name}.provider`, source.provider);
  assertNonEmptyString(`${name}.model`, source.model);
  validatePolicy(source, name);
}

/** Validate the fields common to defaults and exact-target partial overrides. */
function validatePolicy(config, name) {
  const thresholdRatio = config.thresholdRatio;
  const retainTurns = config.retainTurns;
  const retainTokens = config.retainTokens;
  const maxTokens = config.maxTokens;
  const compactionRetries = config.compactionRetries;
  const maxOverflowRetries = config.maxOverflowRetries;
  if (thresholdRatio !== undefined) assertRatio(`${name}.thresholdRatio`, thresholdRatio);
  if (retainTurns !== undefined) assertPositiveInteger(`${name}.retainTurns`, retainTurns);
  if (retainTokens !== undefined) assertNonNegativeInteger(`${name}.retainTokens`, retainTokens);
  if (maxTokens !== undefined) assertPositiveInteger(`${name}.maxTokens`, maxTokens);
  if (compactionRetries !== undefined) assertNonNegativeInteger(`${name}.compactionRetries`, compactionRetries);
  if (maxOverflowRetries !== undefined) assertNonNegativeInteger(`${name}.maxOverflowRetries`, maxOverflowRetries);
  validateSummarizationPair(config, name);
  if (config.checkpointScale !== undefined) assertRatio(`${name}.checkpointScale`, config.checkpointScale);
  if (config.checkpointCap !== undefined) assertPositiveInteger(`${name}.checkpointCap`, config.checkpointCap);
  for (const key of ["textTokens", "userTextTokens", "toolCallTokens", "toolResultExcerptTokens"]) {
    if (config[key] !== undefined) assertPositiveInteger(`${name}.${key}`, config[key]);
  }
  if (config.includeReasoning !== undefined && typeof config.includeReasoning !== "boolean") throw new Error(`${name}.includeReasoning must be a boolean`);
  if (config.stripNoiseXml !== undefined && typeof config.stripNoiseXml !== "boolean") throw new Error(`${name}.stripNoiseXml must be a boolean`);
  if (config.noisePatterns !== undefined) {
    if (!Array.isArray(config.noisePatterns) || config.noisePatterns.some((pattern) => typeof pattern !== "string")) throw new Error(`${name}.noisePatterns must be an array of strings`);
  }
  for (const key of ["toolArgTools", "hideTools"]) {
    if (config[key] !== undefined && (!Array.isArray(config[key]) || config[key].some((entry) => typeof entry !== "string" || entry.length === 0))) {
      throw new Error(`${name}.${key} must be an array of non-empty strings`);
    }
  }
  if (config.debug !== undefined && typeof config.debug !== "boolean") throw new Error(`${name}.debug must be a boolean`);
  if (config.debugLogPath !== undefined && typeof config.debugLogPath !== "string") throw new Error(`${name}.debugLogPath must be a string`);
}

/** Detach a validated tool-name list (whitelist or hidden set). */
function resolveToolNameList(configured, fallback, key) {
  // The cordis config pipeline validates rows through the plugin's schemastery
  // schema, whose `~standard` adapter injects `[]` for every absent array key.
  // An empty list therefore means "unset", not "empty on purpose": fall back to
  // the defaults so a missing whitelist keeps rendering tool arguments.
  if (configured === undefined || configured.length === 0) return [...fallback];
  if (!Array.isArray(configured) || configured.some((entry) => typeof entry !== "string" || entry.length === 0)) {
    throw new Error(`InstantCompactionConfig: ${key} must be an array of non-empty strings`);
  }
  return [...new Set(configured)];
}

/** Validate the optional tool-name → preferred-argument-field map. */
function resolveToolKeyFields(configured) {
  if (configured === undefined) return {};
  if (!isUnknownRecord(configured)) throw new Error("InstantCompactionConfig: toolKeyFields must be an object");
  for (const [tool, field] of Object.entries(configured)) {
    if (typeof tool !== "string" || tool.length === 0) throw new Error("InstantCompactionConfig: toolKeyFields keys must be non-empty strings");
    if (typeof field !== "string" || field.length === 0) throw new Error(`InstantCompactionConfig: toolKeyFields["${tool}"] must be a non-empty string`);
  }
  return { ...configured };
}

/** Require one scope to omit, clear, or replace the summarization target as a pair (inert but validated). */
function validateSummarizationPair(config, name) {
  const provider = config.summarizationProvider;
  const model = config.summarizationModel;
  if (provider !== undefined && typeof provider !== "string") throw new Error(`${name}.summarizationProvider must be a string`);
  if (model !== undefined && typeof model !== "string") throw new Error(`${name}.summarizationModel must be a string`);
  if (provider === undefined && model === undefined) return;
  if (provider === undefined || model === undefined || provider.length === 0 !== (model.length === 0)) throw new Error(`${name}: summarizationProvider and summarizationModel must be set together as an empty or non-empty pair`);
}

/** Reject stale or misspelled keys before defaults can hide them. */
function validateKeys(config, keys, name) {
  for (const key of Object.keys(config)) if (!keys.has(key)) throw new Error(`${name}: unknown key "${key}"`);
}

function isUnknownRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertNonEmptyString(name, value) {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${name} must be a non-empty string`);
}

function assertPositiveInteger(name, value) {
  if (typeof value !== "number" || !Number.isInteger(value) || value <= 0) throw new Error(`${name} (${String(value)}) must be a positive integer`);
}

function assertNonNegativeInteger(name, value) {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0) throw new Error(`${name} (${String(value)}) must be a non-negative integer`);
}

function assertRatio(name, value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || value > 1) throw new Error(`${name} (${String(value)}) must be a number in (0, 1]`);
}

// ── the engine ──────────────────────────────────────────────────────────────

/** Resolve the exact provider/model durably routed for the latest request. */
export function routedTarget(session) {
  const config = session.requestHeader()?.config;
  if (config === undefined || config.provider.length === 0 || config.model.length === 0) return;
  return {
    provider: config.provider,
    model: config.model
  };
}

const thresholdRatioSchema = z.number();
const retainTurnsSchema = z.number().step(1).min(1);
const retainTokensSchema = z.number().step(1).min(0);
const checkpointScaleSchema = z.number();
const checkpointCapSchema = z.number().step(1).min(1);
const summarizationProviderSchema = z.string();
const summarizationModelSchema = z.string();
const maxTokensSchema = z.number().step(1).min(1);
const compactionRetriesSchema = z.number().step(1).min(0);
const maxOverflowRetriesSchema = z.number().step(1).min(0);
const modelPolicy = z.object({
  provider: z.string().required(),
  model: z.string().required(),
  thresholdRatio: thresholdRatioSchema,
  retainTurns: retainTurnsSchema,
  retainTokens: retainTokensSchema,
  summarizationProvider: summarizationProviderSchema,
  summarizationModel: summarizationModelSchema,
  maxTokens: maxTokensSchema,
  compactionRetries: compactionRetriesSchema,
  maxOverflowRetries: maxOverflowRetriesSchema
});

/**
 * Deterministic compaction backend using `ctx.tokenMeter` for pressure,
 * retention, and checkpoint-convergence pricing, and the offline VCC-style
 * compiler instead of an LLM summarizer — the replacement step never routes a
 * model request, so compaction is instant and keeps original tokens only.
 *
 * `compile()` is the sole subclass customization hook; the replay-free
 * compiler and durable mutation strategy stay fixed so every pricing decision
 * uses the singleton token meter.
 */
/**
 * Emit one engine-side debug line through the resolved sink when debug is on.
 * @param config - resolved configuration.
 * @param line - the diagnostic text.
 */
function engineDebug(config, line) {
  if (config.debug !== true) return;
  const text = `[dsh-compaction-instant] engine: ${line}`;
  try {
    if (typeof config.debugSink === "function") config.debugSink(text);
  } catch {
    /* a failing debug sink must never break the engine */
  }
  try {
    console.error(text);
  } catch {
    /* stderr may be unavailable in embedded contexts */
  }
}

export class InstantCompactionEngine extends CompactionEngine {
  static inject = [
    "llm",
    "tokenMeter",
    "sessions"
  ];
  static Config = z.object({
    thresholdRatio: thresholdRatioSchema,
    retainTurns: retainTurnsSchema,
    retainTokens: retainTokensSchema,
    checkpointScale: checkpointScaleSchema,
    checkpointCap: checkpointCapSchema,
    summarizationProvider: summarizationProviderSchema,
    summarizationModel: summarizationModelSchema,
    maxTokens: maxTokensSchema,
    compactionRetries: compactionRetriesSchema,
    maxOverflowRetries: maxOverflowRetriesSchema,
    modelPolicies: z.array(modelPolicy),
    auto: z.boolean(),
    textTokens: maxTokensSchema,
    userTextTokens: maxTokensSchema,
    toolCallTokens: maxTokensSchema,
    toolResultExcerptTokens: maxTokensSchema,
    includeReasoning: z.boolean(),
    stripNoiseXml: z.boolean(),
    noisePatterns: z.array(z.string()),
    toolKeyFields: z.dict(z.string()),
    toolArgTools: z.array(z.string()),
    hideTools: z.array(z.string()),
    debug: z.boolean(),
    debugLogPath: z.string()
  });
  /** Settings namespace for user-owned instant-compaction preferences. */
  static SETTINGS_NAMESPACE = "compaction-instant";
  /**
   * Settings-exposed subset of the engine configuration. Defaults mirror the
   * engine's own `DEFAULT_*` constants so the resolved settings layer is
   * exactly what `resolveConfig` would compute. Debug behavior stays
   * engine-config-only (`debug`, `debugLogPath`, `DSH_COMPACTION_DEBUG`).
   */
  static SETTINGS_SCHEMA = z.object({
    checkpointCap: z.number().step(1).min(1).default(DEFAULT_CHECKPOINT_CAP),
    auto: z.boolean().default(true),
    thresholdRatio: z.number().min(0).max(1).default(DEFAULT_THRESHOLD_RATIO),
    retainTurns: z.number().step(1).min(1).default(DEFAULT_RETAIN_TURNS),
    retainTokens: z.number().step(1).min(0).default(DEFAULT_RETAIN_TOKENS)
  });
  /** Resolved and validated compaction configuration. */
  config;
  warnedPressureConfigTargets = new Set();
  overflowRetries = new WeakMap();
  overflowAgents = new WeakMap();
  constructor(ctx, config = {}) {
    super(ctx);
    this.entry = config;
    this.source = () => config;
    this.config = resolveConfig(config);
    engineDebug(this.config, `constructed rev=${COMPILER_REV} debugLog=${this.config.debugLogPath} argTools=[${this.config.toolArgTools.join(",")}] hideTools=[${this.config.hideTools.join(",")}]`);
    this._autoDisposer = null;
    this._autoActive = false;
    this._installSettingsSection(ctx);
    this._syncAuto();
  }
  /**
   * Re-resolve the engine configuration from the authoritative source (the
   * settings namespace when mounted, the composition entry otherwise) and
   * re-arm automatic compaction when the `auto` flag flipped.
   */
  _reloadConfig() {
    const raw = this.source?.() ?? this.entry ?? {};
    this.config = resolveConfig(raw);
    this._syncAuto();
    engineDebug(this.config, `config reloaded rev=${COMPILER_REV} source=${this.source ? "settings" : "entry"} argTools=[${this.config.toolArgTools.join(",")}] auto=${this.config.auto}`);
  }
  /**
   * Mount the optional settings namespace over this engine's configuration
   * source. With a settings service present, the user layer (settings.yaml)
   * overlays the composition entry's exposed subset; without one, the engine
   * keeps reading the entry exactly as composed. The whole engine validation
   * runs on every settings write, so a value that would break the engine is
   * refused before it is persisted.
   */
  _installSettingsSection(ctx) {
    const entry = this.entry;
    const base = pickSettingsFields(entry);
    // dsh 0.1.2 removed installSettingsSection/settingsNamespace. Register
    // through ctx.settings when the service exists; otherwise keep the
    // composition entry as the config source.
    ctx.inject(["settings"], (scoped) => {
      const scope = scoped.settings.register(
        InstantCompactionEngine.SETTINGS_NAMESPACE,
        InstantCompactionEngine.SETTINGS_SCHEMA,
        {
          base,
          validate: (value) => {
            resolveConfig({ ...entry, ...value });
          },
        },
      );
      this.source = () => ({ ...entry, ...scope.get() });
      scoped.effect(() => () => {
        this.source = () => entry;
        this._reloadConfig();
      });
      this._reloadConfig();
      scope.watch(() => {
        this.source = () => ({ ...entry, ...scope.get() });
        this._reloadConfig();
      });
    });
  }
  /** Register or dispose the automatic-compaction listeners with the `auto` flag. */
  _syncAuto() {
    const active = this.config.auto === true;
    if (active === this._autoActive) return;
    if (this._autoDisposer !== null) {
      this._autoDisposer();
      this._autoDisposer = null;
    }
    this._autoActive = active;
    if (active) this._autoDisposer = this._registerAutomaticCompaction();
  }
  /**
   * Register automatic between-step pressure and model-request overflow
   * recovery. `compactIfNeeded` stays dynamically dispatched so subclass
   * overrides are honored at event time.
   */
  _registerAutomaticCompaction() {
    const { ctx } = this;
    const logResult = (result, trigger) => {
      ctx.logger.info(`compaction (${trigger}): shadowed ${result.shadowedSeqs.length} surface nodes (seqs ${result.shadowedRange.start}-${result.shadowedRange.end}, ~${result.shadowedTokenCount} tokens) with the instant compiler`);
    };
    const disposers = [];
    disposers.push(ctx.on("agent/pre-step", async ({ agent, signal }, next) => {
      if (!signal.aborted) try {
        const result = await this.compactIfNeeded(agent, "pressure", signal);
        if (result !== null) logResult(result, "step pressure");
      } catch (error) {
        if (error instanceof TargetPressureConfigError) {
          if (this.warnedPressureConfigTargets.has(error.targetKey)) return next();
          this.warnedPressureConfigTargets.add(error.targetKey);
        }
        const message = error instanceof Error ? error.message : String(error);
        ctx.logger.warn(`step compaction failed: ${message}; continuing the turn`);
      }
      return next();
    }));
    disposers.push(ctx.on("agent/status", ({ agent, status }) => {
      if (status === "idle") this.overflowRetries.delete(agent);
    }));
    disposers.push(ctx.on("session/event", (session, event) => {
      if (event.type !== "assistant/message") return;
      const agent = this.overflowAgents.get(session);
      if (agent !== undefined) this.overflowRetries.delete(agent);
    }));
    disposers.push(ctx.on("agent/request-error", async ({ agent, failure, signal }, next) => {
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted) return next();
      this.overflowAgents.set(agent.session, agent);
      const target = routedTarget(agent.session);
      if (target === undefined) return next();
      const policy = resolveTargetPolicy(this.config, target);
      const retries = this.overflowRetries.get(agent) ?? 0;
      if (retries >= policy.maxOverflowRetries) return next();
      const generation = agent.session.surface.replaceGeneration;
      let result;
      try {
        result = await this.compactIfNeeded(agent, "context-overflow", signal);
      } catch (recoveryError) {
        const message = recoveryError instanceof Error ? recoveryError.message : String(recoveryError);
        if (!signal.aborted && agent.session.surface.replaceGeneration > generation) {
          ctx.logger.warn(`context-overflow compaction failed after durable surface progress: ${message}; retrying from the replacement surface`);
          this.overflowRetries.set(agent, retries + 1);
          return { kind: "retry" };
        }
        ctx.logger.warn(`context-overflow compaction failed: ${message}; ${signal.aborted ? "cancellation prevents retry" : "preserving the original request error"}`);
        return next();
      }
      if (signal.aborted || agent.session.surface.replaceGeneration <= generation) return next();
      if (result !== null) logResult(result, "context overflow recovery");
      this.overflowRetries.set(agent, retries + 1);
      return { kind: "retry" };
    }));
    return () => {
      for (const dispose of disposers) {
        try {
          dispose();
        } catch {
          /* a failing listener teardown must not break the disposal sweep */
        }
      }
    };
  }
  /**
   * Resolve the total cap for one compiled checkpoint: always the configured
   * `checkpointCap` (default 65536 compiler tokens). The old proportional
   * scaling (`checkpointScale`) and floor (`maxTokens`) were removed — a
   * checkpoint either fits under the cap (near-lossless) or elides down to it.
   * @param shadowedTokenCount - retained for subclass override signature
   * parity; the cap no longer scales with the shadowed span.
   * @returns the cap in compiler tokens.
   */
  effectiveMaxTokens(shadowedTokenCount) {
    return this.config.checkpointCap;
  }
  /**
   * Compile one priced region with the deterministic VCC-style compiler.
   * Override this sole hook for a hybrid or remote compiler. The returned
   * entries are framed and priced by the shared transaction.
   * @param prepared - priced selection snapshot (selection, session, pricing).
   * @param agent - retained for signature parity; the default compiler never
   *   routes a model call through it.
   * @param signal - optional cancellation checked before the compile.
   * @returns ordered checkpoint entries plus backend provenance and stats.
   */  async compile(prepared, agent, signal) {
    signal?.throwIfAborted();
    const nodes = prepared.shadowedSeqs.map((seq) => {
      const event = sessionEventAt(prepared.session, seq);
      if (event === undefined || event.seq !== seq) throw new Error(`compaction: surface seq ${seq} has no matching session event (corrupt surface)`);
      return {
        seq,
        message: prepared.session.deriveEventMessage(event)
      };
    });
    // Session-wide checkpoint ordinals (1 = oldest compaction): the compiler
    // uses them to leave `[checkpoint N]` lines when a prior checkpoint is
    // elided under cap pressure, so the agent can recall the dropped layer.
    const checkpointOrdinals = new Map();
    let checkpointCount = 0;
    for (const event of sessionEvents(prepared.session)) {
      if (event.type === "user/message" && isCheckpointSource(event.data?.source)) {
        checkpointCount += 1;
        checkpointOrdinals.set(event.seq, checkpointCount);
      }
    }
    engineDebug(this.config, `compile span=${prepared.shadowedSeqs.length} seqs=${prepared.shadowedSeqs[0]}-${prepared.shadowedSeqs[prepared.shadowedSeqs.length - 1]} shadowedTokens=${prepared.shadowedTokenCount} cap=${this.effectiveMaxTokens(prepared.shadowedTokenCount)} checkpoints=${checkpointCount}`);
    const { entries, stats, capped } = compileRegion(nodes, {
      ...this.config,
      maxTokens: this.effectiveMaxTokens(prepared.shadowedTokenCount),
      checkpointOrdinals
    });
    engineDebug(this.config, `compile done entries=${entries.length} tokens=${stats.tokens} capped=${capped} toolCalls=${stats.toolCalls} toolResults=${stats.toolResults}`);
    return {
      entries,
      stats,
      capped,
      provider: COMPILER_PROVIDER,
      model: COMPILER_MODEL
    };
  }
  /**
   * Compact for replayed step-boundary pressure or one provider-confirmed
   * context overflow. Both triggers price the latest durable routed request
   * envelope; overflow bypasses the normal threshold and retained-tail policy
   * so it can force one useful balanced reduction.
   * @param agent - agent whose latest durable routed request is measured.
   * @param trigger - normal step-boundary pressure or context-overflow recovery.
   * @param signal - live turn cancellation signal forwarded to the compile.
   * @returns the latest compaction result, or `null` when none ran.
   */
  async compactIfNeeded(agent, trigger, signal) {
    const target = routedTarget(agent.session);
    if (target === undefined) return null;
    const policy = resolveTargetPolicy(this.config, target);
    const meter = this.ctx.tokenMeter;
    let measurement = meter.measure(agent.session);
    switch (trigger) {
      case "context-overflow": break;
      case "pressure": break;
      /* v8 ignore next -- closed-union exhaustiveness guard */
      default: assertNever(trigger, "compaction trigger");
    }
    const prune = this.ctx.get("toolResultPruner");
    if (trigger === "context-overflow") {
      if (prune !== undefined) {
        prune.pruneSession(agent.session);
        measurement = meter.measure(agent.session);
      }
      const range = selectCompactableRange(agent.session, measurement, 1, 0);
      if (range === null) return null;
      return this.compactRegion(range.start, range.end, agent, signal);
    }
    const context = (await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)).context;
    assertNoActiveCompaction(agent.session, "automatic pressure compaction");
    const targetKey = `${target.provider}/${target.model}`;
    if (context === undefined) throw new TargetPressureConfigError(targetKey, `compaction-instant: no context capacity for ${targetKey}; configure contextWindow on that adapter model`);
    const spec = resolveCompactSpec(policy, context.contextWindow);
    if (measurement.totalTokens < spec.thresholdTokens) return null;
    if (prune !== undefined) {
      prune.pruneSession(agent.session);
      measurement = meter.measure(agent.session);
    }
    if (measurement.totalTokens < spec.thresholdTokens) return null;
    let result = null;
    for (let attempt = 0; attempt <= spec.compactionRetries; attempt += 1) {
      const range = selectCompactableRange(agent.session, measurement, spec.retainTurns, spec.retainTokens);
      if (range === null) {
        /* v8 ignore else -- concrete replacement preserves a compactable checkpoint; subclass hooks cannot mutate it. */
        if (result === null) return null;
        /* v8 ignore next -- paired with the defensive post-success branch above. */
        break;
      }
      result = await this.compactRegion(range.start, range.end, agent, signal);
      measurement = meter.measure(agent.session);
      if (measurement.totalTokens < spec.thresholdTokens) return result;
    }
    throw new Error(`compaction still above threshold after ${spec.compactionRetries + 1} compaction attempts (${measurement.totalTokens} estimated tokens >= threshold ${spec.thresholdTokens})`);
  }
  /**
   * Compact one inclusive positional range from the agent-owned surface using
   * the effective token meter for all retention and shrink pricing.
   * @param start - inclusive first surface-node seq.
   * @param end - inclusive last surface-node seq.
   * @param agent - owner of the target session, retained for signature parity.
   * @param signal - optional cancellation signal.
   * @returns the successful durable compaction result.
   */
  async compactRegion(start, end, agent, signal) {
    return compactSurfaceRegion(this.regionDependencies(), agent.session, start, end, agent, {
      owner: "current-turn",
      stability: "whole-surface"
    }, signal);
  }
  /**
   * Force one useful idle-session compaction below the pressure threshold, and
   * resolve only after its standalone marker pair is durably checkpointed.
   * @param agent - idle agent whose next-turn admission this call reserves.
   * @param signal - cancellation scoped to this compaction request.
   * @param sourceCommandId - initiating command identity for presentation correlation.
   * @returns the committed result, or `null` when no safe useful range exists.
   */
  compactNow(agent, signal, sourceCommandId) {
    signal.throwIfAborted();
    try {
      return agent.runMaintenance(async (agentSignal) => {
        const operationSignal = AbortSignal.any([agentSignal, signal]);
        try {
          operationSignal.throwIfAborted();
          const measurement = this.ctx.tokenMeter.measure(agent.session);
          const range = selectCompactableRange(agent.session, measurement, this.config.retainTurns, this.config.retainTokens);
          if (range === null) return null;
          return await compactSurfaceRegion(this.regionDependencies(), agent.session, range.start, range.end, agent, {
            owner: null,
            stability: "selected-span",
            ...sourceCommandId === undefined ? {} : { sourceCommandId },
            flush: async () => {
              await this.ctx.sessions.flush(agent.session);
            }
          }, operationSignal);
        } catch (error) {
          if (agentSignal.aborted && operationSignal.reason === agentSignal.reason) throw new ManualCompactionError("cancelled", "manual compaction was cancelled", { cause: error });
          operationSignal.throwIfAborted();
          throw error;
        }
      });
    } catch (error) {
      throw new ManualCompactionError("busy", "manual compaction requires an idle agent with no waking queued work", { cause: error });
    }
  }
  /** Bind the effective token meter and dynamically dispatched compile hook. */
  regionDependencies() {
    return {
      meter: this.ctx.tokenMeter,
      compile: (prepared, owner, abort) => this.compile(prepared, owner, abort)
    };
  }
}

export { COMPILER_MODEL, COMPILER_PROVIDER };
export default InstantCompactionEngine;
