/**
 * The typed Client/Error pair over a {@link Transport} — spraypaint's
 * `SpraypaintClient`/`SpraypaintError`, HTTP-native.
 *
 * The client decides WHAT to call and how to read the reply; the transport decides
 * WHERE. Two request styles, deliberately different:
 *
 *  - {@link EngineClient.health} returns a *verdict* and never throws — a missing
 *    engine is a normal state (the engine is an optional download). This is the
 *    exit-code-as-verdict pattern: the outcome is data, not an exception.
 *  - {@link EngineClient.analyse} *throws* {@link EngineError} on any non-2xx or
 *    network failure — once the user has asked to analyse, an unreachable or
 *    erroring engine is a real failure the federation loop should collect.
 */

import type { Transport } from "./transport";
import type { EngineAnalyseResponse, EngineHealth } from "./types";

/** A real engine failure (distinct from the normal "no engine" verdict). */
export class EngineError extends Error {
  readonly route: string;
  readonly status: number;
  readonly networkError?: string;

  constructor(message: string, route: string, status: number, networkError?: string) {
    super(message);
    this.name = "EngineError";
    this.route = route;
    this.status = status;
    this.networkError = networkError;
  }
}

/** The verdict from a health probe — never an exception. */
export type HealthVerdict =
  | { status: "connected"; base: string; info: EngineHealth }
  | { status: "disconnected"; base: string; reason: string };

export interface EngineClientOpts {
  /** Cap on files returned by `/analyse` — mirrors the GitHub path's default. */
  maxFiles?: number;
  /** Cap on bytes per file — mirrors the GitHub path's blob cap. */
  maxFileBytes?: number;
}

const looksLikeHealth = (b: unknown): b is EngineHealth =>
  !!b &&
  typeof b === "object" &&
  (b as EngineHealth).service === "bloodhound-engine" &&
  typeof (b as EngineHealth).version === "string" &&
  Array.isArray((b as EngineHealth).capabilities);

export class EngineClient {
  readonly base: string;
  private readonly transport: Transport;
  private readonly maxFiles: number;
  private readonly maxFileBytes: number;

  constructor(transport: Transport, opts: EngineClientOpts = {}) {
    this.transport = transport;
    this.base = transport.base;
    this.maxFiles = opts.maxFiles ?? 400;
    this.maxFileBytes = opts.maxFileBytes ?? 120_000;
  }

  /**
   * `GET /health` as a verdict. Tolerates every failure — non-2xx, garbage body,
   * connection refused, blocked preflight, timeout — and reports `disconnected`
   * with a human reason. Never throws.
   */
  async health(signal?: AbortSignal): Promise<HealthVerdict> {
    const res = await this.transport.request("/health", { method: "GET", signal });
    if (res.status === 0) {
      return { status: "disconnected", base: this.base, reason: res.networkError ?? "unreachable" };
    }
    if (!res.ok) {
      return { status: "disconnected", base: this.base, reason: `HTTP ${res.status}` };
    }
    if (!looksLikeHealth(res.body)) {
      return { status: "disconnected", base: this.base, reason: "not a bloodhound engine" };
    }
    return { status: "connected", base: this.base, info: res.body };
  }

  /**
   * `POST /analyse`. Throws {@link EngineError} on any non-2xx or network failure —
   * the federation loop collects it into `errors[]`, exactly as the GitHub path's
   * per-repo failures are collected. On success returns the discriminated
   * {@link EngineAnalyseResponse} (thin `snapshot` in v1; `analysed` reserved).
   */
  async analyse(path: string, signal?: AbortSignal): Promise<EngineAnalyseResponse> {
    const res = await this.transport.request("/analyse", {
      method: "POST",
      json: { path, maxFiles: this.maxFiles, maxFileBytes: this.maxFileBytes },
      signal,
    });
    if (res.status === 0) {
      throw new EngineError(
        `engine unreachable at ${this.base}: ${res.networkError ?? "no response"}`,
        "/analyse",
        0,
        res.networkError
      );
    }
    if (!res.ok) {
      const detail =
        res.body && typeof res.body === "object" && "error" in (res.body as Record<string, unknown>)
          ? String((res.body as Record<string, unknown>).error)
          : res.raw.slice(0, 200) || `HTTP ${res.status}`;
      throw new EngineError(`analyse failed: ${detail}`, "/analyse", res.status);
    }
    return res.body as EngineAnalyseResponse;
  }

  // future (out of v1): build / exercise / judge / dispatch — each a new method
  // here, a new route in the engine, and (for judge) a new client verb. The
  // read-only seam above does not change when they land.
}
