/**
 * Wire schema for the local bloodhound engine (§7.6 of the confluence design).
 *
 * This is the one source of truth for the JSON exchanged over localhost HTTP —
 * the analogue of spraypaint's `types.ts`. If the Rust engine's output changes,
 * these types change with it; there is no second grammar to keep in sync.
 *
 * The engine reads a local repo from the filesystem (something a browser sandbox
 * cannot do) and hands the bytes back; the *analysis* (symbols, χ) still runs in
 * the tab via `chi.ts`, exactly as the GitHub path does. That is the "thin bridge"
 * (default, `mode:"snapshot"`). A future "thick" engine that runs the analysis in
 * Rust is typed here (`mode:"analysed"`) so the union is closed, but v1 does not
 * emit it.
 */

import type { AnalysedRepo } from "../model";

/** `GET /health` — the handshake. Present ⇒ an engine is reachable. */
export interface EngineHealth {
  service: "bloodhound-engine";
  version: string;
  /** Capability flags the engine advertises, e.g. ["analyse"] in v1; later "build"/"judge". */
  capabilities: string[];
}

/**
 * `POST /analyse` request body. Carries only a path and bounds — never a token or
 * any secret (the engine holds the GitHub token locally; it never travels here).
 */
export interface EngineAnalyseRequest {
  path: string;
  /** Max source files to return (mirrors the GitHub path's default of 400). */
  maxFiles?: number;
  /** Max bytes per file (mirrors the GitHub path's ~120 KiB blob cap). */
  maxFileBytes?: number;
}

/**
 * Thin-bridge payload: everything the tab needs to run its own χ. `files` is
 * structurally identical to github.ts's `SourceFile`, so `extractAllSymbols`
 * consumes it unchanged.
 */
export interface EngineRepoSnapshot {
  /** Absolute on-disk path the user pointed at — the local repo's identity. */
  path: string;
  /** Trailing path segment, used as the short label in cramped UI. */
  name: string;
  /** Resolved working-tree HEAD, if the engine could read one. */
  commit?: string;
  defaultBranch?: string;
  description?: string | null;
  language?: string | null;
  files: { path: string; ext: string; text: string; size: number }[];
  /** Files present in the tree but not returned (too big / too many / non-source). */
  skipped: number;
  /** Non-fatal notes surfaced to the UI (truncation, ignore-filtering, etc.). */
  notes: string[];
}

/**
 * Thick-engine payload (deferred): the engine ran the analysis itself and returns
 * a fully-built `AnalysedRepo`. Typed now to close the response union; v1 never
 * emits `mode:"analysed"`, and `analyse-local.ts` validates before trusting it.
 */
export interface EngineAnalysedRepo {
  repo: AnalysedRepo;
}

/**
 * `POST /analyse` response. Discriminated on `mode` so the thick engine drops in
 * later as an alternate branch with no change to any caller.
 */
export type EngineAnalyseResponse =
  | { mode: "snapshot"; repo: EngineRepoSnapshot }
  | { mode: "analysed"; repo: AnalysedRepo };
