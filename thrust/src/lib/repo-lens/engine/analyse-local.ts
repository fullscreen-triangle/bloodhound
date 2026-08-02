/**
 * The local-engine counterpart of `analyse.ts` — the second producer of the exact
 * same `{ federation, errors }` shape the tab already consumes. GitHub and local
 * are sibling producers: neither imports the other, and `repo-lens.js` branches
 * between them at one call site.
 *
 * The thin bridge: the engine supplies bytes (`EngineRepoSnapshot.files`), and the
 * *same* pure `extractAllSymbols` + `computeCharacter` from `chi.ts` compute χ — the
 * one χ implementation the GitHub path also uses. No second source of truth, so no
 * drift. (A future thick engine returns `mode:"analysed"` with χ already computed;
 * this file validates and trusts it then, unchanged for callers.)
 */

import { computeCharacter, extractAllSymbols } from "../chi";
import { localOrigin } from "../model";
import type { AnalysedRepo, Federation } from "../model";
import type { RepoSnapshot, SourceFile } from "../github";
import type { AnalyseProgress } from "../analyse";
import { EngineClient } from "./client";
import type { EngineRepoSnapshot } from "./types";

export interface AnalyseLocalOptions {
  client: EngineClient;
  onProgress?: (p: AnalyseProgress) => void;
  signal?: AbortSignal;
}

/**
 * Adapt the engine's thin snapshot to the existing `RepoSnapshot` interface so no
 * renderer changes. The GitHub-only fields have honest zero/empty defaults — they
 * are cosmetic here (every visualisation reads `symbols`/`character`, both computed
 * below). `ref.owner` is empty because a local repo has no owner; `originName`
 * already uses the path, not the ref, for local identity.
 */
export function buildSnapshot(eng: EngineRepoSnapshot): RepoSnapshot {
  const files: SourceFile[] = eng.files.map((f) => ({
    path: f.path,
    ext: f.ext,
    text: f.text,
    size: f.size,
  }));
  return {
    ref: { owner: "", name: eng.name },
    defaultBranch: eng.defaultBranch ?? "",
    description: eng.description ?? null,
    language: eng.language ?? null,
    topics: [],
    stars: 0,
    openIssues: 0,
    pushedAt: null,
    files,
    skipped: eng.skipped,
    commitCount: 0,
    contributors: [],
    notes: eng.notes,
  };
}

/** Analyse one local repo end to end — the local analogue of `analyseRepo`. */
export async function analyseLocalRepo(
  path: string,
  opts: AnalyseLocalOptions
): Promise<AnalysedRepo> {
  const report = opts.onProgress ?? (() => {});

  report({ repo: path, phase: "fetch", done: 0, total: 1, label: "reading from engine" });
  const resp = await opts.client.analyse(path, opts.signal);

  // Thick engine (deferred): it already ran the analysis. Validate the essentials
  // before trusting it, then hand it straight back.
  if (resp.mode === "analysed") {
    const r = resp.repo;
    if (!r || r.origin?.kind !== "local" || !r.snapshot || !r.symbols || !r.character) {
      throw new Error("engine returned mode:analysed but the payload is not a valid AnalysedRepo");
    }
    report({ repo: path, phase: "done", done: 1, total: 1, label: "done" });
    return r;
  }

  // Thin bridge (v1): the engine gave us bytes; we compute χ here with the same
  // pure functions the GitHub path uses.
  const snapshot = buildSnapshot(resp.repo);

  report({ repo: path, phase: "analyse", done: 0, total: 1, label: "extracting symbols" });
  const symbols = extractAllSymbols(snapshot.files);

  report({ repo: path, phase: "analyse", done: 1, total: 1, label: "computing χ" });
  const character = computeCharacter(symbols);

  report({ repo: path, phase: "done", done: 1, total: 1, label: "done" });
  return {
    origin: localOrigin({ path: resp.repo.path, name: resp.repo.name, commit: resp.repo.commit }),
    snapshot,
    symbols,
    character,
  };
}

/**
 * Analyse several local paths — the local analogue of `analyseFederation`, with
 * byte-identical error parity: each failing path is collected into `errors[]` and
 * the rest proceed.
 */
export async function analyseLocalFederation(
  paths: string[],
  opts: AnalyseLocalOptions
): Promise<{ federation: Federation; errors: { repo: string; error: string }[] }> {
  const repos: AnalysedRepo[] = [];
  const errors: { repo: string; error: string }[] = [];
  for (const raw of paths) {
    const path = raw.trim();
    if (!path) continue;
    try {
      repos.push(await analyseLocalRepo(path, opts));
    } catch (e) {
      errors.push({ repo: path, error: (e as Error).message });
    }
  }
  return { federation: { repos }, errors };
}
