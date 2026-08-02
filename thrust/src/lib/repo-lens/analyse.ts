/**
 * Top-level analysis: token + repo refs → fetched snapshots → symbols → χ.
 * Produces the `Federation` both input paths (st-Hurbert & AI) run against, and
 * the numbers the dashboard renders.
 */

import { fetchRepo, parseRepoRef, RepoRef } from "./github";
import { computeCharacter, extractAllSymbols } from "./chi";
import { githubOrigin } from "./model";
import type { AnalysedRepo, Federation } from "./model";

export interface AnalyseProgress {
  repo: string;
  phase: "fetch" | "analyse" | "done";
  done: number;
  total: number;
  label: string;
}

export interface AnalyseOptions {
  token: string;
  maxFiles?: number;
  onProgress?: (p: AnalyseProgress) => void;
}

/** Analyse one repo end to end. */
export async function analyseRepo(ref: RepoRef, opts: AnalyseOptions): Promise<AnalysedRepo> {
  const report = opts.onProgress ?? (() => {});
  const name = `${ref.owner}/${ref.name}`;

  const snapshot = await fetchRepo(ref, {
    token: opts.token,
    maxFiles: opts.maxFiles,
    onProgress: (done, total, label) =>
      report({ repo: name, phase: "fetch", done, total, label }),
  });

  report({ repo: name, phase: "analyse", done: 0, total: 1, label: "extracting symbols" });
  const symbols = extractAllSymbols(snapshot.files);

  report({ repo: name, phase: "analyse", done: 1, total: 1, label: "computing χ" });
  const character = computeCharacter(symbols);

  report({ repo: name, phase: "done", done: 1, total: 1, label: "done" });
  return { origin: githubOrigin(snapshot), snapshot, symbols, character };
}

/** Analyse several repos (a federation). Repo strings may be owner/name or URLs. */
export async function analyseFederation(
  repoInputs: string[],
  opts: AnalyseOptions
): Promise<{ federation: Federation; errors: { repo: string; error: string }[] }> {
  const repos: AnalysedRepo[] = [];
  const errors: { repo: string; error: string }[] = [];
  for (const input of repoInputs) {
    let ref: RepoRef;
    try {
      ref = parseRepoRef(input);
    } catch (e) {
      errors.push({ repo: input, error: (e as Error).message });
      continue;
    }
    try {
      repos.push(await analyseRepo(ref, opts));
    } catch (e) {
      errors.push({ repo: input, error: (e as Error).message });
    }
  }
  return { federation: { repos }, errors };
}
