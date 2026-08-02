/**
 * AI-prompt path (the *uninformed user's* door): natural language → st-Hurbert,
 * grounded by the real compiler. This mirrors Purpose Tool A's generate → validate
 * → repair loop: the model drafts st-Hurbert, the compiler (not the model) is the
 * ground truth, and a rejected draft is fed back once with the error for repair.
 *
 * Model calls go through the Next.js API proxy (`/api/repo-lens/ai`) so tokens/keys
 * stay server-side; the provider is `huggingface` or `ollama`.
 */

import { compile } from "./sthurbert";
import type { Federation } from "./model";
import { repoName } from "./model";

export type Provider = "huggingface" | "ollama";

export interface AiRequest {
  provider: Provider;
  /** HF model id or Ollama model name. */
  model: string;
  prompt: string;
}

/** A compact description of the federation the model is allowed to query. */
export function federationBrief(fed: Federation): string {
  const repos = fed.repos
    .map((r) => {
      const c = r.character;
      const top = c.salient.slice(0, 3).map((s) => s.file).join(", ");
      return `- ${repoName(r)} — χ=${c.chi.toFixed(2)}, ${c.blocks} blocks, ${r.symbols.length} symbols; key files: ${top}`;
    })
    .join("\n");
  return repos || "(no repos analysed)";
}

/** The grammar/examples that ground generation (examples, not facts). */
const GRAMMAR_PRIMER = `st-Hurbert repo-query language (subset). One statement per line or separated by ';'.
Commands:
  navigate <repo|*>            select a repo by name, or * for the whole federation
  slice [<kind>] [where <cond>]  filter working symbols by kind and/or condition
  find "<text>"                search symbol names/snippets
  show <sense|chi|salient|fragments|files|symbols>
Conditions: <field> <op> <value>   field in {name,kind,file,line}
  op in { == != ~ contains > < >= <= } ; combine with 'and' / 'or'
Examples:
  navigate * ; show chi
  navigate myrepo ; slice fn where name contains parse ; show symbols
  navigate * ; find "entropy" ; show files
  slice where kind == class and file contains model ; show symbols`;

function buildMessages(fed: Federation, userPrompt: string, priorError?: string) {
  const system = `You translate a user's natural-language request into a valid st-Hurbert program that queries an analysed group of code repositories. Output ONLY st-Hurbert source, no prose, no code fences.

${GRAMMAR_PRIMER}

Repos available (use these exact names in 'navigate'):
${federationBrief(fed)}`;
  const user = priorError
    ? `${userPrompt}\n\nYour previous program did not compile: ${priorError}\nReturn a corrected st-Hurbert program only.`
    : userPrompt;
  return { system, user };
}

export interface GenerateResult {
  ok: boolean;
  /** The final st-Hurbert source (valid if ok). */
  code?: string;
  /** Raw model outputs across attempts, for transparency in the UI. */
  attempts: { raw: string; error?: string }[];
  error?: string;
}

/** Strip accidental markdown fences / prose the model may add. */
function cleanCode(raw: string): string {
  let s = raw.trim();
  const fence = s.match(/```(?:\w+)?\n([\s\S]*?)```/);
  if (fence) s = fence[1].trim();
  return s;
}

async function callProxy(req: AiRequest, system: string, user: string): Promise<string> {
  const res = await fetch("/api/repo-lens/ai", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider: req.provider, model: req.model, system, user }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `AI proxy returned ${res.status}`);
  }
  const data = await res.json();
  return data.text ?? "";
}

/**
 * Generate st-Hurbert from a prompt, validating against the real compiler and
 * repairing once on failure. Returns the first draft that compiles.
 */
export async function generateStHurbert(
  fed: Federation,
  req: AiRequest,
  maxRepairs = 1
): Promise<GenerateResult> {
  const attempts: { raw: string; error?: string }[] = [];
  let priorError: string | undefined;

  for (let attempt = 0; attempt <= maxRepairs; attempt++) {
    const { system, user } = buildMessages(fed, req.prompt, priorError);
    let raw: string;
    try {
      raw = await callProxy(req, system, user);
    } catch (e) {
      return { ok: false, attempts, error: (e as Error).message };
    }
    const code = cleanCode(raw);
    const compiled = compile(code);
    if (compiled.ok) {
      attempts.push({ raw: code });
      return { ok: true, code, attempts };
    }
    const errMsg = compiled.error?.message ?? "unknown compile error";
    attempts.push({ raw: code, error: errMsg });
    priorError = errMsg;
  }

  return {
    ok: false,
    attempts,
    error: `model could not produce compilable st-Hurbert after ${maxRepairs + 1} attempt(s)`,
  };
}
