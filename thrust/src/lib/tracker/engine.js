/**
 * The local tracker engine (`tracker serve`) as seen from the browser.
 *
 * The site is on Vercel, but the repos, the forge tokens and the Ollama model live on
 * the user's machine, so pages talk to the engine on the loopback address. The engine
 * only answers the site's origins, and only to a browser holding its pairing token.
 * `tracker serve` prints a link ending in `#pair=<token>`; the fragment never leaves
 * the browser, and `claimPairing` moves it into localStorage and off the address bar.
 */

export const ENGINE_BASE = "http://127.0.0.1:8734";
const TOKEN_KEY = "tracker.engine.token";

export function engineToken() {
  try {
    return window.localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function forgetPairing() {
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

/** Take a `#pair=TOKEN` fragment, if present, and remove it from the URL. */
export function claimPairing() {
  if (typeof window === "undefined") return false;
  const m = window.location.hash.match(/pair=([0-9a-f]{32,})/i);
  if (!m) return false;
  try {
    window.localStorage.setItem(TOKEN_KEY, m[1]);
  } catch {}
  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  return true;
}

export class EngineError extends Error {
  constructor(message, status, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request(path, { method = "GET", body, timeoutMs = 20000, signal } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  if (signal) signal.addEventListener("abort", () => ctrl.abort(), { once: true });
  const headers = {};
  const token = engineToken();
  if (token) headers.authorization = `Bearer ${token}`;
  if (body !== undefined) headers["content-type"] = "application/json";
  let res;
  try {
    res = await fetch(ENGINE_BASE + path, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
      signal: ctrl.signal,
    });
  } catch (e) {
    throw new EngineError(
      ctrl.signal.aborted ? `the engine did not answer within ${Math.round(timeoutMs / 1000)}s` : "the engine is not running",
      0,
      "offline"
    );
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {}
  if (!res.ok) {
    const err = data && data.error;
    throw new EngineError(err ? err.message : `engine answered ${res.status}`, res.status, err ? err.code : "http");
  }
  return data;
}

/** `{state: "offline" | "unpaired" | "ready", version}` — never throws. */
export async function engineStatus() {
  try {
    const h = await request("/health", { timeoutMs: 2500 });
    if (h.service !== "bloodhound-engine") return { state: "offline" };
    return { state: h.paired ? "ready" : "unpaired", version: h.version, capabilities: h.capabilities };
  } catch {
    return { state: "offline" };
  }
}

export const engine = {
  graph: () => request("/graph"),
  recent: (limit = 30) => request(`/repos?limit=${limit}`),
  models: () => request("/models"),
  call: (op, args = {}) => request(`/call/${op}`, { method: "POST", body: args, timeoutMs: 60000 }),
  chat: (body, signal) => request("/chat", { method: "POST", body, timeoutMs: 600000, signal }),
  propose: (op, args = {}) => request("/propose", { method: "POST", body: { op, args } }),
  confirm: (id) => request(`/confirm/${id}`, { method: "POST", timeoutMs: 600000 }),
  reject: (id) => request(`/reject/${id}`, { method: "POST" }),
  pending: () => request("/proposals"),
  /**
   * An action the person started with their own click (Save, Commit, Push): made a
   * proposal and confirmed at once, so it leaves the same trail as a chat proposal.
   * The chat never uses this — its actions wait for a separate Confirm.
   */
  act: async (op, args = {}) => {
    const { proposal } = await request("/propose", { method: "POST", body: { op, args } });
    const r = await request(`/confirm/${proposal.id}`, { method: "POST", timeoutMs: 600000 });
    return r.result;
  },
};

// ── recently inspected repos (per browser) ───────────────────────────────────

const SEEN_KEY = "tracker.seen";

export function seenRepos() {
  try {
    return JSON.parse(window.localStorage.getItem(SEEN_KEY) || "[]");
  } catch {
    return [];
  }
}

/** Remember that the user looked at `repo` (a name), newest first, at most 20. */
export function markSeen(repo) {
  if (!repo) return seenRepos();
  const list = [{ name: repo, at: new Date().toISOString() }, ...seenRepos().filter((r) => r.name !== repo)].slice(0, 20);
  try {
    window.localStorage.setItem(SEEN_KEY, JSON.stringify(list));
  } catch {}
  return list;
}

/** "3 h ago" for an ISO date. */
export function ago(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 86400 * 30) return `${Math.floor(s / 86400)} d ago`;
  if (s < 86400 * 365) return `${Math.floor(s / 86400 / 30)} mo ago`;
  return `${Math.floor(s / 86400 / 365)} y ago`;
}
