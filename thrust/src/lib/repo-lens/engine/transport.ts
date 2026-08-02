/**
 * The swappable *where* of an engine call (spraypaint's Runner, made HTTP-native).
 *
 * A {@link Transport} turns a route + request init into a {@link TransportResponse}.
 * The one discipline it must keep — the reason it exists as a seam — is
 * **resolve-never-throw**: a refused connection, a blocked preflight, a timeout, or
 * a non-2xx status are all *returned as data*, never raised. A missing local engine
 * is a normal state of this app, not an exception; only the higher layers
 * ({@link import("./client").EngineClient}) decide which of those outcomes is a
 * real error worth throwing on.
 *
 * Everything here is browser-safe: `fetch` + `AbortController` only, no Node. So
 * unlike spraypaint's Node-only `runner-node` there is no subpath split.
 */

/**
 * The uniform outcome of a transport call. `status === 0` is the sentinel for
 * "never reached a server" (connection refused, DNS/loopback blocked, preflight
 * denied, or aborted) — distinct from any real HTTP status the engine returned.
 */
export interface TransportResponse {
  /** True iff a response came back with a 2xx status. */
  ok: boolean;
  /** HTTP status, or 0 when no response was received at all. */
  status: number;
  /** Parsed JSON body when the response was JSON; otherwise undefined. */
  body: unknown;
  /** Raw response text (kept for error messages / non-JSON bodies). */
  raw: string;
  /** Present only when the request never reached the server (status 0). */
  networkError?: string;
}

export interface TransportRequestInit {
  method?: "GET" | "POST";
  /** JSON-serialisable body; set as `application/json`. */
  json?: unknown;
  /** Caller's abort signal (composed with the per-request timeout). */
  signal?: AbortSignal;
}

/** The seam: a Transport knows a base URL and how to make one request against it. */
export interface Transport {
  readonly base: string;
  request(route: string, init?: TransportRequestInit): Promise<TransportResponse>;
}

/** Default per-request timeout — a local engine on loopback answers in single-digit ms. */
const DEFAULT_TIMEOUT_MS = 4000;

/**
 * The real transport: `fetch` against a localhost engine.
 *
 * Uses `127.0.0.1` (not the string `localhost`, which can resolve to `::1` first and
 * stall, and is treated less uniformly by the Secure-Context loopback exception).
 * Catches the fetch rejection — which is exactly how the browser reports connection
 * refused and a denied Private Network Access preflight — into a `status:0`
 * `networkError`, so the caller sees a verdict, never a thrown error.
 */
export class FetchTransport implements Transport {
  readonly base: string;
  private readonly timeoutMs: number;

  constructor(base = "http://127.0.0.1", opts: { timeoutMs?: number } = {}) {
    this.base = base.replace(/\/+$/, "");
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  }

  async request(route: string, init: TransportRequestInit = {}): Promise<TransportResponse> {
    const url = this.base + (route.startsWith("/") ? route : "/" + route);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.timeoutMs);
    // Abort our request if the caller's signal fires, without losing our timeout.
    if (init.signal) {
      if (init.signal.aborted) ctrl.abort();
      else init.signal.addEventListener("abort", () => ctrl.abort(), { once: true });
    }

    try {
      const res = await fetch(url, {
        method: init.method ?? "GET",
        signal: ctrl.signal,
        headers: init.json !== undefined ? { "content-type": "application/json" } : undefined,
        body: init.json !== undefined ? JSON.stringify(init.json) : undefined,
      });
      const raw = await res.text();
      let body: unknown;
      try {
        body = raw.length ? JSON.parse(raw) : undefined;
      } catch {
        body = undefined; // non-JSON body; `raw` still carries it for diagnostics
      }
      return { ok: res.ok, status: res.status, body, raw };
    } catch (err) {
      // fetch rejects on: connection refused, DNS failure, blocked loopback/PNA
      // preflight, or abort (timeout or caller cancel). All are "never reached the
      // server" from our point of view → status 0, surfaced as a networkError.
      const networkError =
        ctrl.signal.aborted && !(init.signal && init.signal.aborted)
          ? `timeout after ${this.timeoutMs}ms`
          : err instanceof Error
            ? err.message
            : String(err);
      return { ok: false, status: 0, body: undefined, raw: "", networkError };
    } finally {
      clearTimeout(timer);
    }
  }
}
