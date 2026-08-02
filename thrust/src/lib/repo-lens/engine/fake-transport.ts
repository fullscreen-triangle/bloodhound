/**
 * A {@link Transport} backed by a canned route→response map — the testable seam
 * that drives the whole tab with no Rust engine and no network (spraypaint's
 * fake-runner). Every unit test of `EngineClient`/`analyse-local` runs through this.
 *
 * Not imported by the app; only by tests and stubs. Kept beside the real transport
 * so the two stay interface-identical.
 */

import type { Transport, TransportRequestInit, TransportResponse } from "./transport";

/** A canned reply, or a function of the request for tests that need to branch. */
export type FakeReply =
  | Partial<TransportResponse>
  | ((route: string, init: TransportRequestInit) => Partial<TransportResponse>);

/** Fill in the boring fields so tests only state what they care about. */
function complete(partial: Partial<TransportResponse>): TransportResponse {
  const status = partial.status ?? (partial.ok === false ? 0 : 200);
  const ok = partial.ok ?? (status >= 200 && status < 300);
  const raw = partial.raw ?? (partial.body !== undefined ? JSON.stringify(partial.body) : "");
  return { ok, status, body: partial.body, raw, networkError: partial.networkError };
}

export class FakeTransport implements Transport {
  readonly base: string;
  /** Every request, in order — lets tests assert what the client actually sent. */
  readonly calls: { route: string; init: TransportRequestInit }[] = [];
  private readonly routes: Record<string, FakeReply>;

  constructor(routes: Record<string, FakeReply>, base = "http://127.0.0.1:0") {
    this.routes = routes;
    this.base = base;
  }

  async request(route: string, init: TransportRequestInit = {}): Promise<TransportResponse> {
    this.calls.push({ route, init });
    const key = route.startsWith("/") ? route : "/" + route;
    const reply = this.routes[key];
    if (reply === undefined) {
      // Unmapped route → behave like an unreachable server, not a silent 200.
      return complete({ ok: false, status: 0, networkError: `no fake route for ${key}` });
    }
    return complete(typeof reply === "function" ? reply(route, init) : reply);
  }
}
