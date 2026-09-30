import Head from "next/head";
import { useRouter } from "next/router";
import { useState } from "react";
import { safeNext } from "@/lib/auth";

export default function Login() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (res.ok) {
        // A full load, so the gated page is fetched with the new cookie.
        window.location.assign(safeNext(router.query.next));
        return;
      }
      const data = await res.json().catch(() => ({}));
      setError(data.error || `login failed (${res.status})`);
    } catch {
      setError("could not reach the site");
    }
    setBusy(false);
  };

  return (
    <>
      <Head>
        <title>Sign in | Bloodhound</title>
        <meta name="robots" content="noindex" />
      </Head>
      <div className="flex min-h-screen items-center justify-center bg-dark px-6">
        <form onSubmit={submit} className="w-full max-w-sm rounded-2xl border border-primary/15 bg-darkSecondary/70 p-8 shadow-glow">
          <div className="mb-1 font-mono text-xs uppercase tracking-widest text-primary">Bloodhound</div>
          <h1 className="mb-6 text-xl font-semibold text-light">Sign in</h1>
          <label className="mb-2 block text-xs text-muted" htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            autoFocus
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mb-4 w-full rounded-lg border border-primary/20 bg-surface px-3 py-2 text-sm text-light focus:border-primary/60 focus:outline-none"
          />
          {error && <div className="mb-4 text-xs text-danger">{error}</div>}
          <button
            type="submit"
            disabled={busy || !password}
            className="w-full rounded-lg bg-primary py-2 text-sm font-semibold text-dark disabled:opacity-50"
          >
            {busy ? "Checking…" : "Sign in"}
          </button>
        </form>
      </div>
    </>
  );
}
