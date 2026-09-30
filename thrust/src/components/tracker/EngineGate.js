import { useCallback, useEffect, useState } from "react";
import { claimPairing, engineStatus, forgetPairing } from "@/lib/tracker/engine";

/**
 * Renders `children(status)` once the local tracker engine is running and this
 * browser is paired; otherwise explains the one step that is missing.
 */
export function useEngine() {
  const [status, setStatus] = useState({ state: "checking" });
  const check = useCallback(async () => {
    claimPairing();
    setStatus({ state: "checking" });
    setStatus(await engineStatus());
  }, []);
  useEffect(() => {
    check();
  }, [check]);
  return [status, check];
}

function Code({ children }) {
  return <code className="rounded bg-dark px-2 py-1 font-mono text-[12px] text-accent">{children}</code>;
}

export default function EngineGate({ status, retry, children }) {
  if (status.state === "ready") return children;
  const tone = status.state === "checking" ? "text-muted" : "text-accent";
  return (
    <div className="rounded-2xl border border-primary/15 bg-darkSecondary/60 p-8">
      <div className={`mb-3 font-mono text-xs uppercase tracking-widest ${tone}`}>
        {status.state === "checking" && "looking for the tracker engine…"}
        {status.state === "offline" && "tracker engine not running"}
        {status.state === "unpaired" && "engine found — this browser is not paired"}
      </div>
      {status.state === "offline" && (
        <div className="space-y-3 text-sm text-light/80">
          <p>
            Everything shown here is read from your machine: your repos, your forge tokens (via KeePassXC) and your
            Ollama model. Start the engine in a terminal:
          </p>
          <p><Code>tracker serve</Code></p>
          <p className="text-muted">
            It listens on 127.0.0.1:8734 only, answers only this site, and prints a pairing link to open once.
            Chrome may ask to allow access to devices on your local network — allow it.
          </p>
        </div>
      )}
      {status.state === "unpaired" && (
        <div className="space-y-3 text-sm text-light/80">
          <p>
            Open the pairing link <Code>tracker serve</Code> printed (it ends in <Code>#pair=…</Code>). The token stays in
            this browser; it is never sent to the site.
          </p>
          <button onClick={forgetPairing} className="text-xs text-muted underline hover:text-light">
            forget a stale token
          </button>
        </div>
      )}
      <button
        onClick={retry}
        className="mt-5 rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-dark transition-opacity hover:opacity-90"
      >
        Retry
      </button>
    </div>
  );
}
