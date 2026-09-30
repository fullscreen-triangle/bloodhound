import { useState } from "react";
import Chart from "@/components/tracker/Chart";
import { engine } from "@/lib/tracker/engine";

/**
 * An action the tracker is ready to take but has not taken. It runs on the engine
 * only when the person presses Confirm, and then shows what actually happened.
 */
export default function ProposalCard({ proposal, onDone }) {
  const [state, setState] = useState("pending"); // pending | running | done | failed | cancelled
  const [out, setOut] = useState(null);

  const run = async () => {
    setState("running");
    try {
      const r = await engine.confirm(proposal.id);
      setOut(r);
      setState(r.result && r.result.ok === false ? "failed" : "done");
      onDone && onDone(r);
    } catch (e) {
      setOut({ error: e.message });
      setState("failed");
    }
  };
  const cancel = async () => {
    await engine.reject(proposal.id).catch(() => {});
    setState("cancelled");
  };

  const result = out && out.result;
  const border = { failed: "border-danger/40", done: "border-primary/40", cancelled: "border-primary/10" }[state] || "border-accent/40";
  return (
    <div className={`rounded-xl border ${border} bg-darkSecondary/70 p-4 text-xs`}>
      <div className="mb-1 flex items-center gap-2">
        <span className="rounded bg-accent/15 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-accent">
          {state === "pending" ? "needs your confirmation" : state}
        </span>
        <span className="font-mono text-[10px] text-muted">{proposal.op}</span>
      </div>
      <div className="my-2 whitespace-pre-wrap break-all rounded bg-dark px-3 py-2 font-mono text-[12px] text-light">{proposal.preview}</div>
      <details className="mb-2 text-[11px] text-muted">
        <summary className="cursor-pointer">exact arguments</summary>
        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap break-all">{JSON.stringify(proposal.args, null, 2)}</pre>
      </details>
      {state === "pending" && (
        <div className="flex gap-2">
          <button onClick={run} className="rounded-lg bg-accent px-4 py-1.5 font-semibold text-dark hover:opacity-90">Confirm & run</button>
          <button onClick={cancel} className="rounded-lg bg-surface px-4 py-1.5 text-muted hover:text-light">Cancel</button>
        </div>
      )}
      {state === "running" && <div className="animate-pulse text-muted">running on your machine…</div>}
      {out && out.error && <div className="text-danger">{out.error}</div>}
      {result && (
        <div className="mt-2 space-y-2">
          {result.web_url && (
            <a href={result.web_url} target="_blank" rel="noreferrer" className="inline-block rounded bg-primary px-3 py-1.5 font-semibold text-dark">
              Open {result.name || "codespace"} ↗
            </a>
          )}
          {(result.stdout || result.stderr) && (
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-dark p-2 font-mono text-[11px] text-light/80">
              {[result.command, result.stdout, result.stderr].filter(Boolean).join("\n")}
            </pre>
          )}
          {!result.web_url && !result.stdout && !result.stderr && (
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-dark p-2 font-mono text-[11px] text-light/80">
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
          {(out.charts || []).map((c, i) => <Chart key={i} spec={c} />)}
        </div>
      )}
    </div>
  );
}
