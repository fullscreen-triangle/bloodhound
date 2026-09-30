import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { useCallback, useEffect, useRef, useState } from "react";
import Chart from "@/components/tracker/Chart";
import EngineGate, { useEngine } from "@/components/tracker/EngineGate";
import ProposalCard from "@/components/tracker/ProposalCard";
import { ago, engine, markSeen, seenRepos } from "@/lib/tracker/engine";

/**
 * The federated tracker as a conversation.
 *
 * Ask about your repos and a local Ollama model answers with the tracker's own
 * operations as tools — charts included. Ask it to do something (push a branch to
 * one forge, hide files from one origin, open a Codespace) and it comes back as a
 * proposal you confirm; nothing that changes a repo runs on the model's say-so.
 *
 * Left: repos by last commit. Right: repos you looked at recently (this browser).
 */

const STORE = "tracker.chat";

function load() {
  try {
    return JSON.parse(window.localStorage.getItem(STORE) || "[]");
  } catch {
    return [];
  }
}
function save(messages) {
  try {
    window.localStorage.setItem(STORE, JSON.stringify(messages.slice(-40)));
  } catch {}
}

/** Paragraphs, `code` and **bold** — as React elements, never as HTML. */
function Prose({ text }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed text-light/90">
      {text.split(/\n{2,}/).map((para, i) => (
        <p key={i} className="whitespace-pre-wrap">
          {para.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((part, j) =>
            part.startsWith("`") && part.endsWith("`") ? (
              <code key={j} className="rounded bg-dark px-1 py-0.5 font-mono text-[12px] text-accent">{part.slice(1, -1)}</code>
            ) : part.startsWith("**") && part.endsWith("**") ? (
              <strong key={j} className="text-light">{part.slice(2, -2)}</strong>
            ) : (
              part
            )
          )}
        </p>
      ))}
    </div>
  );
}

function Steps({ steps }) {
  if (!steps || !steps.length) return null;
  return (
    <details className="text-[11px] text-muted">
      <summary className="cursor-pointer select-none">{steps.length} tool call{steps.length > 1 ? "s" : ""}</summary>
      <div className="mt-1 space-y-1">
        {steps.map((s, i) => (
          <div key={i} className="font-mono">
            <span className={s.ok ? "text-primary" : "text-danger"}>{s.ok ? "✓" : "✗"}</span>{" "}
            <span className="text-light/80">{s.op}</span> <span className="break-all">{JSON.stringify(s.args)}</span>
            {s.kind === "proposal" && <span className="text-accent"> → proposal</span>}
            {s.error && <div className="text-danger">{s.error}</div>}
          </div>
        ))}
      </div>
    </details>
  );
}

function Message({ m, onSelectRepo }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-br-sm bg-primary/15 px-4 py-2.5 text-sm text-light">{m.content}</div>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-widest text-primary">
        tracker {m.model && <span className="text-muted normal-case tracking-normal">· {m.model}</span>}
      </div>
      {m.error ? <div className="text-sm text-danger">{m.error}</div> : <Prose text={m.content || "(no answer)"} />}
      <Steps steps={m.steps} />
      {(m.charts || []).map((c, i) => <Chart key={i} spec={c} onSelect={onSelectRepo} />)}
      {(m.proposals || []).map((p) => <ProposalCard key={p.id} proposal={p} />)}
    </div>
  );
}

function RepoRow({ name, sub, meta, active, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`block w-full rounded-lg px-3 py-2 text-left transition-colors ${active ? "bg-primary/15" : "hover:bg-surface"}`}
    >
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate text-[13px] font-medium text-light">{name}</span>
        <span className="shrink-0 text-[10px] text-muted">{meta}</span>
      </div>
      {sub && <div className="truncate text-[11px] text-muted">{sub}</div>}
    </button>
  );
}

function suggestions(focus) {
  const r = focus || "bloodhound";
  return [
    "Which repos did I commit to most recently?",
    "Which repos are about entropy?",
    `What is the state of ${r}? Any uncommitted changes?`,
    `Show the last commits of ${r} and who made them`,
    `Push the main branch of ${r} to my github account`,
    `Open a GitHub codespace for ${r}`,
    `In ${r}, keep drafts/ hidden from the university origin`,
    "Are all my forge tokens still valid?",
  ];
}

function Chat({ status }) {
  const router = useRouter();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recent, setRecent] = useState([]);
  const [seen, setSeen] = useState([]);
  const [focus, setFocus] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [railErr, setRailErr] = useState(null);
  const bottom = useRef(null);
  const abort = useRef(null);

  useEffect(() => {
    setMessages(load());
    setSeen(seenRepos());
    engine.recent(30).then((r) => setRecent(r.repos)).catch((e) => setRailErr(e.message));
    engine.models().then((r) => {
      setModels(r.models || []);
      setModel((r.models || []).find((m) => m.startsWith(r.default)) || (r.models || [])[0] || "");
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (router.query.repo) choose(String(router.query.repo));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [router.query.repo]);

  useEffect(() => {
    bottom.current && bottom.current.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000);
    return () => clearInterval(id);
  }, [busy]);

  const choose = useCallback((name) => {
    setFocus(name);
    setSeen(markSeen(name));
  }, []);

  const send = async (text) => {
    const content = (text ?? input).trim();
    if (!content || busy) return;
    const next = [...messages, { role: "user", content }];
    setMessages(next);
    save(next);
    setInput("");
    setBusy(true);
    setElapsed(0);
    abort.current = new AbortController();
    try {
      const history = next.map((m) => ({ role: m.role, content: m.content || "" })).slice(-16);
      const r = await engine.chat({ messages: history, model: model || undefined, focus: focus || undefined }, abort.current.signal);
      const done = [...next, { role: "assistant", content: r.reply, model: r.model, steps: r.steps, charts: r.charts, proposals: r.proposals }];
      setMessages(done);
      save(done);
    } catch (e) {
      const done = [...next, { role: "assistant", error: e.message }];
      setMessages(done);
      save(done);
    } finally {
      setBusy(false);
    }
  };

  const clear = () => {
    setMessages([]);
    save([]);
  };

  return (
    <div className="flex h-[calc(100vh-96px)] gap-4 px-6 pb-6 lg:px-3">
      {/* left: last updated */}
      <aside className="flex w-64 shrink-0 flex-col rounded-2xl border border-primary/10 bg-darkSecondary/50 xl:w-56 lg:hidden">
        <div className="border-b border-primary/10 px-4 py-3 text-[10px] font-mono uppercase tracking-widest text-primary">Last updated</div>
        <div className="flex-1 space-y-0.5 overflow-y-auto p-2">
          {railErr && <div className="p-2 text-[11px] text-danger">{railErr}</div>}
          {recent.map((r) => (
            <RepoRow
              key={r.key}
              name={r.name}
              sub={r.last_subject}
              meta={ago(r.last_commit)}
              active={focus === r.name}
              onClick={() => choose(r.name)}
            />
          ))}
        </div>
      </aside>

      {/* centre: the conversation */}
      <section className="flex min-w-0 flex-1 flex-col rounded-2xl border border-primary/10 bg-darkSecondary/30">
        <header className="flex flex-wrap items-center gap-3 border-b border-primary/10 px-5 py-3">
          <div className="text-sm font-semibold text-light">Federated tracker</div>
          {focus ? (
            <span className="flex items-center gap-1 rounded-full bg-primary/15 px-3 py-1 text-[11px] text-primary">
              looking at <strong>{focus}</strong>
              <Link href={`/code?repo=${encodeURIComponent((recent.find((r) => r.name === focus) || {}).key || focus)}`} className="ml-1 underline">code</Link>
              <button onClick={() => setFocus(null)} className="ml-1 text-muted hover:text-light" aria-label="clear focus">×</button>
            </span>
          ) : (
            <span className="text-[11px] text-muted">pick a repo on either side to focus the conversation</span>
          )}
          <div className="ml-auto flex items-center gap-2">
            {models.length > 0 && (
              <select value={model} onChange={(e) => setModel(e.target.value)} className="rounded-lg border border-primary/15 bg-surface px-2 py-1 text-[11px] text-light">
                {models.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            )}
            <button onClick={clear} className="text-[11px] text-muted hover:text-light">clear</button>
            <span className="text-[10px] font-mono text-muted">engine v{status.version}</span>
          </div>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto px-6 py-5">
          {messages.length === 0 && (
            <div className="mx-auto max-w-2xl pt-10 text-center">
              <div className="mb-2 text-lg font-semibold text-light">Ask about your repos, or tell the tracker what to do</div>
              <p className="mb-6 text-sm text-muted">
                Answers come from your local Ollama model using the tracker&apos;s own operations. Anything that would change a repo
                or a forge is shown to you first and runs only when you confirm it.
              </p>
              <div className="grid grid-cols-2 gap-2 md:grid-cols-1">
                {suggestions(focus).map((s) => (
                  <button key={s} onClick={() => send(s)} className="rounded-xl border border-primary/15 bg-surface/60 px-4 py-3 text-left text-[12px] text-light/85 hover:border-primary/40">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => <Message key={i} m={m} onSelectRepo={(r) => choose(String(r).split("/").pop())} />)}
          {busy && (
            <div className="flex items-center gap-3 text-sm text-muted">
              <span className="h-2 w-2 animate-pulse rounded-full bg-primary" />
              thinking with {model || "the local model"}… {elapsed}s
              <button onClick={() => abort.current && abort.current.abort()} className="text-[11px] underline hover:text-light">stop</button>
            </div>
          )}
          <div ref={bottom} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="border-t border-primary/10 p-4"
        >
          <div className="flex items-end gap-3 rounded-xl border border-primary/20 bg-surface px-4 py-3 focus-within:border-primary/50">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={Math.min(6, Math.max(1, input.split("\n").length))}
              placeholder={focus ? `Ask about ${focus}, or tell the tracker what to do…` : "Ask about your repos, or tell the tracker what to do…"}
              className="max-h-40 flex-1 resize-none bg-transparent text-sm text-light placeholder:text-muted focus:outline-none"
            />
            <button type="submit" disabled={busy || !input.trim()} className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-dark disabled:opacity-40">
              Send
            </button>
          </div>
          <div className="mt-1.5 text-[10px] text-muted">Enter to send · Shift+Enter for a new line · actions always wait for your confirmation</div>
        </form>
      </section>

      {/* right: recently inspected */}
      <aside className="flex w-64 shrink-0 flex-col rounded-2xl border border-primary/10 bg-darkSecondary/50 xl:w-56 lg:hidden">
        <div className="border-b border-primary/10 px-4 py-3 text-[10px] font-mono uppercase tracking-widest text-accent">Recently inspected</div>
        <div className="flex-1 space-y-0.5 overflow-y-auto p-2">
          {seen.length === 0 && <div className="p-2 text-[11px] text-muted">Repos you focus here or open in Repo Lens show up here.</div>}
          {seen.map((r) => (
            <RepoRow key={r.name} name={r.name} meta={ago(r.at)} active={focus === r.name} onClick={() => choose(r.name)} />
          ))}
        </div>
      </aside>
    </div>
  );
}

export default function TrackerPage() {
  const [status, retry] = useEngine();
  return (
    <>
      <Head>
        <title>Tracker | Bloodhound</title>
      </Head>
      {status.state === "ready" ? (
        <Chat status={status} />
      ) : (
        <div className="mx-auto max-w-2xl px-6 py-16">
          <EngineGate status={status} retry={retry} />
        </div>
      )}
    </>
  );
}
