import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { useCallback, useEffect, useMemo, useState } from "react";
import EngineGate, { useEngine } from "@/components/tracker/EngineGate";
import { ago, engine, markSeen } from "@/lib/tracker/engine";

/**
 * Any local repo, opened as on a forge: browse files at any branch, read the history,
 * edit and create files, review the changes, commit them (on a new branch if you
 * like) and push to whichever origin you choose. Everything runs through the local
 * tracker engine against the repo on disk.
 *
 * URL: /code?repo=<key>&tab=code|changes|commits&path=<file>&rev=<branch>
 */

const STATE = {
  M: ["M", "text-accent", "modified"],
  A: ["A", "text-primary", "added"],
  D: ["D", "text-danger", "deleted"],
  R: ["R", "text-accent", "renamed"],
  "??": ["U", "text-primary", "untracked"],
};
const stateOf = (s) => (s ? STATE[s] || STATE[s[0]] || [s, "text-muted", s] : null);

// ── small pieces ─────────────────────────────────────────────────────────────

function Btn({ children, onClick, tone = "surface", disabled, title }) {
  const cls = {
    primary: "bg-primary text-dark font-semibold",
    accent: "bg-accent text-dark font-semibold",
    danger: "bg-danger/80 text-light",
    surface: "bg-surface text-light hover:bg-darkTertiary",
  }[tone];
  return (
    <button title={title} disabled={disabled} onClick={onClick} className={`rounded-lg px-3 py-1.5 text-[12px] transition-opacity disabled:opacity-40 ${cls}`}>
      {children}
    </button>
  );
}

function ErrorLine({ error }) {
  return error ? <div className="rounded-lg border border-danger/30 bg-danger/5 px-3 py-2 text-[12px] text-danger">{error}</div> : null;
}

/** Nested folders from flat paths. */
function buildTree(files) {
  const root = { dirs: {}, files: [] };
  for (const f of files) {
    const parts = f.path.split("/");
    let node = root;
    for (const p of parts.slice(0, -1)) node = node.dirs[p] || (node.dirs[p] = { dirs: {}, files: [] });
    node.files.push({ name: parts[parts.length - 1], ...f });
  }
  return root;
}

function dirChanged(node) {
  return node.files.some((f) => f.state) || Object.values(node.dirs).some(dirChanged);
}

function TreeNode({ node, prefix, open, toggle, selected, onOpen }) {
  const dirs = Object.keys(node.dirs).sort();
  const files = [...node.files].sort((a, b) => a.name.localeCompare(b.name));
  return (
    <div>
      {dirs.map((d) => {
        const path = prefix + d;
        const isOpen = open.has(path);
        return (
          <div key={path}>
            <button onClick={() => toggle(path)} className="flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-[12px] text-light/90 hover:bg-surface">
              <span className="w-3 text-muted">{isOpen ? "▾" : "▸"}</span>
              <span className="text-accent/80">▰</span>
              <span className="truncate">{d}</span>
              {dirChanged(node.dirs[d]) && <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />}
            </button>
            {isOpen && (
              <div className="ml-3 border-l border-primary/10 pl-1">
                <TreeNode node={node.dirs[d]} prefix={path + "/"} open={open} toggle={toggle} selected={selected} onOpen={onOpen} />
              </div>
            )}
          </div>
        );
      })}
      {files.map((f) => {
        const st = stateOf(f.state);
        return (
          <button
            key={f.path}
            onClick={() => onOpen(f.path)}
            className={`flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-[12px] ${selected === f.path ? "bg-primary/15 text-light" : "text-light/80 hover:bg-surface"}`}
          >
            <span className="w-3" />
            <span className="text-muted">▫</span>
            <span className="truncate">{f.name}</span>
            {st && <span className={`ml-auto shrink-0 font-mono text-[10px] ${st[1]}`} title={st[2]}>{st[0]}</span>}
          </button>
        );
      })}
    </div>
  );
}

function CodeView({ text }) {
  const lines = text.split(/\r?\n/);
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  return (
    <div className="overflow-auto rounded-b-xl bg-dark/80">
      <table className="w-full border-collapse font-mono text-[12px] leading-5">
        <tbody>
          {lines.map((l, i) => (
            <tr key={i} className="hover:bg-surface/40">
              <td className="w-12 select-none pr-3 text-right align-top text-muted/60">{i + 1}</td>
              <td className="whitespace-pre pr-4 text-light/90">{l || " "}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** A unified diff, one card per file. */
function DiffView({ diff, selectable, chosen, setChosen }) {
  const files = useMemo(() => {
    const out = [];
    let cur = null;
    for (const line of (diff || "").split("\n")) {
      const m = line.match(/^diff --git a\/(.+?) b\/(.+)$/);
      if (m) {
        cur = { path: m[2], lines: [] };
        out.push(cur);
      } else if (cur) {
        cur.lines.push(line);
      }
    }
    return out;
  }, [diff]);
  if (!files.length) return <div className="text-[12px] text-muted">no line changes</div>;
  return (
    <div className="space-y-4">
      {files.map((f) => (
        <div key={f.path} className="overflow-hidden rounded-xl border border-primary/10">
          <div className="flex items-center gap-2 bg-darkSecondary px-3 py-2 font-mono text-[12px] text-light">
            {selectable && (
              <input
                type="checkbox"
                checked={chosen.has(f.path)}
                onChange={(e) => {
                  const n = new Set(chosen);
                  e.target.checked ? n.add(f.path) : n.delete(f.path);
                  setChosen(n);
                }}
              />
            )}
            {f.path}
          </div>
          <pre className="max-h-[480px] overflow-auto bg-dark/80 p-0 font-mono text-[12px] leading-5">
            {f.lines
              .filter((l) => !/^(index |--- |\+\+\+ |new file mode|deleted file mode|similarity|rename )/.test(l))
              .map((l, i) => (
                <div
                  key={i}
                  className={`whitespace-pre px-3 ${
                    l.startsWith("+") ? "bg-primary/10 text-primary" : l.startsWith("-") ? "bg-danger/10 text-danger" : l.startsWith("@@") ? "text-[#6D8FD8]" : "text-light/70"
                  }`}
                >
                  {l || " "}
                </div>
              ))}
          </pre>
        </div>
      ))}
    </div>
  );
}

// ── the three tabs ───────────────────────────────────────────────────────────

function CodeTab({ repo, rev, path, setPath, tree, reload }) {
  const [open, setOpen] = useState(new Set());
  const [file, setFile] = useState(null);
  const [draft, setDraft] = useState(null); // null = not editing
  const [newPath, setNewPath] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const readme = useMemo(() => tree && tree.files.find((f) => /^readme(\.md|\.rst|\.txt)?$/i.test(f.path)), [tree]);
  const shown = path || (readme && readme.path);

  // Open every folder on the way to the selected file.
  useEffect(() => {
    if (!path) return;
    const parts = path.split("/");
    setOpen((o) => {
      const n = new Set(o);
      for (let i = 1; i < parts.length; i++) n.add(parts.slice(0, i).join("/"));
      return n;
    });
  }, [path]);

  useEffect(() => {
    setFile(null);
    setDraft(null);
    setError(null);
    if (!shown) return;
    engine.call("repo_file", { repo: repo.path, path: shown, ...(rev ? { rev } : {}) })
      .then((r) => setFile(r.result))
      .catch((e) => setError(e.message));
  }, [repo.path, rev, shown]);

  const toggle = (p) => setOpen((o) => {
    const n = new Set(o);
    n.has(p) ? n.delete(p) : n.add(p);
    return n;
  });

  const save = async (target, content) => {
    setBusy(true);
    setError(null);
    try {
      await engine.act("repo_write", { repo: repo.path, path: target, content });
      setDraft(null);
      setNewPath(null);
      await reload();
      setPath(target);
      const r = await engine.call("repo_file", { repo: repo.path, path: target });
      setFile(r.result);
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  };
  const remove = async () => {
    if (!window.confirm(`Delete ${shown} from the working tree?`)) return;
    setBusy(true);
    try {
      await engine.act("repo_write", { repo: repo.path, path: shown, delete: true });
      setPath(null);
      await reload();
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  };

  const editable = !rev && file && file.kind === "text";
  const crumbs = shown ? shown.split("/") : [];

  return (
    <div className="grid grid-cols-[280px_1fr] gap-4 lg:grid-cols-1">
      <aside className="max-h-[75vh] overflow-y-auto rounded-xl border border-primary/10 bg-darkSecondary/50 p-2">
        <div className="mb-2 flex items-center justify-between px-1.5">
          <span className="text-[10px] font-mono uppercase tracking-widest text-muted">{tree ? `${tree.files.length} files` : "files"}</span>
          {!rev && <button onClick={() => setNewPath("")} className="text-[11px] text-primary hover:underline">+ new file</button>}
        </div>
        {tree ? <TreeNode node={buildTree(tree.files)} prefix="" open={open} toggle={toggle} selected={shown} onOpen={setPath} /> : <div className="p-2 text-[12px] text-muted">loading…</div>}
        {tree && tree.deleted && tree.deleted.length > 0 && (
          <div className="mt-3 border-t border-primary/10 px-1.5 pt-2 text-[11px] text-danger">
            deleted: {tree.deleted.join(", ")}
          </div>
        )}
      </aside>

      <section className="min-w-0 space-y-3">
        <ErrorLine error={error} />
        {newPath !== null ? (
          <div className="rounded-xl border border-primary/15 bg-darkSecondary/60">
            <div className="flex items-center gap-2 border-b border-primary/10 px-3 py-2">
              <span className="text-[12px] text-muted">new file</span>
              <input
                autoFocus
                value={newPath}
                onChange={(e) => setNewPath(e.target.value)}
                placeholder="path/inside/repo.md"
                className="flex-1 rounded bg-surface px-2 py-1 font-mono text-[12px] text-light focus:outline-none"
              />
              <Btn tone="primary" disabled={busy || !newPath.trim()} onClick={() => save(newPath.trim(), draft || "")}>Create</Btn>
              <Btn onClick={() => { setNewPath(null); setDraft(null); }}>Cancel</Btn>
            </div>
            <Editor value={draft || ""} onChange={setDraft} />
          </div>
        ) : shown ? (
          <div className="rounded-xl border border-primary/15 bg-darkSecondary/60">
            <div className="flex flex-wrap items-center gap-2 border-b border-primary/10 px-3 py-2">
              <div className="flex min-w-0 flex-wrap items-center font-mono text-[12px]">
                <button onClick={() => setPath(null)} className="text-primary hover:underline">{repo.name}</button>
                {crumbs.map((c, i) => (
                  <span key={i} className="text-light/80"><span className="mx-1 text-muted">/</span>{c}</span>
                ))}
              </div>
              {file && <span className="text-[11px] text-muted">{(file.size / 1024).toFixed(1)} KB{rev ? ` · at ${rev}` : ""}</span>}
              <div className="ml-auto flex gap-2">
                {editable && draft === null && <Btn onClick={() => setDraft(file.text)}>Edit</Btn>}
                {editable && draft !== null && (
                  <>
                    <Btn tone="primary" disabled={busy || draft === file.text} onClick={() => save(shown, draft)}>{busy ? "Saving…" : "Save"}</Btn>
                    <Btn onClick={() => setDraft(null)}>Cancel</Btn>
                  </>
                )}
                {!rev && draft === null && file && <Btn tone="danger" disabled={busy} onClick={remove}>Delete</Btn>}
                <Link href={`/tracker?repo=${encodeURIComponent(repo.name)}`} className="rounded-lg bg-surface px-3 py-1.5 text-[12px] text-light hover:bg-darkTertiary">Ask</Link>
              </div>
            </div>
            {!file && !error && <div className="p-4 text-[12px] text-muted">loading…</div>}
            {file && draft !== null && <Editor value={draft} onChange={setDraft} />}
            {file && draft === null && file.kind === "text" && <CodeView text={file.text} />}
            {file && file.kind === "image" && (
              // eslint-disable-next-line @next/next/no-img-element
              <img alt={shown} src={`data:${file.mime};base64,${file.base64}`} className="mx-auto max-h-[70vh] p-4" />
            )}
            {file && (file.kind === "binary" || file.kind === "too_large") && (
              <div className="p-6 text-[12px] text-muted">{file.kind === "binary" ? "binary file" : "file too large to show"} ({(file.size / 1024).toFixed(0)} KB)</div>
            )}
          </div>
        ) : (
          <div className="rounded-xl border border-primary/10 p-6 text-[12px] text-muted">Pick a file on the left.</div>
        )}
      </section>
    </div>
  );
}

/** A plain editor: monospace, Tab inserts spaces, Ctrl/Cmd+S is left to the Save button. */
function Editor({ value, onChange }) {
  return (
    <textarea
      value={value}
      spellCheck={false}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Tab") {
          e.preventDefault();
          const t = e.target;
          const s = t.selectionStart;
          const next = value.slice(0, s) + "    " + value.slice(t.selectionEnd);
          onChange(next);
          requestAnimationFrame(() => { t.selectionStart = t.selectionEnd = s + 4; });
        }
      }}
      className="block h-[65vh] w-full resize-y rounded-b-xl bg-dark/80 p-4 font-mono text-[12px] leading-5 text-light focus:outline-none"
    />
  );
}

function ChangesTab({ repo, status, tree, reload }) {
  const [diff, setDiff] = useState(null);
  const [chosen, setChosen] = useState(new Set());
  const [message, setMessage] = useState("");
  const [branch, setBranch] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [committed, setCommitted] = useState(null);
  const [pushes, setPushes] = useState({});

  const changed = (status && status.changes) || [];
  const load = useCallback(async () => {
    setError(null);
    try {
      const r = await engine.call("repo_diff", { repo: repo.path });
      setDiff(r.result);
    } catch (e) {
      setError(e.message);
    }
  }, [repo.path]);
  useEffect(() => { load(); }, [load, status]);
  useEffect(() => { setChosen(new Set(changed.map((c) => c.path))); }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

  const commit = async () => {
    setBusy(true);
    setError(null);
    try {
      const all = chosen.size === changed.length;
      const r = await engine.act("repo_commit", {
        repo: repo.path,
        message,
        ...(all ? {} : { paths: Array.from(chosen) }),
        ...(branch.trim() ? { branch: branch.trim() } : {}),
      });
      setCommitted(r);
      setMessage("");
      setBranch("");
      await reload();
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  };

  const hasSync = tree && tree.files.some((f) => f.path === ".sync.toml");
  const push = async (key, op, args) => {
    setPushes((p) => ({ ...p, [key]: { state: "running" } }));
    try {
      const r = await engine.act(op, args);
      const ok = r.ok !== false;
      setPushes((p) => ({ ...p, [key]: { state: ok ? "done" : "failed", out: [r.stdout, r.stderr].filter(Boolean).join("\n") || JSON.stringify(r) } }));
    } catch (e) {
      setPushes((p) => ({ ...p, [key]: { state: "failed", out: e.message } }));
    }
  };
  const currentBranch = (committed && committed.branch) || (status && status.branch);
  const remotes = (repo.remotes || []);

  return (
    <div className="grid grid-cols-[1fr_340px] gap-5 lg:grid-cols-1">
      <div className="min-w-0 space-y-3">
        <ErrorLine error={error} />
        {changed.length === 0 && <div className="rounded-xl border border-primary/10 p-6 text-[12px] text-muted">No uncommitted changes. Edit a file in the Code tab.</div>}
        {changed.length > 0 && (
          <div className="rounded-xl border border-primary/10 bg-darkSecondary/50 p-3 text-[12px]">
            {changed.map((c) => {
              const st = stateOf(c.state);
              return (
                <label key={c.path} className="flex items-center gap-2 py-0.5">
                  <input
                    type="checkbox"
                    checked={chosen.has(c.path)}
                    onChange={(e) => {
                      const n = new Set(chosen);
                      e.target.checked ? n.add(c.path) : n.delete(c.path);
                      setChosen(n);
                    }}
                  />
                  <span className={`w-4 font-mono ${st ? st[1] : ""}`}>{st ? st[0] : c.state}</span>
                  <span className="font-mono text-light/85">{c.path}</span>
                </label>
              );
            })}
          </div>
        )}
        {diff && <DiffView diff={diff.diff} />}
      </div>

      <aside className="space-y-4">
        <div className="rounded-xl border border-primary/15 bg-darkSecondary/60 p-4">
          <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-primary">Commit</div>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            rows={4}
            placeholder="What changed, and why"
            className="mb-2 w-full resize-y rounded-lg border border-primary/15 bg-surface p-2 text-[12px] text-light focus:outline-none"
          />
          <input
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            placeholder={`branch (default: ${status ? status.branch : "current"})`}
            className="mb-3 w-full rounded-lg border border-primary/15 bg-surface px-2 py-1.5 font-mono text-[12px] text-light focus:outline-none"
          />
          <Btn tone="primary" disabled={busy || !message.trim() || chosen.size === 0} onClick={commit}>
            {busy ? "Committing…" : `Commit ${chosen.size} file${chosen.size === 1 ? "" : "s"}`}
          </Btn>
          {committed && <div className="mt-2 text-[11px] text-primary">committed {committed.sha.slice(0, 8)} on {committed.branch}</div>}
        </div>

        <div className="rounded-xl border border-primary/15 bg-darkSecondary/60 p-4">
          <div className="mb-2 text-[10px] font-mono uppercase tracking-widest text-accent">Push {currentBranch}</div>
          {hasSync ? (
            <>
              <p className="mb-2 text-[11px] text-muted">This repo has a .sync.toml: each origin gets only the paths it may see.</p>
              <Btn tone="accent" onClick={() => push("sync", "sync_run", { repo: repo.path })}>Sync all origins</Btn>
              <PushState s={pushes.sync} />
            </>
          ) : remotes.length === 0 ? (
            <p className="text-[11px] text-muted">This repo has no remotes.</p>
          ) : (
            remotes.map((r) => (
              <div key={r.name} className="mb-2">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[12px] text-light">{r.name} <span className="text-muted">{r.host}{r.account ? ` · ${r.account}` : ""}</span></span>
                  <Btn
                    onClick={() =>
                      r.account
                        ? push(r.name, "push_branch", { repo: repo.path, branch: currentBranch, account: r.account })
                        : push(r.name, "git_exec", { repo: repo.path, args: ["push", r.name, currentBranch] })
                    }
                  >
                    Push
                  </Btn>
                </div>
                <PushState s={pushes[r.name]} />
              </div>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}

function PushState({ s }) {
  if (!s) return null;
  return (
    <div className={`mt-1 text-[11px] ${s.state === "failed" ? "text-danger" : s.state === "done" ? "text-primary" : "text-muted"}`}>
      {s.state === "running" ? "pushing…" : s.state}
      {s.out && <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap rounded bg-dark p-2 font-mono text-[10px] text-light/70">{s.out}</pre>}
    </div>
  );
}

function CommitsTab({ repo, rev }) {
  const [log, setLog] = useState(null);
  const [open, setOpen] = useState(null);
  const [diff, setDiff] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    setLog(null);
    engine.call("repo_log", { repo: repo.path, limit: 100, ...(rev ? { rev } : {}) })
      .then((r) => setLog(r.result.commits))
      .catch((e) => setError(e.message));
  }, [repo.path, rev]);
  const show = async (sha) => {
    setOpen(sha);
    setDiff(null);
    try {
      setDiff((await engine.call("repo_diff", { repo: repo.path, commit: sha })).result.diff);
    } catch (e) {
      setError(e.message);
    }
  };
  return (
    <div className="grid grid-cols-[minmax(320px,420px)_1fr] gap-5 lg:grid-cols-1">
      <div className="max-h-[75vh] overflow-y-auto rounded-xl border border-primary/10 bg-darkSecondary/50">
        <ErrorLine error={error} />
        {!log && <div className="p-4 text-[12px] text-muted">loading…</div>}
        {log && log.map((c) => (
          <button key={c.sha} onClick={() => show(c.sha)} className={`block w-full border-b border-primary/5 px-4 py-2.5 text-left ${open === c.sha ? "bg-primary/10" : "hover:bg-surface"}`}>
            <div className="truncate text-[13px] text-light">{c.subject}</div>
            <div className="mt-0.5 flex gap-2 text-[11px] text-muted">
              <span className="font-mono text-accent/80">{c.short}</span>
              <span>{c.author}</span>
              <span>{ago(c.date)}</span>
              {c.refs && <span className="truncate text-primary/80">{c.refs}</span>}
            </div>
          </button>
        ))}
      </div>
      <div className="min-w-0">
        {!open && <div className="rounded-xl border border-primary/10 p-6 text-[12px] text-muted">Pick a commit to see what it changed.</div>}
        {open && !diff && <div className="text-[12px] text-muted">loading…</div>}
        {diff && (
          <>
            <pre className="mb-4 whitespace-pre-wrap rounded-xl bg-darkSecondary/60 p-4 font-mono text-[12px] text-light/85">
              {diff.split("\ndiff --git")[0]}
            </pre>
            <DiffView diff={diff} />
          </>
        )}
      </div>
    </div>
  );
}

// ── the page ─────────────────────────────────────────────────────────────────

function RepoPicker({ repos, current, onPick }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const list = repos.filter((r) => (r.key + " " + r.name).toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="relative">
      <button onClick={() => setOpen(!open)} className="flex items-center gap-2 rounded-lg border border-primary/20 bg-surface px-3 py-2 text-left">
        <span className="text-[13px] font-semibold text-light">{current ? current.name : "Choose a repo"}</span>
        {current && <span className="font-mono text-[11px] text-muted">{current.key}</span>}
        <span className="text-muted">▾</span>
      </button>
      {open && (
        <div className="absolute z-30 mt-1 w-96 rounded-xl border border-primary/20 bg-darkSecondary p-2 shadow-glow-lg">
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="filter…" className="mb-2 w-full rounded bg-surface px-2 py-1.5 text-[12px] text-light focus:outline-none" />
          <div className="max-h-80 overflow-y-auto">
            {list.map((r) => (
              <button key={r.key} onClick={() => { onPick(r); setOpen(false); setQ(""); }} className="flex w-full items-baseline justify-between gap-2 rounded px-2 py-1.5 text-left hover:bg-surface">
                <span className="text-[12px] text-light">{r.name}</span>
                <span className="truncate font-mono text-[10px] text-muted">{r.key} · {ago(r.last_commit)}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Browser() {
  const router = useRouter();
  const { repo: repoKey, tab = "code", path = null, rev = null } = router.query;
  const [repos, setRepos] = useState([]);
  const [status, setStatus] = useState(null);
  const [tree, setTree] = useState(null);
  const [error, setError] = useState(null);
  const repo = repos.find((r) => r.key === repoKey);

  const go = useCallback(
    (patch) => {
      const q = { ...router.query, ...patch };
      Object.keys(q).forEach((k) => (q[k] === null || q[k] === undefined || q[k] === "") && delete q[k]);
      router.replace({ pathname: "/code", query: q }, undefined, { shallow: true });
    },
    [router]
  );

  useEffect(() => {
    engine.graph().then((g) => setRepos([...g.repos].sort((a, b) => (b.last_commit || "").localeCompare(a.last_commit || "")))).catch((e) => setError(e.message));
  }, []);

  const reload = useCallback(async () => {
    if (!repo) return;
    setError(null);
    try {
      const [s, t] = await Promise.all([
        engine.call("repo_status", { repo: repo.path }),
        engine.call("repo_tree", { repo: repo.path, ...(rev ? { rev } : {}) }),
      ]);
      setStatus(s.result);
      setTree(t.result);
    } catch (e) {
      setError(e.message);
    }
  }, [repo, rev]);

  useEffect(() => {
    setStatus(null);
    setTree(null);
    if (repo) markSeen(repo.name);
    reload();
  }, [reload, repo]);

  const switchBranch = async (b) => {
    setError(null);
    try {
      const r = await engine.act("git_exec", { repo: repo.path, args: ["switch", b] });
      if (r.ok === false) throw new Error(r.stderr || "git switch failed");
      go({ rev: null, path: null });
      await reload();
    } catch (e) {
      setError(e.message);
    }
  };

  const changes = (status && status.changes && status.changes.length) || 0;
  const onGithub = repo && (repo.remotes || []).some((r) => r.host === "github.com");
  const [codespace, setCodespace] = useState(null);
  const openCodespace = async () => {
    setCodespace({ state: "starting" });
    try {
      const r = await engine.act("codespace_open", { repo: repo.path, ...(rev || (status && status.branch) ? { branch: rev || status.branch } : {}) });
      setCodespace({ state: "ready", url: r.web_url });
      if (r.web_url) window.open(r.web_url, "_blank", "noopener");
    } catch (e) {
      setCodespace({ state: "failed", error: e.message });
    }
  };

  return (
    <div className="mx-auto max-w-[1600px] space-y-4 px-8 pb-16 pt-4 lg:px-4">
      <div className="flex flex-wrap items-center gap-3">
        <RepoPicker repos={repos} current={repo} onPick={(r) => go({ repo: r.key, path: null, rev: null })} />
        {status && (
          <select
            value={rev || ""}
            onChange={(e) => go({ rev: e.target.value || null, path: null })}
            className="rounded-lg border border-primary/20 bg-surface px-2 py-2 font-mono text-[12px] text-light"
          >
            <option value="">{status.branch} (working tree)</option>
            {status.branches.filter((b) => b !== status.branch).map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        )}
        {rev && <Btn tone="accent" onClick={() => switchBranch(rev)} title="git switch — uncommitted changes come along">Switch to {rev}</Btn>}
        {repo && (
          <div className="flex gap-1 rounded-lg bg-surface p-0.5">
            {[["code", "Code"], ["changes", `Changes${changes ? ` · ${changes}` : ""}`], ["commits", "Commits"]].map(([t, label]) => (
              <button key={t} onClick={() => go({ tab: t })} className={`rounded-md px-3 py-1.5 text-[12px] ${tab === t ? "bg-primary text-dark font-semibold" : "text-muted hover:text-light"}`}>
                {label}
              </button>
            ))}
          </div>
        )}
        {repo && (
          <div className="ml-auto flex items-center gap-2">
            {onGithub && <Btn onClick={openCodespace} disabled={codespace && codespace.state === "starting"}>{codespace && codespace.state === "starting" ? "Starting Codespace…" : "Codespace ↗"}</Btn>}
            <Link href={`/tracker?repo=${encodeURIComponent(repo.name)}`} className="rounded-lg bg-surface px-3 py-1.5 text-[12px] text-light hover:bg-darkTertiary">Ask the tracker</Link>
          </div>
        )}
      </div>
      {codespace && codespace.state === "failed" && <ErrorLine error={codespace.error} />}
      {codespace && codespace.state === "ready" && codespace.url && (
        <div className="text-[12px] text-primary">Codespace ready: <a className="underline" href={codespace.url} target="_blank" rel="noreferrer">{codespace.url}</a></div>
      )}
      {repo && (
        <div className="flex flex-wrap gap-x-4 text-[11px] text-muted">
          <span className="font-mono">{repo.path}</span>
          {(repo.remotes || []).map((r) => <span key={r.name}>{r.name}: {r.host}</span>)}
          {rev && <span className="text-accent">viewing {rev} read-only</span>}
        </div>
      )}
      <ErrorLine error={error} />
      {!repo && repos.length > 0 && (
        <div className="grid grid-cols-3 gap-3 lg:grid-cols-2 sm:grid-cols-1">
          {repos.map((r) => (
            <button key={r.key} onClick={() => go({ repo: r.key })} className="rounded-xl border border-primary/10 bg-darkSecondary/50 p-4 text-left hover:border-primary/40">
              <div className="text-[14px] font-semibold text-light">{r.name}</div>
              <div className="font-mono text-[10px] text-muted">{r.key}</div>
              <div className="mt-2 truncate text-[11px] text-light/70">{r.last_subject}</div>
              <div className="mt-1 text-[10px] text-muted">{ago(r.last_commit)} · {r.commits} commits · {Array.from(new Set((r.remotes || []).map((x) => x.host))).join(", ") || "local only"}</div>
            </button>
          ))}
        </div>
      )}
      {repo && tab === "code" && <CodeTab repo={repo} rev={rev} path={path} setPath={(p) => go({ path: p })} tree={tree} reload={reload} />}
      {repo && tab === "changes" && (rev ? <div className="text-[12px] text-muted">Changes are in the working tree; pick “{status && status.branch} (working tree)”.</div> : <ChangesTab repo={repo} status={status} tree={tree} reload={reload} />)}
      {repo && tab === "commits" && <CommitsTab repo={repo} rev={rev} />}
    </div>
  );
}

export default function CodePage() {
  const [status, retry] = useEngine();
  return (
    <>
      <Head>
        <title>Code | Bloodhound</title>
      </Head>
      {status.state === "ready" ? (
        <Browser />
      ) : (
        <div className="mx-auto max-w-2xl px-6 py-16">
          <EngineGate status={status} retry={retry} />
        </div>
      )}
    </>
  );
}
