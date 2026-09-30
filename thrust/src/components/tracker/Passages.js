import Link from "next/link";

/**
 * A spraypaint search result: the coverage verdict first, then each passage's
 * evidence lines with the words it matched. The verdict says whether the repo
 * contains the query's words at all — never whether a passage answers anything.
 */

const TONE = {
  covered: "border-primary/50 bg-primary/10 text-primary",
  partial: "border-accent/50 bg-accent/10 text-accent",
  declined: "border-danger/50 bg-danger/10 text-danger",
};

export function Verdict({ verdict, reason }) {
  return (
    <div className="flex flex-wrap items-baseline gap-2">
      <span className={`rounded-full border px-2.5 py-0.5 font-mono text-[11px] uppercase tracking-wider ${TONE[verdict] || TONE.partial}`}>
        {verdict}
      </span>
      <span className="text-[12px] text-light/80">{reason}</span>
    </div>
  );
}

/** `repo` is a repo name or key (for links into /code); `onOpen(path, line)` replaces the link. */
export default function Passages({ result, repo, onOpen }) {
  const items = result.items || [];
  return (
    <div className="space-y-3">
      <Verdict verdict={result.verdict} reason={result.reason} />
      {result.withheld > 0 && (
        <div className="text-[11px] text-muted">
          {result.withheld} passage{result.withheld === 1 ? "" : "s"} from secret-bearing files (.env, *.local.json, keys) not shown.
        </div>
      )}
      {items.map((p, i) => {
        const where = `${p.path}:${p.start}-${p.end}`;
        const lines = String(p.snippet || "").split("\n");
        const head = onOpen ? (
          <button onClick={() => onOpen(p.path, p.start)} className="font-mono text-[12px] text-primary hover:underline">{where}</button>
        ) : repo ? (
          <Link href={`/code?repo=${encodeURIComponent(repo)}&path=${encodeURIComponent(p.path)}`} className="font-mono text-[12px] text-primary hover:underline">
            {where}
          </Link>
        ) : (
          <span className="font-mono text-[12px] text-primary">{where}</span>
        );
        return (
          <div key={i} className="overflow-hidden rounded-lg border border-primary/10">
            <div className="flex flex-wrap items-center gap-2 bg-darkSecondary px-3 py-1.5">
              {head}
              {p.scene && <span className="text-[10px] text-muted">[{p.scene}]</span>}
              <span className="ml-auto flex flex-wrap gap-1">
                {(p.matched || []).map((t) => (
                  <span key={t} className="rounded bg-accent/10 px-1.5 font-mono text-[10px] text-accent">{t}</span>
                ))}
              </span>
            </div>
            <pre className="overflow-x-auto bg-dark/80 py-1.5 font-mono text-[12px] leading-5">
              {lines.map((l, j) => (
                <div key={j} className="flex">
                  <span className="w-12 shrink-0 select-none pr-3 text-right text-muted/60">{Number(p.start) + j}</span>
                  <span className="whitespace-pre text-light/85">{l || " "}</span>
                </div>
              ))}
            </pre>
          </div>
        );
      })}
      {items.length === 0 && result.verdict !== "declined" && <div className="text-[12px] text-muted">no passages</div>}
    </div>
  );
}
