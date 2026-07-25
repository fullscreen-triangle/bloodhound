import Head from "next/head";
import Link from "next/link";
import dynamic from "next/dynamic";
import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

const Desk = dynamic(() => import("@/components/Desk"), { ssr: false });

const NAV_SECTIONS = [
  {
    label: "Explore",
    links: [
      { href: "/about",        title: "About" },
      { href: "/architecture", title: "Architecture" },
      { href: "/use-cases",    title: "Use Cases" },
      { href: "/docs",         title: "Docs" },
      { href: "/roadmap",      title: "Roadmap" },
      { href: "/collaborate",  title: "Collaborate" },
    ],
  },
  {
    label: "Framework Pillars",
    links: [
      { href: "/phase-space",  title: "Phase Space" },
      { href: "/compilation",  title: "Compilation" },
      { href: "/federated",    title: "Federated" },
      { href: "/pipeline",     title: "Pipeline" },
      { href: "/validation",   title: "Validation" },
    ],
  },
  {
    label: "Tools",
    links: [
      { href: "/repo-lens",    title: "Repo Lens", highlight: true },
      { href: "/cytochrome",   title: "CYP450 Research Engine", highlight: true },
    ],
  },
];

export default function Home() {
  const [navOpen, setNavOpen] = useState(false);

  const close = useCallback(() => setNavOpen(false), []);

  // Escape key dismisses
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") close(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [close]);

  return (
    <>
      <Head>
        <title>Bloodhound</title>
        <meta name="description" content="Bloodhound: automated deep research through purpose-driven trajectory completion in bounded phase space." />
      </Head>

      {/* Full-viewport 3D scene */}
      <div className="fixed inset-0 bg-dark">
        <Desk />
      </div>

      {/* ── Persistent minimal overlay ──────────────────────────────────── */}
      <div className="fixed inset-0 pointer-events-none z-10">

        {/* Wordmark — top left */}
        <span className="absolute top-6 left-10 sm:left-6 text-xs font-mono tracking-[0.25em] uppercase text-primary/60 pointer-events-none select-none">
          Bloodhound
        </span>

        {/* Menu toggle — top right */}
        <button
          onClick={() => setNavOpen((o) => !o)}
          className="absolute top-5 right-10 sm:right-6 pointer-events-auto
            flex flex-col justify-center items-center gap-[5px] w-9 h-9
            text-muted/50 hover:text-primary transition-colors duration-300 group"
          aria-label="Toggle navigation"
        >
          <motion.span
            animate={navOpen ? { rotate: 45, y: 7 } : { rotate: 0, y: 0 }}
            transition={{ duration: 0.25 }}
            className="block w-5 h-px bg-current rounded-full"
          />
          <motion.span
            animate={navOpen ? { opacity: 0, scaleX: 0 } : { opacity: 1, scaleX: 1 }}
            transition={{ duration: 0.15 }}
            className="block w-5 h-px bg-current rounded-full"
          />
          <motion.span
            animate={navOpen ? { rotate: -45, y: -7 } : { rotate: 0, y: 0 }}
            transition={{ duration: 0.25 }}
            className="block w-5 h-px bg-current rounded-full"
          />
        </button>

        {/* Bottom descriptor */}
        <p className="absolute bottom-8 left-10 sm:left-6 text-xs font-mono text-muted/30 pointer-events-none select-none">
          automated deep research · bounded phase space · no data movement
        </p>
      </div>

      {/* ── Nav overlay ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {navOpen && (
          <>
            {/* Backdrop — click outside to close */}
            <motion.div
              className="fixed inset-0 z-20 bg-dark/60 backdrop-blur-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              onClick={close}
            />

            {/* Nav panel — slides down from top */}
            <motion.nav
              className="fixed top-0 left-0 right-0 z-30
                bg-dark/95 backdrop-blur-xl border-b border-primary/10
                px-16 pt-20 pb-12 md:px-10 sm:px-6"
              initial={{ y: "-100%" }}
              animate={{ y: 0 }}
              exit={{ y: "-100%" }}
              transition={{ type: "spring", stiffness: 340, damping: 38 }}
            >
              <div className="max-w-4xl mx-auto grid grid-cols-3 gap-12 md:grid-cols-2 sm:grid-cols-1 sm:gap-8">
                {NAV_SECTIONS.map((section, si) => (
                  <div key={section.label}>
                    <div className="text-xs font-mono text-muted/40 uppercase tracking-widest mb-5">
                      {section.label}
                    </div>
                    <ul className="flex flex-col gap-3">
                      {section.links.map((link, li) => (
                        <li key={link.href}>
                          <motion.div
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: si * 0.05 + li * 0.04 }}
                          >
                            <Link
                              href={link.href}
                              onClick={close}
                              className={`text-base font-medium transition-colors duration-200
                                ${link.highlight
                                  ? "text-primary hover:text-primary/80"
                                  : "text-light/70 hover:text-light"
                                }`}
                            >
                              {link.title}
                              {link.highlight && (
                                <span className="ml-2 text-xs font-mono px-1.5 py-0.5 rounded border border-primary/30 bg-primary/10 text-primary">
                                  live
                                </span>
                              )}
                            </Link>
                          </motion.div>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            </motion.nav>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
