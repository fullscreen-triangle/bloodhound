import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const StatusBadge = ({ status }) => {
  const styles = {
    complete: "bg-primary/20 text-primary",
    partial: "bg-accent/20 text-accent",
    planned: "bg-muted/20 text-muted",
    experimental: "bg-[#A23B72]/20 text-[#A23B72]",
  };
  const labels = {
    complete: "Complete",
    partial: "In Progress",
    planned: "Planned",
    experimental: "Experimental",
  };
  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-mono ${styles[status]}`}>
      {labels[status]}
    </span>
  );
};

const MilestoneCard = ({ title, status, items, timeline }) => (
  <motion.div
    className="card h-full"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <div className="flex items-center justify-between mb-3">
      <h3 className="font-bold text-lg">{title}</h3>
      <StatusBadge status={status} />
    </div>
    {timeline && <div className="text-muted text-xs font-mono mb-4">{timeline}</div>}
    <ul className="space-y-2">
      {items.map((item, i) => (
        <li key={i} className="flex items-start gap-2 text-sm">
          <span className={`mt-0.5 shrink-0 ${item.done ? "text-primary" : "text-muted/40"}`}>
            {item.done ? "✓" : "○"}
          </span>
          <span className={item.done ? "text-muted" : "text-muted/70"}>{item.text}</span>
        </li>
      ))}
    </ul>
  </motion.div>
);

const ValidationRow = ({ theorem, status, notes }) => (
  <div className="flex items-center justify-between py-3 border-b border-primary/5 last:border-0">
    <div className="flex items-center gap-3">
      <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold ${
        status === "verified" ? "bg-primary/20 text-primary" : "bg-accent/20 text-accent"
      }`}>
        {status === "verified" ? "✓" : "⚠"}
      </div>
      <span className="text-sm font-medium">{theorem}</span>
    </div>
    <span className="text-muted text-xs font-mono">{notes}</span>
  </div>
);

export default function Roadmap() {
  return (
    <>
      <Head>
        <title>Roadmap | Bloodhound</title>
        <meta name="description" content="Bloodhound development roadmap: what's built, what's in progress, and what's planned for the distributed virtual machine framework." />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Roadmap</div>
            <h1 className="section-heading">Development Trajectory</h1>
            <p className="section-subheading mb-16">
              Where the framework stands today and where it is heading.
              Each phase is independently valuable while contributing to the whole.
            </p>
          </motion.div>

          {/* Current Status Overview */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Current Status</h2>
            <div className="grid grid-cols-4 gap-4 md:grid-cols-2 sm:grid-cols-1 mb-8">
              <motion.div className="text-center card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
                <div className="text-3xl font-bold text-primary">10/10</div>
                <div className="text-muted text-sm mt-1">Theorems Verified</div>
              </motion.div>
              <motion.div className="text-center card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.1 }}>
                <div className="text-3xl font-bold text-primary">7/7</div>
                <div className="text-muted text-sm mt-1">Validation Checks</div>
              </motion.div>
              <motion.div className="text-center card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.2 }}>
                <div className="text-3xl font-bold text-accent">3</div>
                <div className="text-muted text-sm mt-1">Domain Compilers</div>
              </motion.div>
              <motion.div className="text-center card" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: 0.3 }}>
                <div className="text-3xl font-bold text-accent">47+</div>
                <div className="text-muted text-sm mt-1">Purpose Models</div>
              </motion.div>
            </div>
          </div>

          {/* Theorem Verification Status */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-6">Theorem Verification</h2>
            <div className="card max-w-3xl">
              <ValidationRow theorem="Triple Equivalence" status="verified" notes="All M,n combinations" />
              <ValidationRow theorem="Trit-Cell Correspondence" status="verified" notes="Bijective for k=3,4,5,6" />
              <ValidationRow theorem="Trajectory-Position Identity" status="verified" notes="100 samples" />
              <ValidationRow theorem="Completion Equivalence" status="verified" notes="navigate ≡ verify" />
              <ValidationRow theorem="Zero-Cost Sorting" status="verified" notes="E = 0 for 50 sorts" />
              <ValidationRow theorem="Observable Commutation" status="verified" notes="All measurements commute" />
              <ValidationRow theorem="Exponential Decay" status="verified" notes="τ_measured/τ_theory = 1.00" />
              <ValidationRow theorem="Central State Impossibility" status="verified" notes="E diverges as σ → 0" />
              <ValidationRow theorem="Distance Independence" status="partial" notes="Correlation 0.3554 (threshold 0.3)" />
              <ValidationRow theorem="Continuous Emergence" status="verified" notes="Convergence confirmed" />
            </div>
          </div>

          {/* Development Phases */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Development Phases</h2>

            {/* Phase 1 */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary font-bold text-sm">1</div>
                <h3 className="text-xl font-bold">Foundation</h3>
                <StatusBadge status="complete" />
              </div>
              <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
                <MilestoneCard
                  title="Mathematical Framework"
                  status="complete"
                  items={[
                    { text: "Single-axiom derivation (bounded phase space)", done: true },
                    { text: "Triple equivalence theorem proof", done: true },
                    { text: "S-entropy coordinate system specification", done: true },
                    { text: "Categorical distance formula", done: true },
                    { text: "Trit-cell correspondence (bijective proof)", done: true },
                  ]}
                />
                <MilestoneCard
                  title="Core Implementation"
                  status="complete"
                  items={[
                    { text: "S-entropy coordinate system (Python)", done: true },
                    { text: "Ternary representation & addressing", done: true },
                    { text: "Trajectory navigation engine", done: true },
                    { text: "Categorical memory hierarchy", done: true },
                    { text: "Maxwell demon controller", done: true },
                  ]}
                />
                <MilestoneCard
                  title="Validation Suite"
                  status="complete"
                  items={[
                    { text: "10 core theorem verifications", done: true },
                    { text: "ACTN3 end-to-end validation (live APIs)", done: true },
                    { text: "7/7 empirical checks passed", done: true },
                    { text: "10⁸x compression ratio demonstrated", done: true },
                    { text: "Validation panel generation", done: true },
                  ]}
                />
              </div>
            </div>

            {/* Phase 2 */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center text-accent font-bold text-sm">2</div>
                <h3 className="text-xl font-bold">Expansion</h3>
                <StatusBadge status="partial" />
              </div>
              <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
                <MilestoneCard
                  title="Rust VM Core"
                  status="partial"
                  timeline="12–18 months"
                  items={[
                    { text: "Consciousness-aware processing module", done: true },
                    { text: "Entropy & oscillatory dynamics engines", done: true },
                    { text: "Runtime loop with async execution", done: true },
                    { text: "Production Triangle compiler with type checker", done: false },
                    { text: "Formal verification in Lean 4", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Domain Compilers"
                  status="partial"
                  timeline="18–24 months"
                  items={[
                    { text: "Mufakose Genomics compiler", done: true },
                    { text: "Mufakose Metabolomics compiler", done: true },
                    { text: "Mufakose Pharmaceutical compiler", done: true },
                    { text: "Clinical imaging compiler", done: false },
                    { text: "Environmental monitoring compiler", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Distributed Coordination"
                  status="partial"
                  timeline="18–24 months"
                  items={[
                    { text: "Network-gas correspondence (theory)", done: true },
                    { text: "Variance restoration (verified)", done: true },
                    { text: "Central State Impossibility (verified)", done: true },
                    { text: "Multi-node coordination protocol", done: false },
                    { text: "Cross-institutional pilot deployment", done: false },
                  ]}
                />
              </div>
            </div>

            {/* Phase 3 */}
            <div className="mb-8">
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-lg bg-muted/20 flex items-center justify-center text-muted font-bold text-sm">3</div>
                <h3 className="text-xl font-bold">Scale</h3>
                <StatusBadge status="planned" />
              </div>
              <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
                <MilestoneCard
                  title="Federated Network"
                  status="planned"
                  timeline="24–36 months"
                  items={[
                    { text: "Structural privacy implementation", done: false },
                    { text: "Cross-institutional validation framework", done: false },
                    { text: "Pilot: 3–5 research institutions", done: false },
                    { text: "Regulatory compliance (HIPAA/GDPR)", done: false },
                    { text: "Production-grade networking (libp2p)", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Metacognitive Pipeline"
                  status="planned"
                  timeline="12–24 months"
                  items={[
                    { text: "Question decomposition engine", done: false },
                    { text: "Information-yield resource allocator", done: false },
                    { text: "DPP-based candidate generator", done: false },
                    { text: "5-dimension quality evaluation", done: false },
                    { text: "Autonomous refinement orchestrator", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Clinical Translation"
                  status="planned"
                  timeline="24–36 months"
                  items={[
                    { text: "End-to-end clinical research automation", done: false },
                    { text: "Multi-omics disease characterization", done: false },
                    { text: "Cross-jurisdictional pharmacovigilance", done: false },
                    { text: "Precision medicine case studies", done: false },
                    { text: "Regulatory compliance documentation", done: false },
                  ]}
                />
              </div>
            </div>

            {/* Phase 4: Experimental */}
            <div>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-8 h-8 rounded-lg bg-[#A23B72]/20 flex items-center justify-center text-[#A23B72] font-bold text-sm">4</div>
                <h3 className="text-xl font-bold">Frontier</h3>
                <StatusBadge status="experimental" />
              </div>
              <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
                <MilestoneCard
                  title="Meta-Consciousness"
                  status="experimental"
                  items={[
                    { text: "Processors validating their own mathematical necessity", done: false },
                    { text: "Self-modifying consciousness loops", done: false },
                    { text: "Consciousness reflection architecture", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Quantum & Relativistic"
                  status="experimental"
                  items={[
                    { text: "Quantum S-entropy superposition", done: false },
                    { text: "Relativistic coordinate systems", done: false },
                    { text: "Multidimensional S-entropy (beyond [0,1]³)", done: false },
                  ]}
                />
                <MilestoneCard
                  title="Autonomous Learning"
                  status="experimental"
                  items={[
                    { text: "Few-shot domain adaptation", done: false },
                    { text: "Zero-shot consciousness transfer", done: false },
                    { text: "Self-supervised improvement", done: false },
                  ]}
                />
              </div>
            </div>
          </div>

          {/* Enhancement Mechanisms */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-6">Trans-Planckian Enhancement</h2>
            <p className="text-muted mb-6 max-w-3xl leading-relaxed">
              Five multiplicative enhancement mechanisms achieve computational temporal precision of ~10⁻¹⁵² seconds — far beyond the Planck time of 5.39 × 10⁻⁴⁴ seconds.
            </p>
            <div className="grid grid-cols-5 gap-4 lg:grid-cols-3 md:grid-cols-2 sm:grid-cols-1">
              {[
                { name: "Ternary Encoding", factor: "10³·⁵", desc: "(3/2)^k advantage" },
                { name: "Multi-Modal Synthesis", factor: "10²⁰", desc: "n^(m(m-1)/2) combinations" },
                { name: "Harmonic Coincidence", factor: "10¹·²", desc: "E/N resonance" },
                { name: "Trajectory Completion", factor: "10¹⁶·²", desc: "ωτ/(2π) precision" },
                { name: "Continuous Refinement", factor: "10¹⁰⁰", desc: "exp(ωτ/N₀) enhancement" },
              ].map((m, i) => (
                <motion.div
                  key={i}
                  className="card text-center"
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 }}
                >
                  <div className="text-xl font-bold font-mono text-primary mb-1">{m.factor}</div>
                  <div className="text-sm font-bold mb-1">{m.name}</div>
                  <div className="text-muted text-xs">{m.desc}</div>
                </motion.div>
              ))}
            </div>
            <div className="mt-4 text-center">
              <div className="inline-block card bg-darkTertiary border-primary/20">
                <span className="text-muted text-sm">Total enhancement: </span>
                <span className="text-primary font-mono font-bold">~10¹⁴⁰·⁹</span>
                <span className="text-muted text-sm"> → temporal precision </span>
                <span className="text-primary font-mono font-bold">δt ≈ 10⁻¹⁵² s</span>
              </div>
            </div>
          </div>

          {/* CTA */}
          <div className="text-center border-t border-primary/10 pt-16">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2 className="text-3xl font-bold mb-4 md:text-2xl">Help shape the trajectory.</h2>
              <p className="text-muted text-lg mb-8 max-w-xl mx-auto">
                Every phase is independently fundable. Every contribution — code, expertise, infrastructure, or capital — accelerates convergence.
              </p>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link href="/collaborate" className="btn-primary">Partner With Us</Link>
                <Link href="/validation" className="btn-outline">See the Evidence</Link>
              </div>
            </motion.div>
          </div>
        </Layout>
      </section>
    </>
  );
}
