import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const TimelineItem = ({ year, title, description, index }) => (
  <motion.div
    className="relative pl-8 pb-10 border-l border-primary/20 last:pb-0"
    initial={{ opacity: 0, x: -20 }}
    whileInView={{ opacity: 1, x: 0 }}
    viewport={{ once: true }}
    transition={{ delay: index * 0.1 }}
  >
    <div className="absolute left-0 top-0 w-3 h-3 rounded-full bg-primary -translate-x-[7px]" />
    <div className="text-primary font-mono text-xs mb-1">{year}</div>
    <h3 className="text-lg font-bold mb-2">{title}</h3>
    <p className="text-muted text-sm leading-relaxed">{description}</p>
  </motion.div>
);

const PrincipleCard = ({ title, description, index }) => (
  <motion.div
    className="card h-full"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay: index * 0.1 }}
  >
    <h3 className="text-lg font-bold mb-3">{title}</h3>
    <p className="text-muted text-sm leading-relaxed">{description}</p>
  </motion.div>
);

export default function About() {
  return (
    <>
      <Head>
        <title>About | Bloodhound</title>
        <meta name="description" content="The origin, vision, and team behind Bloodhound — a distributed virtual machine framework for automated deep research." />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">About</div>
            <h1 className="section-heading">The Framework &amp; Its Origin</h1>
            <p className="section-subheading mb-16">
              Bloodhound began with a single axiom and the conviction that computation has been asking the wrong question.
              Not &ldquo;how do we process faster?&rdquo; but &ldquo;why are we processing at all?&rdquo;
            </p>
          </motion.div>

          {/* The Founding Axiom */}
          <div className="mb-20">
            <motion.div
              className="max-w-4xl"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
            >
              <h2 className="text-2xl font-bold mb-8">The Founding Axiom</h2>
              <div className="card bg-darkTertiary border-primary/20 mb-8">
                <div className="text-primary font-mono text-xs mb-3">AXIOM</div>
                <div className="text-2xl font-bold font-mono mb-4 md:text-xl">
                  &ldquo;Physical systems occupy finite phase space volume.&rdquo;
                </div>
                <div className="text-muted leading-relaxed space-y-4">
                  <p>
                    From this single statement, the entire Bloodhound framework derives. Bounded phase space implies
                    Poincar&eacute; recurrence — trajectories return arbitrarily close to initial configurations.
                    Recurrence implies oscillatory dynamics. Oscillatory dynamics, categorical structure, and partition
                    are three descriptions of the same mathematical object.
                  </p>
                  <p>
                    The consequence is the <span className="text-light font-semibold">Triple Equivalence Theorem</span>:
                  </p>
                  <div className="bg-dark/50 rounded-lg p-4 font-mono text-center text-lg">
                    S<sub>osc</sub> = S<sub>cat</sub> = S<sub>part</sub> = k<sub>B</sub> &middot; M &middot; ln(n)
                  </div>
                  <p>
                    Oscillation, category, and partition are not analogies — they are mathematical identities. Any proof
                    in one description transfers to the others. This is the foundation of cross-modal composition.
                  </p>
                </div>
              </div>
            </motion.div>
          </div>

          {/* The Paradigm Shift */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">The Paradigm Shift</h2>
            <div className="grid grid-cols-2 gap-8 lg:grid-cols-1">
              <motion.div
                className="card border-l-4 border-l-danger"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
              >
                <div className="text-danger font-mono text-xs mb-2">CONVENTIONAL COMPUTING</div>
                <h3 className="text-lg font-bold mb-3">Computation as Instruction Execution</h3>
                <p className="text-muted text-sm leading-relaxed">
                  Traditional computing executes instructions on unbounded tape. Data exists independently of questions.
                  You load everything, filter, transform, reduce. The computational cost scales with data size,
                  regardless of how much of that data is relevant to your actual question.
                </p>
              </motion.div>
              <motion.div
                className="card border-l-4 border-l-primary"
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
              >
                <div className="text-primary font-mono text-xs mb-2">BLOODHOUND</div>
                <h3 className="text-lg font-bold mb-3">Computation as Trajectory Completion</h3>
                <p className="text-muted text-sm leading-relaxed">
                  Bloodhound reformulates computation as navigation through bounded three-dimensional phase space.
                  Answers exist as locations in categorical space — navigated to, not computed. The question creates the
                  data representation. Without a question, no representation exists. The path taken IS the address IS
                  the result.
                </p>
              </motion.div>
            </div>
          </div>

          {/* Core Principles */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Core Principles</h2>
            <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
              <PrincipleCard
                index={0}
                title="Navigation, Not Computation"
                description="Answers are locations in categorical space. The system navigates to them through morphism chains — composable, type-checked transformations that preserve S-entropy conservation. Navigation cost is independent of dataset size."
              />
              <PrincipleCard
                index={1}
                title="Structural Privacy by Construction"
                description="Irrelevant data is never processed — not merely protected with noise or differential privacy. There is no privacy-utility trade-off because irrelevant information never enters the computation. Privacy is architectural, not parametric."
              />
              <PrincipleCard
                index={2}
                title="Mathematical Guarantees"
                description="Every claim rests on formal theorems, not heuristics. Triple equivalence, S-entropy conservation, convergence bounds, and information minimality are all provable properties — verified in the validation suite and targeted for formal proof in Lean 4."
              />
              <PrincipleCard
                index={3}
                title="Question-Shaped Understanding"
                description="What traverses the network is not data, not model parameters, but understanding fragments shaped by the research question. 968 bytes instead of 218.9 GB. The question is the scalpel."
              />
              <PrincipleCard
                index={4}
                title="Trajectory-Address Equivalence"
                description="The path taken through categorical space simultaneously encodes position, trajectory, and address. A k-trit sequence is a bijective map to one cell in a 3^k partition. Navigation, addressing, and data identification are the same operation."
              />
              <PrincipleCard
                index={5}
                title="Open Science"
                description="MIT licensed. Reproducible research. Every validation runs against live public APIs. The framework is designed for collaborative development — domain experts, systems engineers, formal methods researchers, and clinicians all have entry points."
              />
            </div>
          </div>

          {/* Research Context */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Research Context</h2>
            <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
              <div>
                <p className="text-muted leading-relaxed mb-6">
                  Bloodhound draws on foundational work across statistical mechanics, category theory, information
                  theory, and distributed systems. The framework synthesizes insights from:
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <span className="text-primary mt-1 shrink-0">&#x2022;</span>
                    <div>
                      <span className="font-semibold">Poincar&eacute; (1890)</span>
                      <span className="text-muted text-sm"> — Recurrence theorem for bounded dynamical systems. The mathematical foundation for trajectory completion.</span>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-primary mt-1 shrink-0">&#x2022;</span>
                    <div>
                      <span className="font-semibold">Boltzmann (1877)</span>
                      <span className="text-muted text-sm"> — Statistical mechanics and the relationship between entropy and microstate counting. The basis for S-entropy coordinates.</span>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-primary mt-1 shrink-0">&#x2022;</span>
                    <div>
                      <span className="font-semibold">Landauer (1961)</span>
                      <span className="text-muted text-sm"> — Irreversibility and heat generation in computing. The thermodynamic cost that the Maxwell demon controller circumvents through categorical operations.</span>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <span className="text-primary mt-1 shrink-0">&#x2022;</span>
                    <div>
                      <span className="font-semibold">Bennett (1982)</span>
                      <span className="text-muted text-sm"> — Reversible computation and the thermodynamics of information processing. Theoretical justification for zero-cost categorical sorting.</span>
                    </div>
                  </div>
                </div>
              </div>
              <div>
                <TimelineItem index={0} year="Foundation" title="Single Axiom Derivation"
                  description="The entire framework derived from one axiom: physical systems occupy finite phase space volume. From this, Poincaré recurrence, oscillatory dynamics, and categorical structure emerge naturally." />
                <TimelineItem index={1} year="Theory" title="Triple Equivalence Proof"
                  description="Proof that oscillatory, categorical, and partition descriptions yield identical entropy. This enables cross-modal composition — the mathematical basis for federated understanding." />
                <TimelineItem index={2} year="Implementation" title="St-Hurbert Engine & Triangle DSL"
                  description="Construction of the execution engine (S-entropy core, categorical memory, Maxwell demon controller) and the Triangle domain-specific language for research protocol specification." />
                <TimelineItem index={3} year="Validation" title="ACTN3 Multi-Omics Demonstration"
                  description="End-to-end validation on a real multi-omics problem: ACTN3 R577X polymorphism and cardiac adaptation in elite athletes. 7/7 checks passed, 10⁸x compression achieved." />
              </div>
            </div>
          </div>

          {/* Team */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Team</h2>
            <div className="grid grid-cols-2 gap-8 lg:grid-cols-1">
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
              >
                <h3 className="text-xl font-bold mb-1">Kundai Farai Sachikonye</h3>
                <div className="text-primary text-sm mb-4">Principal Researcher &amp; Framework Architect</div>
                <p className="text-muted text-sm leading-relaxed mb-4">
                  Independent researcher in theoretical physics and computational engineering. Developed the
                  foundational theory, architecture, and implementation of the Bloodhound framework — from the
                  single-axiom derivation through the distributed virtual machine to the domain-specific compilers.
                </p>
                <div className="text-muted text-sm font-mono">
                  kundai.sachikonye@wzw.tum.de
                </div>
              </motion.div>
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 }}
              >
                <h3 className="text-xl font-bold mb-1">Open Collaboration</h3>
                <div className="text-accent text-sm mb-4">Seeking Partners Across Disciplines</div>
                <p className="text-muted text-sm leading-relaxed mb-4">
                  Bloodhound is designed for collaborative development. The framework needs domain experts
                  (bioinformaticians, clinicians, environmental scientists), systems engineers (Rust, distributed systems),
                  formal methods researchers (Lean 4, Coq), and institutional partners for pilot deployments.
                </p>
                <Link href="/collaborate" className="text-primary text-sm font-medium hover:underline">
                  See collaboration tracks &rarr;
                </Link>
              </motion.div>
            </div>
          </div>

          {/* Publications */}
          <div className="border-t border-primary/10 pt-16">
            <h2 className="text-2xl font-bold mb-8">Publications</h2>
            <div className="space-y-6 max-w-3xl">
              <div className="card">
                <div className="text-primary font-mono text-xs mb-2">PRIMARY</div>
                <h3 className="font-bold mb-2">Bloodhound: A Distributed Virtual Machine Architecture Based on Categorical Navigation in Bounded Phase Space</h3>
                <p className="text-muted text-sm">Sachikonye, K.F. (2025). Introduces the single-axiom derivation, triple equivalence theorem, S-entropy coordinate system, and the federated understanding paradigm.</p>
              </div>
              <div className="card">
                <div className="text-accent font-mono text-xs mb-2">DOMAIN APPLICATION</div>
                <h3 className="font-bold mb-2">Mufakose Frameworks: Domain-Specific Compilers for Genomics, Metabolomics, and Pharmaceutical Research</h3>
                <p className="text-muted text-sm">Sachikonye, K.F. (2025). Demonstrates application of categorical navigation to variant detection, mass spectrometry analysis, and drug discovery with O(log N) computational complexity.</p>
              </div>
              <div className="card">
                <div className="text-accent font-mono text-xs mb-2">TEMPORAL RESOLUTION</div>
                <h3 className="font-bold mb-2">Trans-Planckian Temporal Resolution Through Categorical Enhancement Mechanisms</h3>
                <p className="text-muted text-sm">Sachikonye, K.F. (2025). Five multiplicative enhancement mechanisms achieving computational temporal precision of ~10⁻¹⁵² seconds — far beyond the Planck time of 5.39 × 10⁻⁴⁴ seconds.</p>
              </div>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
