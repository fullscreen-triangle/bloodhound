import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const EntropyFlowChart = dynamic(() => import("@/components/charts/EntropyFlowChart"), { ssr: false });

const Axiom = ({ number, title, children }) => (
  <motion.div className="card mb-6" initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}>
    <div className="text-primary font-mono text-xs mb-2">AXIOM {number}</div>
    <h3 className="text-lg font-bold mb-2">{title}</h3>
    <p className="text-muted text-sm leading-relaxed">{children}</p>
  </motion.div>
);

export default function PhaseSpace() {
  return (
    <>
      <Head><title>Bounded Phase Space | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Pillar 01</div>
            <h1 className="section-heading">Bounded Phase Space</h1>
            <p className="section-subheading mb-12">Every computation inhabits a bounded coordinate system. Every state has an address. Every trajectory has a destination. This is why the framework has guarantees—not heuristics.</p>
          </motion.div>

          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
            <div>
              <h2 className="text-2xl font-bold mb-6">S-Entropy Coordinates</h2>
              <p className="text-muted mb-6">All information in the system is represented as a point in the unit cube S = [0,1]³, with three coordinates:</p>

              <div className="space-y-4 mb-8">
                <div className="flex items-start gap-4">
                  <div className="w-3 h-3 rounded-full bg-[#E63946] mt-1.5 shrink-0" />
                  <div>
                    <div className="font-bold">S<sub>k</sub> — Knowledge Entropy</div>
                    <div className="text-muted text-sm">How much is unknown about the content. High = uncertain, low = crystallized knowledge.</div>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-3 h-3 rounded-full bg-[#457B9D] mt-1.5 shrink-0" />
                  <div>
                    <div className="font-bold">S<sub>t</sub> — Temporal Entropy</div>
                    <div className="text-muted text-sm">Uncertainty in when information was generated or how current it is.</div>
                  </div>
                </div>
                <div className="flex items-start gap-4">
                  <div className="w-3 h-3 rounded-full bg-[#2A9D8F] mt-1.5 shrink-0" />
                  <div>
                    <div className="font-bold">S<sub>e</sub> — Evolution Entropy</div>
                    <div className="text-muted text-sm">How likely the information is to change. High for active research, low for established facts.</div>
                  </div>
                </div>
              </div>

              <div className="card bg-darkTertiary border-primary/20">
                <div className="text-primary font-mono text-xs mb-2">CONSERVATION LAW</div>
                <div className="text-xl font-bold font-mono">S<sub>k</sub> + S<sub>t</sub> + S<sub>e</sub> = S<sub>total</sub></div>
                <p className="text-muted text-sm mt-2">Total entropy is conserved through every morphism chain. Knowledge gained must come from temporal or evolution entropy reduced. Nothing is created or destroyed—only transformed.</p>
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-bold mb-6">Entropy Conservation in Practice</h2>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <EntropyFlowChart width={480} height={320} />
              </div>
              <p className="text-muted text-sm mt-4">S-entropy coordinates through the ACTN3 validation: three source extractions followed by two compositions. The total (dashed line) remains constant while individual components redistribute.</p>
            </div>
          </div>

          <div className="mt-16">
            <h2 className="text-2xl font-bold mb-8">Foundational Axioms</h2>
            <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
              <Axiom number="1" title="Bounded Phase Space">
                All computational states inhabit S = [0,1]³. There is no state outside the unit cube. This boundedness ensures that every trajectory is finite, every search terminates, and every convergence is measurable.
              </Axiom>
              <Axiom number="2" title="Triple Equivalence">
                Three descriptions of the same system—oscillatory, categorical, and partition—yield identical entropy: S = k_B M ln n. Any proof in one description transfers to the others. This is the foundation of cross-modal composition.
              </Axiom>
              <Axiom number="3" title="Ternary Representation">
                Memory is addressed in base-3, not base-2. Each trit encodes three states (known, unknown, partially-known) rather than two, yielding 3^k categorical addresses. Information density per digit is maximized at base e ≈ 2.718, making ternary the closest integer optimum.
              </Axiom>
              <Axiom number="4" title="Categorical Navigation">
                Movement through phase space follows morphism chains—composable, type-checked transformations that preserve S-entropy conservation. Navigation is not search; it is categorical composition with guarantees.
              </Axiom>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
