import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const ConvergenceChart = dynamic(() => import("@/components/charts/ConvergenceChart"), { ssr: false });

const Stage = ({ number, title, description, detail, color }) => (
  <motion.div
    className="relative"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay: number * 0.08 }}
  >
    <div className="card h-full">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-sm font-bold" style={{ backgroundColor: color + "20", color }}>
          {number}
        </div>
        <h3 className="font-bold text-lg">{title}</h3>
      </div>
      <p className="text-muted text-sm mb-3">{description}</p>
      <div className="text-xs font-mono text-primary/60 bg-primary/5 rounded-lg p-3">{detail}</div>
    </div>
  </motion.div>
);

export default function Pipeline() {
  return (
    <>
      <Head><title>Metacognitive Pipeline | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Pillar 04</div>
            <h1 className="section-heading">Metacognitive Pipeline</h1>
            <p className="section-subheading mb-12">Six stages replace the human researcher sitting in front of a computer for hours. The system decomposes the question, allocates resources, generates candidates, evaluates quality, verifies correctness, and orchestrates refinement—autonomously.</p>
          </motion.div>

          <div className="grid grid-cols-3 gap-6 mb-16 lg:grid-cols-2 md:grid-cols-1">
            <Stage number={1} title="Decompose" color="#E63946"
              description="The research question is decomposed into atomic sub-questions, each targeting a specific information need from a specific modality."
              detail="Q → {q₁, q₂, ..., qₖ} where each qᵢ maps to a single slice statement" />
            <Stage number={2} title="Allocate" color="#F4A261"
              description="Computational resources are allocated across sub-questions based on expected information yield—more resources for higher-yield extractions."
              detail="GQIC allocation: resource ∝ I(qᵢ; A_Q) / cost(qᵢ)" />
            <Stage number={3} title="Generate" color="#2A9D8F"
              description="Multiple candidate extractions are generated for each sub-question using Determinantal Point Processes for diversity."
              detail="DPP kernel K balances quality (diagonal) with diversity (off-diagonal)" />
            <Stage number={4} title="Evaluate" color="#457B9D"
              description="Each candidate is evaluated across five quality dimensions: factual accuracy, logical coherence, domain consistency, completeness, and uncertainty calibration."
              detail="Quality vector q ∈ [0,1]⁵ with weighted aggregation" />
            <Stage number={5} title="Verify" color="#A23B72"
              description="Domain-expert consensus, adversarial evaluation, and formal verification through proof assistants ensure conclusions are correct."
              detail="Multi-expert: consensus ∧ adversarial ∧ formal (Lean 4/Coq)" />
            <Stage number={6} title="Orchestrate" color="#F18F01"
              description="The orchestrator decides: accept, refine, redirect, or escalate. Refinement loops continue until the convergence criterion is met."
              detail="Action = argmax P(converge | state, action) subject to budget" />
          </div>

          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
            <div>
              <h2 className="text-2xl font-bold mb-6">Convergence: Gas → Liquid → Crystal</h2>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <ConvergenceChart width={480} height={320} />
              </div>
              <p className="text-muted text-sm mt-4">Analysis temperature decreases through refinement iterations. The system transitions from gaseous (high uncertainty, scattered fragments) through liquid (clustering, cross-links forming) to crystal (converged, validated answer).</p>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-6">Phase Transitions</h2>
              <div className="space-y-4">
                <div className="card border-l-4 border-l-[#E63946]">
                  <h3 className="font-bold mb-1">Gas Phase <span className="text-muted font-normal text-sm">T &gt; 0.5</span></h3>
                  <p className="text-muted text-sm">Understanding fragments are dispersed, uncorrelated. Each source extraction produces an isolated signature with no cross-modal connections. High entropy, low confidence.</p>
                </div>
                <div className="card border-l-4 border-l-[#F4A261]">
                  <h3 className="font-bold mb-1">Liquid Phase <span className="text-muted font-normal text-sm">0.2 &lt; T &lt; 0.5</span></h3>
                  <p className="text-muted text-sm">Fragments begin clustering through composition. Cross-modal links emerge (gene-protein connections, tissue context). Semantic structure forms but has not solidified.</p>
                </div>
                <div className="card border-l-4 border-l-[#2A9D8F]">
                  <h3 className="font-bold mb-1">Crystal Phase <span className="text-muted font-normal text-sm">T &lt; 0.2</span></h3>
                  <p className="text-muted text-sm">The answer has crystallized. All fragments are integrated, cross-validated, and formally verified. The trajectory is complete, reproducible, and provably correct.</p>
                </div>
              </div>

              <div className="card bg-darkTertiary border-primary/20 mt-6">
                <div className="text-primary font-mono text-xs mb-2">CONVERGENCE THEOREM</div>
                <p className="text-muted text-sm">Under quality-monotonic refinement and sufficient candidate diversity, the analysis temperature T(n) decreases exponentially: T(n) ≤ T₀ · e^(-λn). Convergence to crystal phase is guaranteed in finite iterations.</p>
              </div>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
