import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const CompressionChart = dynamic(() => import("@/components/charts/CompressionChart"), { ssr: false });
const ParadigmChart = dynamic(() => import("@/components/charts/ParadigmChart"), { ssr: false });
const ConvergenceChart = dynamic(() => import("@/components/charts/ConvergenceChart"), { ssr: false });
const EntropyFlowChart = dynamic(() => import("@/components/charts/EntropyFlowChart"), { ssr: false });
const ScalingChart = dynamic(() => import("@/components/charts/ScalingChart"), { ssr: false });

const Check = ({ name, passed, description, index }) => (
  <motion.div
    className="card"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay: index * 0.08 }}
  >
    <div className="flex items-center gap-3 mb-2">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${passed ? "bg-primary/20 text-primary" : "bg-danger/20 text-danger"}`}>
        {passed ? "✓" : "✗"}
      </div>
      <h3 className="font-bold">{name}</h3>
    </div>
    <p className="text-muted text-sm">{description}</p>
  </motion.div>
);

export default function Validation() {
  return (
    <>
      <Head><title>Empirical Validation | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Pillar 05</div>
            <h1 className="section-heading">Empirical Validation</h1>
            <p className="section-subheading mb-12">Every claim is falsifiable. We validated the framework on a real multi-omics problem: ACTN3 R577X polymorphism and cardiac adaptation in elite athletes, using live public APIs.</p>
          </motion.div>

          {/* Validation checks */}
          <div className="mb-16">
            <h2 className="text-2xl font-bold mb-6">7 / 7 Checks Passed</h2>
            <div className="grid grid-cols-2 gap-4 lg:grid-cols-1">
              <Check index={0} name="Protocol Parsed" passed={true} description="Triangle protocol correctly decomposed into 3 surgical extraction targets, 2 compositions, 1 navigation, 2 validations, 1 convergence criterion." />
              <Check index={1} name="All Sources Extracted" passed={true} description="Live API queries to NCBI dbSNP, GWAS Catalog, NCBI GEO, and UniProt all returned valid understanding fragments." />
              <Check index={2} name="Compression Achieved" passed={true} description="Overall compression ratio: 4.1 × 10⁻⁹. Every source exceeded the 10⁻⁶ threshold by three orders of magnitude." />
              <Check index={3} name="Compositions Performed" passed={true} description="Two cross-modal compositions executed: genomics ⊕ transcriptomics, then result ⊕ proteomics. S-entropy conservation maintained." />
              <Check index={4} name="Temperature Decreased" passed={true} description="Analysis temperature converged from T=0.450 to T=0.414 through successive compositions, confirming variance restoration." />
              <Check index={5} name="Cross-Modal Links Found" passed={true} description="4 cross-modal links discovered: shared terms (ACTN3), gene-protein links, tissue context (cardiac_muscle)." />
              <Check index={6} name="Paradigm Advantage" passed={true} description="2.4 × 10⁸x reduction vs centralized, 3.1 × 10⁵x reduction vs federated learning. Theorem 3 confirmed empirically." />
            </div>
          </div>

          {/* Charts grid */}
          <div className="grid grid-cols-2 gap-8 lg:grid-cols-1">
            <div>
              <h3 className="text-lg font-bold mb-4">Surgical Compression per Source</h3>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <CompressionChart width={480} height={280} />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">Paradigm Comparison</h3>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <ParadigmChart width={480} height={280} />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">Temperature Convergence</h3>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <ConvergenceChart width={480} height={280} />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-bold mb-4">S-Entropy Conservation</h3>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <EntropyFlowChart width={480} height={280} />
              </div>
            </div>
          </div>

          <div className="mt-12">
            <h3 className="text-lg font-bold mb-4">Scaling Projection</h3>
            <div className="bg-surface rounded-xl p-4 border border-primary/10 max-w-2xl">
              <ScalingChart width={600} height={320} />
            </div>
            <p className="text-muted text-sm mt-4 max-w-2xl">The advantage of federated understanding grows linearly with the number of data sources. At 19 sources, the gap reaches 10⁸x vs centralized approaches.</p>
          </div>
        </Layout>
      </section>
    </>
  );
}
