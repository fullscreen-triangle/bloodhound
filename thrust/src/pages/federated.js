import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const ParadigmChart = dynamic(() => import("@/components/charts/ParadigmChart"), { ssr: false });
const ScalingChart = dynamic(() => import("@/components/charts/ScalingChart"), { ssr: false });

const ComparisonRow = ({ paradigm, moves, amount, privacy, color }) => (
  <div className={`flex items-center gap-6 p-4 rounded-xl border border-[${color}]/20 bg-[${color}]/5`}>
    <div className={`w-3 h-3 rounded-full shrink-0`} style={{ backgroundColor: color }} />
    <div className="flex-1">
      <div className="font-bold">{paradigm}</div>
      <div className="text-muted text-sm">Moves: {moves}</div>
    </div>
    <div className="text-right">
      <div className="font-bold font-mono" style={{ color }}>{amount}</div>
      <div className="text-muted text-xs">{privacy}</div>
    </div>
  </div>
);

export default function Federated() {
  return (
    <>
      <Head><title>Federated Understanding | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Pillar 03</div>
            <h1 className="section-heading">Federated Understanding</h1>
            <p className="section-subheading mb-12">A new computational paradigm. Not centralized analysis, which moves all data. Not federated learning, which moves model parameters. Federated understanding moves question-shaped understanding fragments—and nothing else.</p>
          </motion.div>

          {/* Paradigm comparison */}
          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1 mb-16">
            <div>
              <h2 className="text-2xl font-bold mb-6">Three Paradigms, Orders of Magnitude Apart</h2>
              <div className="space-y-3 mb-6">
                <ComparisonRow paradigm="Centralized" moves="Raw data to server" amount="218.9 GB" privacy="No privacy" color="#E63946" />
                <ComparisonRow paradigm="Federated Learning" moves="Model parameters" amount="286.1 MB" privacy="Differential privacy" color="#F4A261" />
                <ComparisonRow paradigm="Federated Understanding" moves="Understanding fragments" amount="968 B" privacy="Structural privacy" color="#2A9D8F" />
              </div>
              <div className="card bg-darkTertiary border-primary/20">
                <div className="text-primary font-mono text-xs mb-2">PARADIGM COMPARISON THEOREM</div>
                <p className="text-muted text-sm">Network traffic under federated understanding is O(I(D; A_Q)), compared to O(H(D)) for federated learning and O(|D|) for centralization. Since I(D; A_Q) ≪ H(D) ≪ |D| for surgical questions, the reduction is orders of magnitude.</p>
              </div>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-6">Network Transfer Comparison</h2>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <ParadigmChart width={480} height={320} />
              </div>
            </div>
          </div>

          {/* Scaling */}
          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1 mb-16">
            <div>
              <h2 className="text-2xl font-bold mb-6">Scaling With Source Count</h2>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <ScalingChart width={480} height={320} />
              </div>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-6">Why This Matters</h2>
              <div className="space-y-4">
                <div className="card">
                  <h3 className="font-bold mb-2">Structural Privacy</h3>
                  <p className="text-muted text-sm">Irrelevant data is never processed—not merely protected with noise. This is stronger than differential privacy: there is no privacy-utility trade-off because irrelevant information never enters the computation.</p>
                </div>
                <div className="card">
                  <h3 className="font-bold mb-2">No Integration Step</h3>
                  <p className="text-muted text-sm">All modalities map to S-space through observe bridges. Integration is composition—a built-in categorical operation. There is no separate ETL pipeline, no schema matching, no data harmonization.</p>
                </div>
                <div className="card">
                  <h3 className="font-bold mb-2">No Stale Data</h3>
                  <p className="text-muted text-sm">Representations are generated on demand from the current question against the current data. There is no cached representation that can become outdated.</p>
                </div>
                <div className="card">
                  <h3 className="font-bold mb-2">Reproducibility by Construction</h3>
                  <p className="text-muted text-sm">The trajectory T* encodes complete methodological provenance. Reproducing the analysis requires only the protocol specification and access to the data sources.</p>
                </div>
              </div>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
