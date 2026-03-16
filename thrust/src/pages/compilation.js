import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import dynamic from "next/dynamic";

const CompressionChart = dynamic(() => import("@/components/charts/CompressionChart"), { ssr: false });

export default function Compilation() {
  return (
    <>
      <Head><title>Problem-Directed Compilation | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Pillar 02</div>
            <h1 className="section-heading">Problem-Directed Compilation</h1>
            <p className="section-subheading mb-12">A research question is not a database query. It is a compilation target. The question compiles into morphism chains that surgically extract only what is relevant to the answer.</p>
          </motion.div>

          <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
            <div>
              <h2 className="text-2xl font-bold mb-6">The Triangle DSL</h2>
              <p className="text-muted mb-6">Research protocols are written in Triangle, an LL(1) domain-specific language where each statement maps to a morphism chain through S-entropy space.</p>

              <div className="bg-surface rounded-xl p-6 border border-primary/10 font-mono text-sm overflow-x-auto">
                <pre className="text-muted">
                  <span className="text-primary">investigate</span>{` "Association between ACTN3
  genotype and cardiac adaptation
  in elite sprinters"
  `}<span className="text-primary">with</span>{` confidence > 0.95
  `}<span className="text-primary">with</span>{` significance < 0.01

`}<span className="text-primary">parallel</span>{` {
  genotype = `}<span className="text-accent">slice</span>{` genomics.ACTN3
    @ cohort(elite_sprinters)
    @ variant(rs1815739)

  cardiac = `}<span className="text-accent">slice</span>{` echocardiography
    @ cohort(elite_sprinters)
    @ measure(LV_mass, EF, GLS)

  protein = `}<span className="text-accent">slice</span>{` proteomics
    @ target(alpha_actinin_3)
    @ tissue(cardiac_muscle)
}

joined = `}<span className="text-accent">compose</span>{` genotype `}<span className="text-primary">with</span>{` cardiac
  `}<span className="text-primary">preserving</span>{` athlete_id

result = `}<span className="text-accent">navigate</span>{` joined `}<span className="text-primary">to</span>{` target
  `}<span className="text-primary">via</span>{` correlation_analysis

`}<span className="text-primary">converge at</span>{` confidence > 0.95`}
                </pre>
              </div>
              <p className="text-muted text-sm mt-4">The researcher specifies <em>what</em> to investigate and <em>what evidence</em> is needed. The system handles <em>how</em>: which domain models to invoke, what morphism chains to construct, when the analysis has converged.</p>
            </div>

            <div>
              <h2 className="text-2xl font-bold mb-6">Surgical Extraction Results</h2>
              <div className="bg-surface rounded-xl p-4 border border-primary/10">
                <CompressionChart width={480} height={320} />
              </div>
              <p className="text-muted text-sm mt-4">Ghost bars: full dataset size. Solid bars: surgically extracted data. Each source achieves 10⁸–10⁹x compression through problem-directed extraction.</p>

              <div className="mt-8 space-y-4">
                <div className="card">
                  <div className="text-sm font-mono text-primary mb-1">INFORMATION MINIMALITY THEOREM</div>
                  <p className="text-muted text-sm">For any research question Q and dataset D, the extracted representation σ is a sufficient statistic with information content bounded by the mutual information I(D; A_Q). The raw data H(D) is never accessed beyond this bound.</p>
                </div>
                <div className="card">
                  <div className="text-sm font-mono text-primary mb-1">TYPE SAFETY</div>
                  <p className="text-muted text-sm">The protocol type system enforces dimensional consistency, conservation compliance, modality compatibility, and confidence monotonicity—all checked at compile time before any data is accessed.</p>
                </div>
                <div className="card">
                  <div className="text-sm font-mono text-primary mb-1">COMPILATION DECOMPOSITION</div>
                  <p className="text-muted text-sm">Any well-typed protocol decomposes into a sequence of atomic morphisms, each preserving S-entropy conservation. Complex analyses are compositions of simple, verified steps.</p>
                </div>
              </div>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
