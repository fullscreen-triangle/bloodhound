import Head from "next/head";
import Link from "next/link";
import { motion } from "framer-motion";
import Layout from "@/components/Layout";
import dynamic from "next/dynamic";

const SailfishModel = dynamic(() => import("@/components/SailfishModel"), { ssr: false });

const StatBlock = ({ value, label }) => (
  <motion.div className="text-center" initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.6 }}>
    <div className="text-4xl font-bold text-primary md:text-3xl">{value}</div>
    <div className="text-muted text-sm mt-1">{label}</div>
  </motion.div>
);

const PillarCard = ({ href, title, description, index }) => (
  <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.5, delay: index * 0.1 }}>
    <Link href={href}>
      <div className="card-glow h-full group cursor-pointer">
        <div className="text-primary/40 text-sm font-mono mb-2">0{index + 1}</div>
        <h3 className="text-xl font-bold mb-3 group-hover:text-primary transition-colors">{title}</h3>
        <p className="text-muted text-sm leading-relaxed">{description}</p>
        <div className="mt-4 text-primary text-sm font-medium opacity-0 group-hover:opacity-100 transition-opacity">Explore &rarr;</div>
      </div>
    </Link>
  </motion.div>
);

export default function Home() {
  return (
    <>
      <Head>
        <title>Bloodhound | Automated Deep Research</title>
        <meta name="description" content="Bloodhound: A distributed virtual machine framework for automated deep research through problem-directed trajectory completion in domain-specific language model networks." />
      </Head>

      <section className="relative min-h-[90vh] flex items-center overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-30" />
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="!pt-0 relative z-10">
          <div className="flex w-full items-center justify-between md:flex-col gap-12">
            <div className="w-1/2 lg:w-full">
              <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
                <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Distributed Virtual Machine Framework</div>
                <h1 className="text-6xl font-bold leading-tight mb-6 xl:text-5xl md:text-4xl sm:text-3xl">
                  You write the question.<br /><span className="text-primary">The system finds the answer.</span>
                </h1>
                <p className="text-muted text-lg mb-8 max-w-xl md:text-base">
                  Bloodhound is a framework for automated deep research. Domain-specific language models surgically extract only question-relevant information from distributed data sources. 968 bytes instead of 218.9 GB. No data movement. No manual analysis. Structural privacy by construction.
                </p>
                <div className="flex gap-4 flex-wrap">
                  <Link href="/federated" className="btn-primary">Explore the Framework</Link>
                  <Link href="/collaborate" className="btn-outline">Collaborate</Link>
                </div>
              </motion.div>
            </div>
            <motion.div className="w-1/2 lg:w-full h-[500px] md:h-[350px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.2, delay: 0.3 }}>
              <SailfishModel />
            </motion.div>
          </div>
        </Layout>
      </section>

      <section className="border-y border-primary/10 bg-surface/50">
        <Layout className="!py-12">
          <div className="grid grid-cols-4 gap-8 md:grid-cols-2 sm:grid-cols-1">
            <StatBlock value="10⁸x" label="Data Reduction Factor" />
            <StatBlock value="968 B" label="Network Transfer" />
            <StatBlock value="218.9 GB" label="Available Data Untouched" />
            <StatBlock value="7/7" label="Validation Checks Passed" />
          </div>
        </Layout>
      </section>

      <section>
        <Layout>
          <motion.div className="text-center mb-16" initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }}>
            <h2 className="section-heading">The Framework</h2>
            <p className="section-subheading mx-auto">Five interconnected pillars form a complete system for automated scientific investigation.</p>
          </motion.div>
          <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
            <PillarCard href="/phase-space" index={0} title="Bounded Phase Space" description="Everything lives in [0,1]³. S-entropy coordinates, ternary categorical memory, and the triple equivalence theorem provide mathematical guarantees—not heuristics." />
            <PillarCard href="/compilation" index={1} title="Problem-Directed Compilation" description="Research questions compile into morphism chains that surgically extract only what matters. This is where the 10⁸x compression ratio comes from." />
            <PillarCard href="/federated" index={2} title="Federated Understanding" description="Not federated learning. Not centralized. What traverses the network is question-shaped understanding—968 bytes instead of 218.9 GB. Structural privacy for free." />
            <PillarCard href="/pipeline" index={3} title="Metacognitive Pipeline" description="Six stages replace a human researcher: decompose, allocate, generate, evaluate, verify, orchestrate. Gas → liquid → crystal convergence with formal guarantees." />
            <PillarCard href="/validation" index={4} title="Empirical Validation" description="Real APIs. Real data. Real results. ACTN3 cardiac adaptation across genomics, transcriptomics, and proteomics—every claim backed by evidence." />
          </div>
        </Layout>
      </section>

      <section className="border-t border-primary/10">
        <Layout>
          <div className="max-w-3xl mx-auto text-center">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2 className="text-3xl font-bold mb-6 md:text-2xl">
                The research question <span className="text-primary">is</span> the scalpel.<br />
                Understanding <span className="text-primary">is</span> the currency.<br />
                The answer <span className="text-primary">crystallizes</span>.
              </h2>
              <p className="text-muted text-lg mb-8">In conventional computation, data exists independently of questions. In federated understanding, the question creates the data representation. Without a question, no representation exists.</p>
              <Link href="/collaborate" className="btn-primary">Partner With Us</Link>
            </motion.div>
          </div>
        </Layout>
      </section>
    </>
  );
}
