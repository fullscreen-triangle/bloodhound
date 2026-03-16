import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const FundingPillar = ({ index, title, amount, description, outcomes, timeline }) => (
  <motion.div
    className="card-glow h-full"
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay: index * 0.1 }}
  >
    <div className="text-primary font-mono text-xs mb-2">PILLAR {index + 1}</div>
    <h3 className="text-xl font-bold mb-1">{title}</h3>
    <div className="text-accent font-bold text-lg mb-4">{amount}</div>
    <p className="text-muted text-sm mb-4 leading-relaxed">{description}</p>
    <div className="border-t border-primary/10 pt-4 mb-4">
      <div className="text-xs font-mono text-muted mb-2">DELIVERABLES</div>
      <ul className="space-y-1.5">
        {outcomes.map((o, i) => (
          <li key={i} className="text-sm text-muted flex items-start gap-2">
            <span className="text-primary mt-0.5">&#x2022;</span>{o}
          </li>
        ))}
      </ul>
    </div>
    <div className="text-xs text-muted font-mono">{timeline}</div>
  </motion.div>
);

const CollabTrack = ({ title, audience, description, action }) => (
  <motion.div
    className="card h-full"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <div className="text-accent text-xs font-mono mb-2 uppercase">{audience}</div>
    <h3 className="text-lg font-bold mb-2">{title}</h3>
    <p className="text-muted text-sm mb-4 leading-relaxed">{description}</p>
    <div className="text-primary text-sm font-medium">{action}</div>
  </motion.div>
);

export default function Collaborate() {
  return (
    <>
      <Head><title>Collaborate | Bloodhound</title></Head>
      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Partnership</div>
            <h1 className="section-heading">Collaborate</h1>
            <p className="section-subheading mb-16">Bloodhound is not a product—it is a research framework seeking the right partners to bring automated deep research from theory to practice. The funding structure mirrors the framework itself: multiple pillars, each self-contained, each strengthening the whole.</p>
          </motion.div>

          {/* Funding Pillars */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Funding Pillars</h2>
            <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
              <FundingPillar
                index={0}
                title="Core Engine"
                amount="Foundation"
                description="Build the Bloodhound VM runtime: the St-Hurbert execution engine, Triangle DSL compiler, ternary categorical memory, and Maxwell demon controller. This is the computational substrate everything else runs on."
                outcomes={[
                  "Production Triangle compiler with type checker",
                  "Ternary memory system with categorical addressing",
                  "St-Hurbert engine with S-entropy conservation",
                  "Maxwell demon resource controller",
                  "Formal verification of core invariants in Lean 4",
                ]}
                timeline="12–18 months"
              />
              <FundingPillar
                index={1}
                title="Domain Compilers"
                amount="Expansion"
                description="Create domain-specific language model compilers for target domains: genomics, proteomics, clinical imaging, environmental monitoring. Each compiler enables surgical extraction from a new modality."
                outcomes={[
                  "Genomics compiler (variant calling, GWAS, expression)",
                  "Proteomics compiler (structure, function, interactions)",
                  "Clinical imaging compiler (radiology, pathology)",
                  "Knowledge distillation pipeline (teacher → student)",
                  "Observe bridge architecture for new modalities",
                ]}
                timeline="18–24 months"
              />
              <FundingPillar
                index={2}
                title="Federated Network"
                amount="Scale"
                description="Deploy the federated understanding protocol across institutional networks. Structural privacy enables multi-institution collaboration without data sharing agreements for irrelevant data."
                outcomes={[
                  "Multi-node coordination protocol",
                  "Structural privacy implementation",
                  "Cross-institutional validation framework",
                  "Variance restoration for distributed convergence",
                  "Pilot deployment across 3–5 research institutions",
                ]}
                timeline="24–36 months"
              />
              <FundingPillar
                index={3}
                title="Metacognitive Intelligence"
                amount="Autonomy"
                description="Develop the full metacognitive pipeline: question decomposition, resource allocation, DPP candidate generation, multi-dimensional quality evaluation, and refinement orchestration."
                outcomes={[
                  "Question decomposition engine",
                  "Information-yield resource allocator",
                  "DPP-based diverse candidate generator",
                  "5-dimension quality evaluation framework",
                  "Autonomous refinement orchestrator",
                ]}
                timeline="12–24 months"
              />
              <FundingPillar
                index={4}
                title="Validation Infrastructure"
                amount="Trust"
                description="Build the multi-expert validation layer: domain-expert consensus models, adversarial quality assessment, and formal verification integration with proof assistants."
                outcomes={[
                  "Domain-expert consensus protocol",
                  "Adversarial evaluation framework",
                  "Lean 4 / Coq integration for formal proofs",
                  "Validation-entropy correspondence metrics",
                  "Reproducibility and provenance tracking",
                ]}
                timeline="12–18 months"
              />
              <FundingPillar
                index={5}
                title="Clinical Translation"
                amount="Impact"
                description="Apply the complete framework to real clinical research problems: multi-omics disease characterization, pharmacovigilance across regulatory jurisdictions, precision medicine investigations."
                outcomes={[
                  "End-to-end clinical research automation",
                  "Multi-omics disease characterization pipeline",
                  "Cross-jurisdictional pharmacovigilance demo",
                  "Precision medicine case studies",
                  "Regulatory compliance documentation",
                ]}
                timeline="24–36 months"
              />
            </div>
          </div>

          {/* Collaboration Tracks */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Collaboration Tracks</h2>
            <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
              <CollabTrack
                title="Research Partnership"
                audience="Academic Institutions"
                description="Joint research on automated deep research methodology. Contribute domain expertise, data access, or theoretical advances. Co-authorship on publications."
                action="Ideal for: Universities, Research Institutes, National Labs"
              />
              <CollabTrack
                title="Domain Compiler Development"
                audience="Domain Experts"
                description="Help build domain-specific compilers for your field. If you have deep expertise in a scientific domain and curated training data, you can enable surgical extraction for your entire community."
                action="Ideal for: Bioinformaticians, Clinicians, Environmental Scientists"
              />
              <CollabTrack
                title="Infrastructure Partnership"
                audience="Technology Partners"
                description="Contribute to the distributed runtime, networking protocol, or formal verification infrastructure. The framework needs high-performance distributed systems expertise."
                action="Ideal for: Cloud Providers, HPC Centers, Systems Engineers"
              />
              <CollabTrack
                title="Pilot Deployment"
                audience="Clinical Networks"
                description="Deploy federated understanding across your institutional network. Be among the first to demonstrate automated multi-institutional research without centralized data sharing."
                action="Ideal for: Hospital Networks, Clinical Research Organizations"
              />
              <CollabTrack
                title="Strategic Investment"
                audience="Funders & Investors"
                description="Fund one or more pillars of the framework development. Each pillar is independently valuable while contributing to the whole. Clear deliverables, measurable milestones."
                action="Ideal for: Research Foundations, VCs, Government Agencies"
              />
              <CollabTrack
                title="Open Source Contribution"
                audience="Developers"
                description="The core framework will be open source. Contribute to the Triangle compiler, domain compilers, validation infrastructure, or documentation."
                action="Ideal for: Systems Programmers (Rust), ML Engineers, Formal Methods"
              />
            </div>
          </div>

          {/* CTA */}
          <div className="text-center border-t border-primary/10 pt-16">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2 className="text-3xl font-bold mb-4">Ready to build the future of research?</h2>
              <p className="text-muted text-lg mb-8 max-w-xl mx-auto">
                Whether you bring domain expertise, engineering capability, institutional access, or funding—there is a place for you in this framework.
              </p>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link href="mailto:contact@bloodhound.dev" className="btn-primary">
                  Get In Touch
                </Link>
                <Link href="/validation" className="btn-outline">
                  See the Evidence
                </Link>
              </div>
            </motion.div>
          </div>
        </Layout>
      </section>
    </>
  );
}
