import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const DomainSection = ({ id, index, title, tagline, color, problem, approach, metrics, outcomes }) => (
  <motion.div
    id={id}
    className="mb-20"
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <div className="flex items-center gap-4 mb-6">
      <div className="w-10 h-10 rounded-lg flex items-center justify-center font-bold font-mono text-sm" style={{ backgroundColor: color + "20", color }}>
        {String(index).padStart(2, "0")}
      </div>
      <div>
        <h2 className="text-2xl font-bold">{title}</h2>
        <div className="text-muted text-sm">{tagline}</div>
      </div>
    </div>

    <div className="grid grid-cols-2 gap-8 lg:grid-cols-1">
      <div>
        <h3 className="font-bold mb-3 text-lg">The Problem</h3>
        <p className="text-muted text-sm leading-relaxed mb-6">{problem}</p>

        <h3 className="font-bold mb-3 text-lg">Bloodhound Approach</h3>
        <p className="text-muted text-sm leading-relaxed mb-4">{approach}</p>
      </div>

      <div>
        {metrics && (
          <div className="mb-6">
            <h3 className="font-bold mb-3 text-lg">Performance</h3>
            <div className="space-y-3">
              {metrics.map((m, i) => (
                <div key={i} className="flex items-center justify-between p-3 bg-surface rounded-xl border border-primary/10">
                  <span className="text-muted text-sm">{m.label}</span>
                  <span className="font-mono font-bold text-sm" style={{ color }}>{m.value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <h3 className="font-bold mb-3 text-lg">Key Outcomes</h3>
        <ul className="space-y-2">
          {outcomes.map((o, i) => (
            <li key={i} className="text-sm text-muted flex items-start gap-2">
              <span className="mt-0.5" style={{ color }}>&#x2022;</span>{o}
            </li>
          ))}
        </ul>
      </div>
    </div>
  </motion.div>
);

const DomainNav = ({ domains }) => (
  <motion.div
    className="flex flex-wrap gap-3 mb-16"
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.2 }}
  >
    {domains.map((d, i) => (
      <a
        key={i}
        href={`#${d.id}`}
        className="px-4 py-2 rounded-lg bg-surface border border-primary/10 text-sm font-medium text-muted hover:text-light hover:border-primary/30 transition-all"
      >
        {d.title}
      </a>
    ))}
  </motion.div>
);

const domains = [
  { id: "genomics", title: "Genomics" },
  { id: "metabolomics", title: "Metabolomics" },
  { id: "proteomics", title: "Proteomics" },
  { id: "pharma", title: "Pharmaceutical" },
  { id: "clinical", title: "Clinical Imaging" },
  { id: "environmental", title: "Environmental" },
];

export default function UseCases() {
  return (
    <>
      <Head>
        <title>Use Cases | Bloodhound</title>
        <meta name="description" content="Domain applications of the Bloodhound framework: genomics, metabolomics, proteomics, pharmaceutical research, clinical imaging, and environmental monitoring." />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Applications</div>
            <h1 className="section-heading">Use Cases</h1>
            <p className="section-subheading mb-8">
              The framework is domain-agnostic — any field where research questions can be specified against
              distributed data sources becomes a navigation problem in S-entropy space. Here are the domains
              where Bloodhound has been developed and validated.
            </p>
          </motion.div>

          <DomainNav domains={domains} />

          <DomainSection
            id="genomics"
            index={1}
            title="Genomics"
            tagline="Variant detection, pharmacogenetics, population genomics"
            color="#E63946"
            problem="Genomic analysis requires processing vast datasets — whole genome sequences at ~100 GB each, variant databases with millions of entries, population cohorts with thousands of individuals. Traditional pipelines load entire datasets, align reads, call variants, then filter. The computational cost scales with genome size, not with the research question."
            approach="The Mufakose Genomics compiler enables confirmation-based variant detection without explicit variant databases. Research questions compile into surgical extraction targets. Instead of loading the entire genome, the system navigates directly to the relevant loci — extracting only the variants, expression levels, and population frequencies relevant to the specific question."
            metrics={[
              { label: "Memory complexity", value: "O(N·V·L) → O(log(N·V))" },
              { label: "Data reduction", value: "~10⁸x" },
              { label: "Variant detection", value: "Confirmation-based" },
            ]}
            outcomes={[
              "Population genomics across millions of variants without loading full genomes",
              "Real-time variant interpretation for clinical decision support",
              "Multi-gene pharmacogenetic analysis for personalized medicine",
              "Alternative splicing space exploration for complex trait analysis",
              "Cross-cohort comparison without centralizing genomic data",
            ]}
          />

          <DomainSection
            id="metabolomics"
            index={2}
            title="Metabolomics"
            tagline="Mass spectrometry, molecular identification, pathway analysis"
            color="#F4A261"
            problem="Mass spectrometry-based metabolomics generates enormous spectral datasets — each sample produces thousands of spectra across multiple acquisition modes (MS1, MS2, retention time, ion mobility). Identifying metabolites requires searching against spectral libraries, performing molecular networking, and integrating pathway context. Traditional approaches process entire spectral datasets."
            approach="The Mufakose Metabolomics compiler integrates oscillatory molecular theory with confirmation-based processing. Molecular feature space is navigated in O(log N) computational complexity with constant memory. The system extracts only the spectral features relevant to the research question — specific m/z ranges, retention time windows, and fragmentation patterns."
            metrics={[
              { label: "True positive rate", value: "94.2% vs 87.3% traditional" },
              { label: "Computational complexity", value: "O(log N)" },
              { label: "Memory usage", value: "Constant" },
            ]}
            outcomes={[
              "Comprehensive molecular space coverage for complex biological samples",
              "Real-time metabolite identification during acquisition",
              "Pathway context integration without separate enrichment analysis",
              "Multi-modal data integration (MS1, MS2, RT, ion mobility) through S-entropy composition",
              "Environmental complexity optimization for enhanced signal-to-noise",
            ]}
          />

          <DomainSection
            id="proteomics"
            index={3}
            title="Proteomics"
            tagline="Protein structure, interactions, post-translational modifications"
            color="#2A9D8F"
            problem="Proteomics research involves analyzing protein expression, structure, interactions, and modifications across tissues, conditions, and time points. Datasets span mass spectrometry quantification, structural databases (PDB), interaction networks (STRING), and functional annotations. Integrating these modalities typically requires separate pipelines with manual harmonization."
            approach="Through the observe bridge architecture, each proteomic data modality maps to S-entropy space. Cross-modal composition is a built-in categorical operation — no separate ETL, no schema matching, no data harmonization. The research question specifies which proteins, modifications, or interactions are relevant, and the system navigates directly to them."
            metrics={[
              { label: "Cross-modal composition", value: "Built-in categorical" },
              { label: "Integration overhead", value: "Zero (no ETL)" },
              { label: "Modality support", value: "MS, structure, interactions" },
            ]}
            outcomes={[
              "Surgical extraction of specific protein targets (e.g., alpha-actinin-3 in cardiac muscle)",
              "Cross-modal links between gene variants and protein expression",
              "Tissue-specific protein characterization without loading full proteome databases",
              "Post-translational modification analysis in disease context",
              "Federated protein analysis across institutional boundaries",
            ]}
          />

          <DomainSection
            id="pharma"
            index={4}
            title="Pharmaceutical Research"
            tagline="Drug discovery, molecular identification, dose-response optimization"
            color="#A23B72"
            problem="Drug discovery requires systematic exploration of chemical space — millions of candidate molecules evaluated against multiple targets, ADMET properties, and safety profiles. Traditional high-throughput screening is exhaustive and expensive. Computational approaches (virtual screening, QSAR) still require processing large compound libraries."
            approach="The Mufakose Pharmaceutical compiler enables systematic pharmaceutical space coverage with O(log N) complexity. Membrane quantum computation with evidence rectification networks allows the system to navigate directly to promising regions of chemical space based on the therapeutic question, rather than exhaustively screening."
            metrics={[
              { label: "Space coverage", value: "Systematic O(log N)" },
              { label: "Evidence processing", value: "Fuzzy-Bayesian networks" },
              { label: "Optimization", value: "Therapeutic amplification" },
            ]}
            outcomes={[
              "Systematic pharmaceutical space coverage without exhaustive screening",
              "Drug discovery guided by research question, not library size",
              "Consciousness-based evidence processing for complex multi-target drugs",
              "Dose-response optimization through S-entropy trajectory navigation",
              "Cross-jurisdictional pharmacovigilance through federated understanding",
            ]}
          />

          <DomainSection
            id="clinical"
            index={5}
            title="Clinical Imaging"
            tagline="Radiology, pathology, multi-institutional analysis"
            color="#457B9D"
            problem="Clinical imaging generates massive datasets — a single CT scan is hundreds of megabytes, MRI sequences are gigabytes, and whole-slide pathology images exceed 1 GB each. Multi-institutional studies require sharing these images across networks, running into bandwidth, privacy, and regulatory constraints. HIPAA, GDPR, and institutional policies create significant barriers."
            approach="Federated understanding eliminates the need to move imaging data. Domain-specific compilers extract only the question-relevant features from each imaging modality — specific anatomical measurements, tissue characteristics, or pathological patterns. What traverses the network is understanding fragments: structured representations of the findings, not raw pixel data."
            metrics={[
              { label: "Network transfer", value: "Understanding fragments only" },
              { label: "Privacy model", value: "Structural (not differential)" },
              { label: "Regulatory compliance", value: "Data never leaves institution" },
            ]}
            outcomes={[
              "Multi-institutional imaging studies without data sharing agreements for raw images",
              "HIPAA/GDPR compliance by construction — irrelevant patient data never enters the computation",
              "Radiology-pathology correlation across institutions through S-entropy composition",
              "Real-time clinical decision support from distributed imaging archives",
              "Longitudinal imaging analysis without centralized data lakes",
            ]}
          />

          <DomainSection
            id="environmental"
            index={6}
            title="Environmental Monitoring"
            tagline="Sensor networks, climate data, ecological assessment"
            color="#F18F01"
            problem="Environmental monitoring involves distributed sensor networks generating continuous streams of data — air quality, water chemistry, soil composition, biodiversity surveys, satellite imagery. Centralizing this data for analysis is impractical: bandwidth is limited, sensors are geographically distributed, and data volumes grow continuously."
            approach="The network-gas correspondence maps naturally to environmental sensor networks. Each sensor node operates as a molecule in the thermodynamic model. Research questions about environmental conditions compile into navigation targets, and the system extracts relevant measurements from distributed sensors without centralizing raw data streams."
            metrics={[
              { label: "Coordination model", value: "Statistical (O(1) scaling)" },
              { label: "Sensor integration", value: "Network-gas correspondence" },
              { label: "Data centralization", value: "Not required" },
            ]}
            outcomes={[
              "Real-time environmental assessment from distributed sensor networks",
              "Cross-modal integration of heterogeneous environmental data sources",
              "Ecological impact analysis without centralizing raw sensor streams",
              "Climate trend detection through trajectory completion in S-entropy space",
              "Scalable monitoring — O(1) coordination regardless of sensor count",
            ]}
          />

          {/* CTA */}
          <div className="text-center border-t border-primary/10 pt-16">
            <motion.div initial={{ opacity: 0, y: 20 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
              <h2 className="text-3xl font-bold mb-4 md:text-2xl">Your domain could be next.</h2>
              <p className="text-muted text-lg mb-8 max-w-xl mx-auto">
                Any field where research questions target distributed data sources is a candidate for
                Bloodhound. The framework needs domain experts to build new compilers.
              </p>
              <div className="flex gap-4 justify-center flex-wrap">
                <Link href="/collaborate" className="btn-primary">Build a Domain Compiler</Link>
                <Link href="/docs" className="btn-outline">Read the Docs</Link>
              </div>
            </motion.div>
          </div>
        </Layout>
      </section>
    </>
  );
}
