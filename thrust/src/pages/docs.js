import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const CodeBlock = ({ title, language, children }) => (
  <div className="mb-6">
    {title && <div className="text-sm font-bold mb-2">{title}</div>}
    <div className="bg-surface rounded-xl p-6 border border-primary/10 font-mono text-sm overflow-x-auto">
      <pre className="text-muted">{children}</pre>
    </div>
  </div>
);

const SideNavItem = ({ href, title, active }) => (
  <a
    href={href}
    className={`block py-1.5 px-3 rounded-lg text-sm transition-colors ${
      active ? "text-primary bg-primary/10" : "text-muted hover:text-light"
    }`}
  >
    {title}
  </a>
);

const ApiEntry = ({ name, signature, description }) => (
  <div className="py-4 border-b border-primary/5 last:border-0">
    <div className="font-mono text-sm font-bold text-primary mb-1">{name}</div>
    <div className="font-mono text-xs text-muted mb-2">{signature}</div>
    <p className="text-muted text-sm">{description}</p>
  </div>
);

export default function Docs() {
  return (
    <>
      <Head>
        <title>Documentation | Bloodhound</title>
        <meta name="description" content="Getting started with Bloodhound: installation, Triangle DSL reference, Python API, configuration, and project structure." />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Documentation</div>
            <h1 className="section-heading">Getting Started</h1>
            <p className="section-subheading mb-16">
              Everything you need to install, configure, and write your first research protocol
              with the Bloodhound framework.
            </p>
          </motion.div>

          <div className="grid grid-cols-[240px_1fr] gap-12 xl:grid-cols-1">
            {/* Side Navigation */}
            <div className="xl:hidden">
              <div className="sticky top-24 space-y-1">
                <div className="text-xs font-mono text-muted mb-3 uppercase">On this page</div>
                <SideNavItem href="#installation" title="Installation" />
                <SideNavItem href="#project-structure" title="Project Structure" />
                <SideNavItem href="#triangle-dsl" title="Triangle DSL" />
                <SideNavItem href="#python-api" title="Python API" />
                <SideNavItem href="#configuration" title="Configuration" />
                <SideNavItem href="#validation" title="Running Validation" />
              </div>
            </div>

            {/* Main Content */}
            <div className="max-w-3xl">
              {/* Installation */}
              <section id="installation" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Installation</h2>
                <CodeBlock title="Clone and install">
                  <span className="text-muted"># Clone the repository</span>{"\n"}
                  <span className="text-primary">git clone</span> https://github.com/bloodhound-framework/bloodhound.git{"\n"}
                  <span className="text-primary">cd</span> bloodhound{"\n\n"}
                  <span className="text-muted"># Install Python dependencies</span>{"\n"}
                  <span className="text-primary">pip install</span> -e .{"\n\n"}
                  <span className="text-muted"># Build the Rust core (optional, for VM runtime)</span>{"\n"}
                  <span className="text-primary">cargo build</span> --release{"\n\n"}
                  <span className="text-muted"># Verify installation</span>{"\n"}
                  <span className="text-primary">python</span> -m st_hurbet.validation.run_validation
                </CodeBlock>

                <div className="card">
                  <div className="text-primary font-mono text-xs mb-2">REQUIREMENTS</div>
                  <ul className="space-y-1.5 text-sm text-muted">
                    <li className="flex items-start gap-2"><span className="text-primary">&#x2022;</span>Python 3.10+</li>
                    <li className="flex items-start gap-2"><span className="text-primary">&#x2022;</span>Rust 1.75+ (for VM core, optional)</li>
                    <li className="flex items-start gap-2"><span className="text-primary">&#x2022;</span>NumPy, SciPy, Pandas (installed automatically)</li>
                    <li className="flex items-start gap-2"><span className="text-primary">&#x2022;</span>PyTorch (for domain compilers, optional)</li>
                  </ul>
                </div>
              </section>

              {/* Project Structure */}
              <section id="project-structure" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Project Structure</h2>
                <CodeBlock>
                  {"bloodhound/\n"}
                  {"├── st_hurbet/                     "}  <span className="text-primary"># Core execution engine</span>{"\n"}
                  {"│   ├── validation/                "}  <span className="text-primary"># Validation & reference implementations</span>{"\n"}
                  {"│   │   ├── s_entropy.py           "}  <span className="text-muted"># S-entropy coordinate system</span>{"\n"}
                  {"│   │   ├── ternary.py             "}  <span className="text-muted"># Ternary representation & addressing</span>{"\n"}
                  {"│   │   ├── trajectory.py          "}  <span className="text-muted"># Trajectory navigation</span>{"\n"}
                  {"│   │   ├── categorical_memory.py  "}  <span className="text-muted"># Memory hierarchy</span>{"\n"}
                  {"│   │   ├── maxwell_demon.py       "}  <span className="text-muted"># Zero-cost sorting controller</span>{"\n"}
                  {"│   │   ├── distributed.py         "}  <span className="text-muted"># Network coordination layer</span>{"\n"}
                  {"│   │   ├── enhancement.py         "}  <span className="text-muted"># Temporal precision enhancement</span>{"\n"}
                  {"│   │   └── run_validation.py      "}  <span className="text-muted"># Full validation suite</span>{"\n"}
                  {"│   ├── docs/                      "}  <span className="text-primary"># Research papers & documentation</span>{"\n"}
                  {"│   └── publication/               "}  <span className="text-primary"># Publication materials</span>{"\n"}
                  {"│\n"}
                  {"├── src/\n"}
                  {"│   └── bloodhound_vm_core/        "}  <span className="text-primary"># Rust VM runtime</span>{"\n"}
                  {"│       ├── consciousness.rs       "}  <span className="text-muted"># Consciousness-aware processing</span>{"\n"}
                  {"│       ├── entropy.rs             "}  <span className="text-muted"># S-entropy implementation</span>{"\n"}
                  {"│       ├── oscillatory.rs         "}  <span className="text-muted"># Oscillatory dynamics</span>{"\n"}
                  {"│       └── runtime.rs             "}  <span className="text-muted"># VM runtime loop</span>{"\n"}
                  {"│\n"}
                  {"├── backend/                       "}  <span className="text-primary"># Python backend & APIs</span>{"\n"}
                  {"├── frontend/                      "}  <span className="text-primary"># React visualization frontend</span>{"\n"}
                  {"├── docs/                          "}  <span className="text-primary"># Extended documentation</span>{"\n"}
                  {"├── bloodhound.toml                "}  <span className="text-primary"># Main configuration</span>{"\n"}
                  {"├── Cargo.toml                     "}  <span className="text-primary"># Rust workspace</span>{"\n"}
                  {"└── pyproject.toml                 "}  <span className="text-primary"># Python project config</span>
                </CodeBlock>
              </section>

              {/* Triangle DSL */}
              <section id="triangle-dsl" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Triangle DSL Reference</h2>
                <p className="text-muted mb-6 leading-relaxed">
                  Triangle is the domain-specific language for specifying research protocols. Each statement
                  maps to a morphism chain through S-entropy space. The language is designed around navigation,
                  not computation — you specify what to investigate and what evidence constitutes convergence.
                </p>

                <CodeBlock title="Coordinate Literals">
                  <span className="text-primary">S</span>(0.5, 0.3, 0.2){"          "}<span className="text-muted"># Direct S-entropy coordinate</span>{"\n"}
                  <span className="text-primary">S</span>.012.201.100{"           "}<span className="text-muted"># Trit address (depth 9)</span>
                </CodeBlock>

                <CodeBlock title="Surgical Extraction (slice)">
                  <span className="text-muted"># Extract specific data from a source</span>{"\n"}
                  genotype = <span className="text-accent">slice</span> genomics.ACTN3{"\n"}
                  {"  "}<span className="text-primary">@</span> cohort(elite_sprinters){"\n"}
                  {"  "}<span className="text-primary">@</span> variant(rs1815739){"\n\n"}
                  <span className="text-muted"># Mass spectrometry extraction</span>{"\n"}
                  spectrum = <span className="text-accent">slice</span> metabolomics{"\n"}
                  {"  "}<span className="text-primary">@</span> mz(400..600){"\n"}
                  {"  "}<span className="text-primary">@</span> rt(12.5..13.2)
                </CodeBlock>

                <CodeBlock title="Composition (compose)">
                  <span className="text-muted"># Compose two understanding fragments</span>{"\n"}
                  joined = <span className="text-accent">compose</span> genotype <span className="text-primary">with</span> cardiac{"\n"}
                  {"  "}<span className="text-primary">preserving</span> athlete_id{"\n\n"}
                  <span className="text-muted"># Multi-source composition</span>{"\n"}
                  integrated = <span className="text-accent">compose</span> genomics <span className="text-primary">with</span> proteomics{"\n"}
                  {"  "}<span className="text-primary">with</span> transcriptomics{"\n"}
                  {"  "}<span className="text-primary">preserving</span> gene_id
                </CodeBlock>

                <CodeBlock title="Navigation & Completion">
                  <span className="text-muted"># Navigate to target with completion condition</span>{"\n"}
                  result = <span className="text-accent">navigate</span> joined <span className="text-primary">to</span> target{"\n"}
                  {"  "}<span className="text-primary">via</span> correlation_analysis{"\n\n"}
                  <span className="text-muted"># Completion conditions</span>{"\n"}
                  <span className="text-primary">complete when</span> distance {"<"} epsilon{"\n"}
                  <span className="text-primary">complete at</span> depth 12{"\n"}
                  <span className="text-primary">complete when</span> confidence {">"} 0.95{"\n"}
                  <span className="text-primary">converge at</span> confidence {">"} 0.95
                </CodeBlock>

                <CodeBlock title="Parallel Execution">
                  <span className="text-muted"># Extract from multiple sources simultaneously</span>{"\n"}
                  <span className="text-primary">parallel</span> {"{"}{"\n"}
                  {"  "}hrv = <span className="text-accent">slice</span> biometrics.hrv{"\n"}
                  {"    "}<span className="text-primary">@</span> cohort(elite_sprinters){"\n\n"}
                  {"  "}genes = <span className="text-accent">slice</span> genomics.ACTN3{"\n"}
                  {"    "}<span className="text-primary">@</span> cohort(elite_sprinters){"\n"}
                  {"}"}
                </CodeBlock>

                <CodeBlock title="Complete Example: ACTN3 Cardiac Adaptation Study">
                  <span className="text-primary">investigate</span> &quot;Association between ACTN3{"\n"}
                  {"  "}genotype and cardiac adaptation{"\n"}
                  {"  "}in elite sprinters&quot;{"\n"}
                  {"  "}<span className="text-primary">with</span> confidence {">"} 0.95{"\n"}
                  {"  "}<span className="text-primary">with</span> significance {"<"} 0.01{"\n\n"}
                  <span className="text-primary">parallel</span> {"{"}{"\n"}
                  {"  "}genotype = <span className="text-accent">slice</span> genomics.ACTN3{"\n"}
                  {"    "}<span className="text-primary">@</span> cohort(elite_sprinters){"\n"}
                  {"    "}<span className="text-primary">@</span> variant(rs1815739){"\n\n"}
                  {"  "}cardiac = <span className="text-accent">slice</span> echocardiography{"\n"}
                  {"    "}<span className="text-primary">@</span> cohort(elite_sprinters){"\n"}
                  {"    "}<span className="text-primary">@</span> measure(LV_mass, EF, GLS){"\n\n"}
                  {"  "}protein = <span className="text-accent">slice</span> proteomics{"\n"}
                  {"    "}<span className="text-primary">@</span> target(alpha_actinin_3){"\n"}
                  {"    "}<span className="text-primary">@</span> tissue(cardiac_muscle){"\n"}
                  {"}"}{"\n\n"}
                  joined = <span className="text-accent">compose</span> genotype <span className="text-primary">with</span> cardiac{"\n"}
                  {"  "}<span className="text-primary">preserving</span> athlete_id{"\n\n"}
                  result = <span className="text-accent">navigate</span> joined <span className="text-primary">to</span> target{"\n"}
                  {"  "}<span className="text-primary">via</span> correlation_analysis{"\n\n"}
                  <span className="text-primary">converge at</span> confidence {">"} 0.95
                </CodeBlock>
              </section>

              {/* Python API */}
              <section id="python-api" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Python API</h2>

                <CodeBlock title="S-Entropy Coordinates">
                  <span className="text-primary">from</span> st_hurbet.validation.s_entropy <span className="text-primary">import</span> SCoordinate, SEntropyCore{"\n\n"}
                  <span className="text-muted"># Create coordinates in bounded [0,1]³ space</span>{"\n"}
                  start = <span className="text-accent">SCoordinate</span>(s_k=0.1, s_t=0.2, s_e=0.3){"\n"}
                  target = <span className="text-accent">SCoordinate</span>(s_k=0.8, s_t=0.7, s_e=0.9){"\n\n"}
                  <span className="text-muted"># Calculate categorical distance</span>{"\n"}
                  core = <span className="text-accent">SEntropyCore</span>(){"\n"}
                  d = core.categorical_distance(start, target){"\n"}
                  <span className="text-primary">print</span>(f&quot;Categorical distance: {"{d}"}&quot;)
                </CodeBlock>

                <CodeBlock title="Trajectory Navigation">
                  <span className="text-primary">from</span> st_hurbet.validation.trajectory <span className="text-primary">import</span> TrajectoryNavigator{"\n\n"}
                  <span className="text-muted"># Navigate from start to target</span>{"\n"}
                  navigator = <span className="text-accent">TrajectoryNavigator</span>(epsilon=1e-3){"\n"}
                  trajectory = navigator.navigate(start, target){"\n\n"}
                  <span className="text-muted"># The trajectory IS the address</span>{"\n"}
                  <span className="text-primary">print</span>(f&quot;Address: {"{trajectory.address}"}&quot;){"\n"}
                  <span className="text-primary">print</span>(f&quot;Path length: {"{trajectory.length()}"}&quot;)
                </CodeBlock>

                <CodeBlock title="Categorical Memory">
                  <span className="text-primary">from</span> st_hurbet.validation.categorical_memory <span className="text-primary">import</span> CategoricalMemory{"\n\n"}
                  <span className="text-muted"># Create hierarchical memory</span>{"\n"}
                  memory = <span className="text-accent">CategoricalMemory</span>(depth=6){"\n\n"}
                  <span className="text-muted"># Store at S-entropy coordinate</span>{"\n"}
                  memory.store(coord, data){"\n\n"}
                  <span className="text-muted"># Retrieve — tier determined by categorical distance</span>{"\n"}
                  result = memory.retrieve(query_coord)
                </CodeBlock>

                <h3 className="font-bold mb-4 mt-8">Core API Reference</h3>
                <div className="card">
                  <ApiEntry
                    name="SCoordinate"
                    signature="SCoordinate(s_k: float, s_t: float, s_e: float)"
                    description="A point in bounded [0,1]³ S-entropy space. All three coordinates must be in [0, 1]. Represents the entropy state of an information fragment."
                  />
                  <ApiEntry
                    name="SEntropyCore.categorical_distance"
                    signature="categorical_distance(a: SCoordinate, b: SCoordinate) → float"
                    description="Compute the categorical distance between two S-coordinates. Independent of Euclidean distance. Used for memory tier assignment and trajectory completion detection."
                  />
                  <ApiEntry
                    name="TrajectoryNavigator.navigate"
                    signature="navigate(start: SCoordinate, target: SCoordinate) → Trajectory"
                    description="Navigate from start to target through S-entropy space. Returns a Trajectory object encoding the path, which simultaneously serves as the address and result identifier."
                  />
                  <ApiEntry
                    name="CategoricalMemory.store"
                    signature="store(coord: SCoordinate, data: Any) → TritAddress"
                    description="Store data at an S-entropy coordinate. Automatically assigns to the correct memory tier based on categorical distance. Returns the ternary address."
                  />
                  <ApiEntry
                    name="TernaryEncoder.encode"
                    signature="encode(coord: SCoordinate, depth: int) → TritAddress"
                    description="Encode an S-coordinate as a ternary address at the specified depth. Bijective mapping: each address maps to exactly one cell in the 3^k partition."
                  />
                </div>
              </section>

              {/* Configuration */}
              <section id="configuration" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Configuration</h2>
                <p className="text-muted mb-6 leading-relaxed">
                  The main configuration file is <span className="font-mono text-light">bloodhound.toml</span> in the
                  project root. Key sections:
                </p>

                <CodeBlock title="S-Entropy Navigation">
                  <span className="text-muted">[s_entropy]</span>{"\n"}
                  enable_navigation = <span className="text-primary">true</span>{"\n"}
                  coordinate_precision = <span className="text-accent">1e-15</span>{"\n"}
                  endpoint_prediction = <span className="text-primary">true</span>{"\n"}
                  zero_time_computation = <span className="text-primary">true</span>{"\n"}
                  knowledge_dimension_weight = <span className="text-accent">0.4</span>{"\n"}
                  time_dimension_weight = <span className="text-accent">0.3</span>{"\n"}
                  entropy_dimension_weight = <span className="text-accent">0.3</span>
                </CodeBlock>

                <CodeBlock title="Consciousness Processing">
                  <span className="text-muted">[consciousness]</span>{"\n"}
                  bmd_frame_selection = <span className="text-primary">true</span>{"\n"}
                  semantic_understanding = <span className="text-primary">true</span>{"\n"}
                  recursive_self_awareness = <span className="text-primary">true</span>{"\n"}
                  consciousness_loops = <span className="text-primary">true</span>
                </CodeBlock>

                <CodeBlock title="Purpose Framework (Domain-Specific Learning)">
                  <span className="text-muted">[purpose_framework]</span>{"\n"}
                  enable_framework = <span className="text-primary">true</span>{"\n"}
                  domain_learning = <span className="text-primary">true</span>{"\n"}
                  enhanced_distillation = <span className="text-primary">true</span>{"\n"}
                  adaptation_precision = <span className="text-accent">1e-12</span>{"\n"}
                  information_density_target = <span className="text-accent">2.5</span>
                </CodeBlock>

                <CodeBlock title="Combine Harvester (Knowledge Integration)">
                  <span className="text-muted">[combine_harvester]</span>{"\n"}
                  enable_framework = <span className="text-primary">true</span>{"\n"}
                  multi_domain_integration = <span className="text-primary">true</span>{"\n"}
                  router_algorithms = <span className="text-accent">[&quot;keyword&quot;, &quot;embedding&quot;, &quot;classifier&quot;, &quot;llm&quot;]</span>{"\n"}
                  optimal_routing = <span className="text-accent">&quot;embedding_based&quot;</span>
                </CodeBlock>
              </section>

              {/* Running Validation */}
              <section id="validation" className="mb-16">
                <h2 className="text-2xl font-bold mb-6">Running Validation</h2>
                <CodeBlock title="Run the full validation suite">
                  <span className="text-primary">python</span> -m st_hurbet.validation.run_validation{"\n\n"}
                  <span className="text-muted"># Expected output: 10 theorems verified</span>{"\n"}
                  <span className="text-muted"># ✓ Triple Equivalence</span>{"\n"}
                  <span className="text-muted"># ✓ Trit-Cell Correspondence</span>{"\n"}
                  <span className="text-muted"># ✓ Trajectory-Position Identity</span>{"\n"}
                  <span className="text-muted"># ✓ Completion Equivalence</span>{"\n"}
                  <span className="text-muted"># ✓ Zero-Cost Sorting</span>{"\n"}
                  <span className="text-muted"># ✓ Observable Commutation</span>{"\n"}
                  <span className="text-muted"># ✓ Exponential Decay</span>{"\n"}
                  <span className="text-muted"># ✓ Central State Impossibility</span>{"\n"}
                  <span className="text-muted"># ✓ Distance Independence</span>{"\n"}
                  <span className="text-muted"># ✓ Continuous Emergence</span>
                </CodeBlock>

                <div className="card">
                  <div className="text-primary font-mono text-xs mb-2">NOTE</div>
                  <p className="text-muted text-sm">
                    The validation suite tests all core theorems against the reference Python implementation.
                    Each test is deterministic and self-contained. No external data or API access is required for
                    the theorem validation. The ACTN3 end-to-end validation (on the{" "}
                    <Link href="/validation" className="text-primary hover:underline">Validation page</Link>)
                    queries live public APIs.
                  </p>
                </div>
              </section>

              {/* Next Steps */}
              <div className="border-t border-primary/10 pt-12">
                <h2 className="text-2xl font-bold mb-6">Next Steps</h2>
                <div className="grid grid-cols-2 gap-4 md:grid-cols-1">
                  <Link href="/architecture" className="card group cursor-pointer">
                    <h3 className="font-bold mb-2 group-hover:text-primary transition-colors">Architecture Deep-Dive</h3>
                    <p className="text-muted text-sm">Understand the three-layer system, S-entropy coordinates, and distributed coordination.</p>
                  </Link>
                  <Link href="/use-cases" className="card group cursor-pointer">
                    <h3 className="font-bold mb-2 group-hover:text-primary transition-colors">Use Cases</h3>
                    <p className="text-muted text-sm">See how the framework applies to genomics, metabolomics, clinical imaging, and more.</p>
                  </Link>
                  <Link href="/validation" className="card group cursor-pointer">
                    <h3 className="font-bold mb-2 group-hover:text-primary transition-colors">Validation Results</h3>
                    <p className="text-muted text-sm">Review the empirical evidence: 7/7 checks passed on the ACTN3 multi-omics study.</p>
                  </Link>
                  <Link href="/collaborate" className="card group cursor-pointer">
                    <h3 className="font-bold mb-2 group-hover:text-primary transition-colors">Collaborate</h3>
                    <p className="text-muted text-sm">Find the right partnership track: research, domain compilers, infrastructure, or funding.</p>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
