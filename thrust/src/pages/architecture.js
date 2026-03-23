import React from "react";
import Head from "next/head";
import Layout from "@/components/Layout";
import { motion } from "framer-motion";
import Link from "next/link";

const LayerCard = ({ number, title, subtitle, children, color }) => (
  <motion.div
    className="card h-full border-t-4"
    style={{ borderTopColor: color }}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ delay: number * 0.15 }}
  >
    <div className="font-mono text-xs mb-2" style={{ color }}>LAYER {number}</div>
    <h3 className="text-xl font-bold mb-1">{title}</h3>
    <div className="text-muted text-sm mb-4">{subtitle}</div>
    {children}
  </motion.div>
);

const MemoryTier = ({ name, distance, latency, color }) => (
  <div className="flex items-center justify-between py-2 border-b border-primary/5 last:border-0">
    <div className="flex items-center gap-3">
      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      <span className="font-mono text-sm">{name}</span>
    </div>
    <div className="text-right">
      <span className="text-muted text-xs font-mono">{distance}</span>
      <span className="text-light text-sm font-mono ml-4">{latency}</span>
    </div>
  </div>
);

const TheoremCard = ({ name, formula, description }) => (
  <motion.div
    className="card"
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
  >
    <div className="text-primary font-mono text-xs mb-2">{name}</div>
    {formula && (
      <div className="bg-dark/50 rounded-lg p-3 font-mono text-sm mb-3 text-center">{formula}</div>
    )}
    <p className="text-muted text-sm leading-relaxed">{description}</p>
  </motion.div>
);

export default function Architecture() {
  return (
    <>
      <Head>
        <title>Architecture | Bloodhound</title>
        <meta name="description" content="The three-layer architecture of the Bloodhound distributed virtual machine: Triangle language, St-Hurbert engine, and distributed coordination." />
      </Head>

      <section className="relative">
        <div className="absolute inset-0 bg-radial-dark" />
        <Layout className="relative z-10">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <div className="text-primary font-mono text-sm mb-4 tracking-widest uppercase">Architecture</div>
            <h1 className="section-heading">Three-Layer System</h1>
            <p className="section-subheading mb-16">
              A domain-specific language compiles research questions into morphism chains.
              An execution engine navigates categorical space. A distributed coordination layer
              maps network physics to thermodynamic properties.
            </p>
          </motion.div>

          {/* Three Layers Overview */}
          <div className="grid grid-cols-3 gap-6 mb-20 lg:grid-cols-1">
            <LayerCard number={1} title="Triangle Language" subtitle="Research Protocol Specification" color="#E63946">
              <ul className="space-y-2 text-sm text-muted">
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>LL(1) grammar with dimensional type checking</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Navigation statements: <span className="font-mono text-light">navigate</span>, <span className="font-mono text-light">slice</span>, <span className="font-mono text-light">compose</span></li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Completion conditions with &epsilon;-boundary</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Parallel extraction blocks</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Compile-time conservation checking</li>
              </ul>
              <div className="mt-4">
                <Link href="/compilation" className="text-primary text-sm font-medium hover:underline">Deep dive &rarr;</Link>
              </div>
            </LayerCard>

            <LayerCard number={2} title="St-Hurbert Engine" subtitle="Categorical Execution Runtime" color="#F4A261">
              <ul className="space-y-2 text-sm text-muted">
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>S-Entropy Core: [0,1]&sup3; coordinate system</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Categorical Memory: 3<sup>k</sup> hierarchical addressing</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Maxwell Demon: zero-cost categorical sorting</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Trajectory Executor: &epsilon;-boundary completion</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Ternary representation (base-3 addressing)</li>
              </ul>
              <div className="mt-4">
                <Link href="/phase-space" className="text-primary text-sm font-medium hover:underline">Deep dive &rarr;</Link>
              </div>
            </LayerCard>

            <LayerCard number={3} title="Distributed Coordination" subtitle="Network-Thermodynamics Mapping" color="#2A9D8F">
              <ul className="space-y-2 text-sm text-muted">
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Network-Gas Correspondence</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Variance Restoration: &tau; &asymp; 0.5 ms</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Phase transitions: Gas &rarr; Liquid &rarr; Crystal</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>Central State Impossibility Theorem</li>
                <li className="flex items-start gap-2"><span className="text-primary mt-0.5">&#x2022;</span>O(1) coordination independent of network size</li>
              </ul>
              <div className="mt-4">
                <Link href="/federated" className="text-primary text-sm font-medium hover:underline">Deep dive &rarr;</Link>
              </div>
            </LayerCard>
          </div>

          {/* S-Entropy Coordinate System */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">S-Entropy Coordinate System</h2>
            <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
              <div>
                <p className="text-muted mb-6 leading-relaxed">
                  All information in the system is represented as a point in the unit cube <span className="font-mono text-light">S = [0,1]&sup3;</span>.
                  Three orthogonal entropy dimensions encode everything the system needs to know about a piece of information:
                </p>
                <div className="space-y-4 mb-8">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-[#E63946]/20 flex items-center justify-center shrink-0">
                      <span className="text-[#E63946] font-bold font-mono text-sm">S<sub>k</sub></span>
                    </div>
                    <div>
                      <div className="font-bold">Knowledge Entropy</div>
                      <div className="text-muted text-sm">Uncertainty in state identification. High = uncertain, low = crystallized knowledge. Measures how much is unknown about the content.</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-[#457B9D]/20 flex items-center justify-center shrink-0">
                      <span className="text-[#457B9D] font-bold font-mono text-sm">S<sub>t</sub></span>
                    </div>
                    <div>
                      <div className="font-bold">Temporal Entropy</div>
                      <div className="text-muted text-sm">Uncertainty in timing. When was this information generated? How current is it? Higher values indicate greater temporal uncertainty.</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-[#2A9D8F]/20 flex items-center justify-center shrink-0">
                      <span className="text-[#2A9D8F] font-bold font-mono text-sm">S<sub>e</sub></span>
                    </div>
                    <div>
                      <div className="font-bold">Evolution Entropy</div>
                      <div className="text-muted text-sm">Uncertainty in trajectory. How likely is this information to change? High for active research frontiers, low for established physical constants.</div>
                    </div>
                  </div>
                </div>
                <div className="card bg-darkTertiary border-primary/20">
                  <div className="text-primary font-mono text-xs mb-2">CONSERVATION LAW</div>
                  <div className="text-xl font-bold font-mono mb-2">S<sub>k</sub> + S<sub>t</sub> + S<sub>e</sub> = S<sub>total</sub></div>
                  <p className="text-muted text-sm">Total entropy is conserved through every morphism chain. Knowledge gained must come from temporal or evolution entropy reduced. Nothing is created or destroyed — only transformed.</p>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-bold mb-4">Categorical Distance</h3>
                <div className="card bg-darkTertiary border-primary/20 mb-6">
                  <div className="text-primary font-mono text-xs mb-2">DISTANCE FORMULA</div>
                  <div className="font-mono text-sm text-center py-2">
                    d<sub>cat</sub>(S&#x2081;, S&#x2082;) = &Sigma; |t<sub>i</sub><sup>(1)</sup> &minus; t<sub>i</sub><sup>(2)</sup>| / 3<sup>(i+1)</sup>
                  </div>
                  <p className="text-muted text-sm mt-2">Categorical distance is mathematically independent of Euclidean distance. Two points close in physical space can be far in categorical space, and vice versa.</p>
                </div>

                <h3 className="text-lg font-bold mb-4">Categorical Memory Hierarchy</h3>
                <div className="card">
                  <MemoryTier name="L1 CACHE" distance="d < 10⁻²³" latency="~1 ns" color="#2A9D8F" />
                  <MemoryTier name="L2 CACHE" distance="10⁻²³ ≤ d < 10⁻²²" latency="~10 ns" color="#457B9D" />
                  <MemoryTier name="L3 CACHE" distance="10⁻²² ≤ d < 10⁻²¹" latency="~50 ns" color="#F4A261" />
                  <MemoryTier name="RAM" distance="10⁻²¹ ≤ d < 10⁻²⁰" latency="~100 ns" color="#E63946" />
                  <MemoryTier name="STORAGE" distance="d ≥ 10⁻²⁰" latency="~1 ms" color="#A23B72" />
                </div>
                <p className="text-muted text-sm mt-3">Memory placement is determined by categorical distance, not access frequency. The 3<sup>k</sup> hierarchical structure is addressed by S-entropy coordinates through ternary encoding.</p>
              </div>
            </div>
          </div>

          {/* Network-Gas Correspondence */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Network-Gas Correspondence</h2>
            <div className="grid grid-cols-2 gap-12 lg:grid-cols-1">
              <div>
                <p className="text-muted mb-6 leading-relaxed">
                  The distributed coordination layer maps network properties to thermodynamic properties.
                  This is not a metaphor — it is a formal mathematical correspondence that enables
                  coordination through bulk statistical properties rather than individual node tracking.
                </p>
                <div className="bg-surface rounded-xl border border-primary/10 overflow-hidden">
                  <div className="grid grid-cols-2">
                    <div className="p-4 border-b border-r border-primary/10 font-bold text-sm">Network</div>
                    <div className="p-4 border-b border-primary/10 font-bold text-sm">Thermodynamics</div>
                    {[
                      ["Nodes", "Molecules"],
                      ["Addresses", "Positions"],
                      ["Queue depths", "Momenta"],
                      ["Packet exchange", "Collisions"],
                      ["Variance", "Temperature"],
                      ["Load", "Pressure"],
                    ].map(([net, thermo], i) => (
                      <React.Fragment key={i}>
                        <div className="p-3 border-b border-r border-primary/5 text-sm font-mono text-muted">{net}</div>
                        <div className="p-3 border-b border-primary/5 text-sm font-mono text-primary">{thermo}</div>
                      </React.Fragment>
                    ))}
                  </div>
                </div>
              </div>
              <div>
                <div className="space-y-4">
                  <div className="card border-l-4 border-l-[#E63946]">
                    <h3 className="font-bold mb-1">Gas Phase <span className="text-muted font-normal text-sm">&sigma;&sup2; &gt; 10&sup3;</span></h3>
                    <p className="text-muted text-sm">Nodes operate independently. High variance, no coordination. Each node processes requests in isolation.</p>
                  </div>
                  <div className="card border-l-4 border-l-[#F4A261]">
                    <h3 className="font-bold mb-1">Liquid Phase <span className="text-muted font-normal text-sm">10⁻⁶ &lt; &sigma;&sup2; &lt; 10⁻&sup3;</span></h3>
                    <p className="text-muted text-sm">Partial coordination. Nodes begin sharing understanding fragments. Cross-modal links form between domains.</p>
                  </div>
                  <div className="card border-l-4 border-l-[#2A9D8F]">
                    <h3 className="font-bold mb-1">Crystal Phase <span className="text-muted font-normal text-sm">&sigma;&sup2; &lt; 10⁻⁶</span></h3>
                    <p className="text-muted text-sm">Perfect synchronization. All nodes converge to consistent state. The answer has crystallized across the network.</p>
                  </div>
                </div>
                <div className="card bg-darkTertiary border-primary/20 mt-6">
                  <div className="text-primary font-mono text-xs mb-2">VARIANCE RESTORATION</div>
                  <div className="font-mono text-sm text-center py-2">
                    &sigma;&sup2;(t) = &sigma;&sup2;&#x2080; exp(-t/&tau;) &nbsp;&nbsp; &tau; &asymp; 0.5 ms
                  </div>
                  <p className="text-muted text-sm mt-2">Variance decays exponentially. The network naturally restores equilibrium without central coordination.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Maxwell Demon & Zero-Cost Sorting */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Maxwell Demon Controller</h2>
            <div className="max-w-3xl">
              <div className="card bg-darkTertiary border-primary/20 mb-6">
                <div className="text-primary font-mono text-xs mb-2">COMMUTATION RELATION</div>
                <div className="font-mono text-lg text-center py-2">
                  [&Ocirc;<sub>cat</sub>, &Ocirc;<sub>phys</sub>] = 0
                </div>
                <p className="text-muted text-sm mt-3">
                  Categorical observables commute with physical observables. This means categorical sorting operations
                  have zero thermodynamic cost — they don&apos;t disturb the physical state of the system.
                  This circumvents the Landauer limit: information can be organized in categorical space without
                  the k<sub>B</sub>T ln 2 energy cost per bit that applies to physical sorting.
                </p>
              </div>
              <p className="text-muted leading-relaxed">
                The Maxwell demon controller leverages this commutation to perform trajectory prediction and
                prefetching. It sorts information categorically — placing data in the right memory tier based on
                categorical distance — without thermodynamic penalty. This is not a violation of the second law;
                it is a consequence of categorical operations living in a different space than physical operations.
              </p>
            </div>
          </div>

          {/* Tech Stack */}
          <div className="mb-20">
            <h2 className="text-2xl font-bold mb-8">Technology Stack</h2>
            <div className="grid grid-cols-3 gap-6 lg:grid-cols-2 md:grid-cols-1">
              <div className="card">
                <div className="text-[#E63946] font-mono text-xs mb-2">RUST CORE</div>
                <h3 className="font-bold mb-3">bloodhound_vm_core</h3>
                <ul className="space-y-1.5 text-sm text-muted">
                  <li><span className="font-mono text-light">tokio 1.35</span> — async runtime</li>
                  <li><span className="font-mono text-light">nalgebra 0.32</span> — linear algebra</li>
                  <li><span className="font-mono text-light">candle</span> — ML inference</li>
                  <li><span className="font-mono text-light">libp2p 0.54</span> — peer-to-peer networking</li>
                  <li><span className="font-mono text-light">tonic 0.10</span> — gRPC</li>
                  <li><span className="font-mono text-light">polars 0.36</span> — dataframes</li>
                  <li><span className="font-mono text-light">ring 0.17</span> — cryptography</li>
                  <li><span className="font-mono text-light">redb 1.5</span> — embedded storage</li>
                </ul>
              </div>
              <div className="card">
                <div className="text-[#F4A261] font-mono text-xs mb-2">PYTHON BACKEND</div>
                <h3 className="font-bold mb-3">Validation &amp; Domain Compilers</h3>
                <ul className="space-y-1.5 text-sm text-muted">
                  <li><span className="font-mono text-light">numpy / scipy</span> — scientific computing</li>
                  <li><span className="font-mono text-light">torch</span> — deep learning</li>
                  <li><span className="font-mono text-light">transformers</span> — language models</li>
                  <li><span className="font-mono text-light">biopython</span> — bioinformatics</li>
                  <li><span className="font-mono text-light">scanpy</span> — single-cell analysis</li>
                  <li><span className="font-mono text-light">pyteomics</span> — mass spectrometry</li>
                  <li><span className="font-mono text-light">fastapi</span> — API server</li>
                  <li><span className="font-mono text-light">polars</span> — high-performance dataframes</li>
                </ul>
              </div>
              <div className="card">
                <div className="text-[#2A9D8F] font-mono text-xs mb-2">INTEGRATED SYSTEMS</div>
                <h3 className="font-bold mb-3">Advanced Modules</h3>
                <ul className="space-y-1.5 text-sm text-muted">
                  <li><span className="font-mono text-light">Kwasa-Kwasa</span> — consciousness interface</li>
                  <li><span className="font-mono text-light">Kambuzuma</span> — neural stack processing</li>
                  <li><span className="font-mono text-light">Buhera</span> — virtual processor OS</li>
                  <li><span className="font-mono text-light">Musande</span> — S-entropy solver</li>
                  <li><span className="font-mono text-light">Purpose Framework</span> — 47+ domain models</li>
                  <li><span className="font-mono text-light">Combine Harvester</span> — knowledge integration</li>
                  <li><span className="font-mono text-light">Four-Sided Triangle</span> — Bayesian optimization</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Key Theorems */}
          <div className="border-t border-primary/10 pt-16">
            <h2 className="text-2xl font-bold mb-8">Mathematical Foundation</h2>
            <div className="grid grid-cols-2 gap-6 lg:grid-cols-1">
              <TheoremCard
                name="TRIPLE EQUIVALENCE THEOREM"
                formula={<>S<sub>osc</sub> = S<sub>cat</sub> = S<sub>part</sub> = k<sub>B</sub> &middot; M &middot; ln(n)</>}
                description="Three descriptions of the same system — oscillatory, categorical, and partition — yield identical entropy. Any proof in one domain transfers to the others. This is the mathematical basis for cross-modal composition."
              />
              <TheoremCard
                name="TRIT-CELL CORRESPONDENCE"
                formula={<>k-trit string &harr; one cell in 3<sup>k</sup> partition (bijective)</>}
                description="A k-trit sequence simultaneously encodes position, trajectory, and address. Navigation through categorical space, memory addressing, and data identification are the same mathematical operation."
              />
              <TheoremCard
                name="INFORMATION MINIMALITY THEOREM"
                formula={<>|&sigma;| &le; I(D; A<sub>Q</sub>) &laquo; H(D)</>}
                description="For any research question Q and dataset D, the extracted representation σ is a sufficient statistic bounded by the mutual information between data and answer. The raw data entropy H(D) is never accessed beyond this bound."
              />
              <TheoremCard
                name="CENTRAL STATE IMPOSSIBILITY"
                formula={<>E<sub>meas</sub> &prop; 1/(&sigma;<sub>pos</sub> &middot; &sigma;<sub>mom</sub>) &rarr; &infin;</>}
                description="Perfect knowledge of individual node state requires infinite entropy — thermodynamically forbidden. Coordination must proceed through bulk statistical properties, not individual tracking. This is why federated understanding works."
              />
            </div>
            <div className="mt-8 text-center">
              <Link href="/validation" className="btn-outline">See Validation Results &rarr;</Link>
            </div>
          </div>
        </Layout>
      </section>
    </>
  );
}
