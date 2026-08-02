/**
 * Intent mapper: translates a free-text research question into a purpose point
 * P = [pk, pt, pe] ∈ [0,1]³ in S-entropy coordinate space.
 *
 * This is the "intent translator" layer of the Buhera blank-screen stack —
 * the bridge between the user's utterance and the PDSVM probe operator.
 *
 * Each intent rule specifies:
 *   terms    – substrings to match (case-insensitive, any match triggers)
 *   purpose  – [Sk, St, Se] centroid of the relevant region in M=[0,1]³
 *   label    – human-readable description of the matched intent
 *   families – CYP families that are primary targets of this intent
 */

const INTENT_RULES = [
  // ── Broadest drug metabolism ──────────────────────────────────────────────
  {
    terms: ["drug metabolism", "drug metaboli", "pharmacokinetic", "bioavailability",
            "first-pass", "clearance", "half-life"],
    purpose: [0.72, 0.60, 0.34],
    label: "Drug Metabolism",
    families: [1, 2, 3],
    description: "Enzymes responsible for metabolizing pharmaceutical compounds",
  },
  // ── Specific major metabolizers ──────────────────────────────────────────
  {
    terms: ["3a4", "cyp3a4", "most drugs", "midazolam", "cyclosporine", "statins",
            "tacrolimus", "rifampin", "ketoconazole inhibition"],
    purpose: [0.88, 0.74, 0.43],
    label: "CYP3A4 (Major Drug Metabolizer)",
    families: [3],
    description: "Metabolizes >50% of all clinically used drugs",
  },
  {
    terms: ["warfarin", "anticoagulant", "phenytoin", "2c9", "cyp2c9", "nsaid",
            "diclofenac", "losartan"],
    purpose: [0.55, 0.49, 0.29],
    label: "CYP2C9 (Warfarin / NSAIDs)",
    families: [2],
    description: "Critical for warfarin and NSAID metabolism; highly polymorphic",
  },
  {
    terms: ["codeine", "opioid", "tramadol", "2d6", "cyp2d6", "antidepressant",
            "haloperidol", "tamoxifen", "metoprolol"],
    purpose: [0.38, 0.43, 0.23],
    label: "CYP2D6 (Opioids / Antidepressants)",
    families: [2],
    description: "Narrow active site; ~7% Caucasians are poor metabolizers",
  },
  {
    terms: ["omeprazole", "proton pump", "clopidogrel", "2c19", "cyp2c19",
            "voriconazole", "diazepam"],
    purpose: [0.48, 0.46, 0.26],
    label: "CYP2C19 (PPIs / Clopidogrel)",
    families: [2],
    description: "Activates clopidogrel; poor metabolizers have higher stroke risk",
  },
  {
    terms: ["alcohol", "ethanol", "acetaminophen", "paracetamol", "2e1", "cyp2e1",
            "reactive oxygen", "ros"],
    purpose: [0.26, 0.38, 0.21],
    label: "CYP2E1 (Ethanol / Acetaminophen)",
    families: [2],
    description: "Major source of ROS in alcoholic liver disease",
  },
  {
    terms: ["efavirenz", "hiv", "cyclophosphamide", "bupropion", "2b6", "cyp2b6",
            "methadone", "ketamine"],
    purpose: [0.58, 0.51, 0.31],
    label: "CYP2B6 (Efavirenz / HIV drugs)",
    families: [2],
    description: "Highly polymorphic; critical for antiretroviral therapy dosing",
  },
  {
    terms: ["nicotine", "smoking", "tobacco", "coumarin", "2a6", "cyp2a6"],
    purpose: [0.31, 0.42, 0.20],
    label: "CYP2A6 (Nicotine)",
    families: [2],
    description: "Primary nicotine-metabolizing enzyme; variants affect smoking cessation",
  },
  // ── Steroidogenesis ───────────────────────────────────────────────────────
  {
    terms: ["steroid", "steroidogenesis", "cortisol", "aldosterone", "adrenal",
            "congenital adrenal", "glucocorticoid", "mineralocorticoid"],
    purpose: [0.12, 0.22, 0.11],
    label: "Steroidogenesis",
    families: [11, 17, 21],
    description: "Adrenal and gonadal enzymes synthesizing glucocorticoids and mineralocorticoids",
  },
  {
    terms: ["testosterone", "androgen", "estrogen", "sex hormone", "aromatase",
            "breast cancer hormone", "19a1", "cyp19a1"],
    purpose: [0.13, 0.23, 0.11],
    label: "Sex Hormone Synthesis (Aromatase)",
    families: [17, 19],
    description: "Converts androgens to estrogens; aromatase inhibitors treat breast cancer",
  },
  {
    terms: ["abiraterone", "prostate cancer", "17a1", "cyp17a1", "castration",
            "lyase", "dhea"],
    purpose: [0.18, 0.26, 0.13],
    label: "CYP17A1 (Prostate Cancer)",
    families: [17],
    description: "Target of abiraterone in castration-resistant prostate cancer",
  },
  // ── Vitamin D ─────────────────────────────────────────────────────────────
  {
    terms: ["vitamin d", "cholecalciferol", "rickets", "calcitriol", "27b1",
            "cyp27b1", "1-alpha hydroxylase", "1,25-dihydroxy"],
    purpose: [0.11, 0.21, 0.10],
    label: "Vitamin D Activation",
    families: [27],
    description: "1α-hydroxylase activates vitamin D; mutations cause VDDR1A",
  },
  {
    terms: ["vitamin d catabolism", "24-hydroxylase", "24a1", "cyp24a1",
            "hypercalcemia"],
    purpose: [0.14, 0.24, 0.12],
    label: "Vitamin D Catabolism (CYP24A1)",
    families: [24],
    description: "Inactivates 1,25-dihydroxyvitamin D3; mutations cause infantile hypercalcemia",
  },
  // ── Bile acids and cholesterol ────────────────────────────────────────────
  {
    terms: ["bile acid", "cholesterol synthesis", "bile salt", "cholic acid",
            "7a1", "cyp7a1", "bile flow"],
    purpose: [0.12, 0.23, 0.11],
    label: "Bile Acid Synthesis",
    families: [7, 8],
    description: "Rate-limiting cholesterol catabolism to bile acids in liver",
  },
  {
    terms: ["cholesterol", "lanosterol", "statin target", "51a1", "cyp51",
            "azole antifungal"],
    purpose: [0.14, 0.24, 0.11],
    label: "Cholesterol Biosynthesis (CYP51A1)",
    families: [51],
    description: "Target of azole antifungals; most evolutionarily conserved CYP",
  },
  {
    terms: ["brain cholesterol", "alzheimer", "46a1", "cyp46a1", "neurosteroid",
            "24s-hydroxycholesterol"],
    purpose: [0.12, 0.21, 0.11],
    label: "Brain Cholesterol (CYP46A1)",
    families: [46],
    description: "Controls cholesterol turnover in neurons; linked to Alzheimer's risk",
  },
  // ── Fatty acids and eicosanoids ───────────────────────────────────────────
  {
    terms: ["fatty acid", "omega-hydroxylation", "4a11", "cyp4a11", "20-hete",
            "hypertension", "renal", "blood pressure"],
    purpose: [0.33, 0.44, 0.24],
    label: "Fatty Acid Metabolism (20-HETE)",
    families: [4],
    description: "ω-Hydroxylation of arachidonic acid to 20-HETE; vasoconstriction",
  },
  {
    terms: ["arachidonic", "leukotriene", "prostaglandin", "eicosanoid",
            "inflammation", "4f2", "cyp4f2", "vitamin k"],
    purpose: [0.26, 0.38, 0.19],
    label: "Eicosanoid / Leukotriene Metabolism",
    families: [4],
    description: "Inactivates inflammatory lipid mediators; CYP4F2 metabolizes vitamin K",
  },
  {
    terms: ["thromboxane", "platelet", "5a1", "cyp5a1", "aspirin", "aggregation"],
    purpose: [0.10, 0.19, 0.09],
    label: "Thromboxane Synthase (CYP5A1)",
    families: [5],
    description: "Produces platelet-aggregating TXA2; upstream of aspirin's mechanism",
  },
  {
    terms: ["prostacyclin", "pgi2", "endothelium", "8a1", "cyp8a1", "vasodilation"],
    purpose: [0.10, 0.19, 0.09],
    label: "Prostacyclin Synthase (CYP8A1)",
    families: [8],
    description: "Produces vasodilatory PGI2; balance with TXA2 determines thrombosis risk",
  },
  // ── Carcinogen activation ─────────────────────────────────────────────────
  {
    terms: ["carcinogen", "cancer", "mutagenesis", "polycyclic aromatic",
            "pah", "benzo pyrene", "heterocyclic amine", "1a1", "cyp1a1"],
    purpose: [0.28, 0.38, 0.18],
    label: "Carcinogen Bioactivation (CYP1A1)",
    families: [1],
    description: "Activates PAH carcinogens; AhR-inducible; major concern in smokers",
  },
  {
    terms: ["caffeine", "clozapine", "theophylline", "1a2", "cyp1a2",
            "aromatic amine"],
    purpose: [0.52, 0.48, 0.24],
    label: "CYP1A2 (Caffeine / Clozapine)",
    families: [1],
    description: "~13% of hepatic CYP content; activated by smoking and cruciferous vegetables",
  },
  // ── Retinoic acid ─────────────────────────────────────────────────────────
  {
    terms: ["retinoic acid", "retinoid", "vitamin a", "all-trans", "embryo",
            "26a1", "cyp26", "retinol"],
    purpose: [0.11, 0.21, 0.10],
    label: "Retinoic Acid Metabolism (CYP26 Family)",
    families: [26],
    description: "Control RA gradients in embryogenesis; mutations cause craniosynostosis",
  },
  // ── Tissue-specific ───────────────────────────────────────────────────────
  {
    terms: ["lung", "pulmonary", "inhaled", "smoking lung", "3-methylindole"],
    purpose: [0.15, 0.26, 0.13],
    label: "Lung-Specific CYPs",
    families: [1, 2],
    description: "CYPs expressed in lung; bioactivate inhaled carcinogens and toxicants",
  },
  {
    terms: ["liver", "hepatic", "hepatocyte", "first pass"],
    purpose: [0.62, 0.52, 0.30],
    label: "Hepatic CYPs",
    families: [1, 2, 3],
    description: "Major hepatic drug-metabolizing enzymes",
  },
  {
    terms: ["brain", "neural", "cns", "neurological", "spg", "spastic paraplegia"],
    purpose: [0.13, 0.22, 0.11],
    label: "Brain / Neurological CYPs",
    families: [2, 4, 7, 27, 46],
    description: "CYPs involved in brain cholesterol and neurosteroid homeostasis",
  },
  {
    terms: ["skin", "ichthyosis", "dermatol", "epiderm", "4f22", "cyp4f22"],
    purpose: [0.13, 0.23, 0.11],
    label: "Skin Barrier CYPs",
    families: [4],
    description: "CYP4F22 mutations cause congenital ichthyosis (ARCI)",
  },
  // ── Polymorphism / pharmacogenomics ───────────────────────────────────────
  {
    terms: ["polymorphic", "pharmacogenomic", "poor metabolizer", "ultra-rapid",
            "genetic variant", "precision medicine"],
    purpose: [0.43, 0.44, 0.24],
    label: "Polymorphic CYPs (Pharmacogenomics)",
    families: [2],
    description: "Highly polymorphic enzymes driving inter-individual drug response variability",
  },
  // ── Paediatric / developmental ────────────────────────────────────────────
  {
    terms: ["fetal", "neonatal", "newborn", "perinatal", "pregnancy", "placenta"],
    purpose: [0.24, 0.32, 0.16],
    label: "Developmental / Fetal CYPs",
    families: [3],
    description: "CYP isoforms expressed during fetal development and placentation",
  },
];

/**
 * Parse a free-text query and return the best-matching intent.
 *
 * @param   {string} query  - The user's research question
 * @returns {{ purpose: number[], label: string, families: number[],
 *             description: string, matched: boolean }}
 */
export function mapQueryToPurpose(query) {
  const q = query.toLowerCase();

  for (const rule of INTENT_RULES) {
    if (rule.terms.some((term) => q.includes(term))) {
      return {
        purpose: rule.purpose,
        label: rule.label,
        families: rule.families,
        description: rule.description,
        matched: true,
      };
    }
  }

  // Fallback: moderate-specificity region (CYP2 family centroid)
  return {
    purpose: [0.35, 0.38, 0.20],
    label: "General CYP Query",
    families: [],
    description: "No specific intent matched — returning nearest enzymes by S-entropy distance",
    matched: false,
  };
}

/**
 * Suggest query refinements based on what the current result set contains.
 * Used to populate "Refine your search" chips in the UI.
 */
export function suggestRefinements(stableItems) {
  const families = [...new Set(stableItems.map((d) => d.family))];
  const suggestions = [];

  if (families.includes(3)) suggestions.push("Show only CYP3A4");
  if (families.includes(2)) suggestions.push("Focus on polymorphic variants");
  if (families.some((f) => [11, 17, 19, 21].includes(f))) suggestions.push("Steroidogenesis pathway");
  if (families.includes(4)) suggestions.push("Fatty acid oxidation");
  if (stableItems.some((d) => d.clinical_relevance === "high")) suggestions.push("High clinical relevance only");
  if (stableItems.some((d) => d.polymorphic)) suggestions.push("Polymorphic enzymes only");

  return suggestions.slice(0, 4);
}
