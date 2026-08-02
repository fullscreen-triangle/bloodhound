/**
 * Browser-side PDSVM (Purpose-Driven Shader Virtual Machine) engine.
 * Implements the probe operator, ternary trie, and spectral dot-product kernel
 * entirely in JavaScript — no server round-trip required.
 *
 * Based on: "Purpose-Driven Shader Virtual Machine: A Unified Theory of
 * Intentional Computation over Bounded Oscillatory Systems"
 */

/** L2 distance from a point (sk, st, se) to a purpose vector [pk, pt, pe] */
export function purposeDistance(sk, st, se, purpose) {
  const [pk, pt, pe] = purpose;
  return Math.sqrt((sk - pk) ** 2 + (st - pt) ** 2 + (se - pe) ** 2);
}

/** Cosine similarity (Spectral Dot Product) between two [sk,st,se] vectors */
export function spectralDP(a, b) {
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const na = Math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2);
  const nb = Math.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2);
  return na > 0 && nb > 0 ? dot / (na * nb) : 0;
}

/**
 * Encode a single scalar x ∈ [0,1] as k base-3 digits (most significant first).
 * Exactly the ternary address scheme from Theorem 4.6 of the PDSVM paper.
 */
function toTrits(x, k) {
  const cap = 3 ** k;
  let v = Math.min(Math.floor(x * cap), cap - 1);
  const trits = new Array(k);
  for (let i = k - 1; i >= 0; i--) {
    trits[i] = v % 3;
    v = Math.floor(v / 3);
  }
  return trits;
}

/** Full 3k-trit address for a point (sk, st, se) with k digits per axis */
export function ternaryAddress(sk, st, se, k = 6) {
  return [...toTrits(sk, k), ...toTrits(st, k), ...toTrits(se, k)];
}

/**
 * Build a ternary trie (nested plain objects) over an array of data points.
 * Each leaf stores the array indices of items that hash to that address.
 * Lookup cost: exactly 3k dict-key comparisons, independent of |data|.
 */
export function buildTrie(data, k = 6) {
  const root = {};
  data.forEach((item, idx) => {
    const addr = ternaryAddress(item.sk, item.st, item.se, k);
    let node = root;
    for (const trit of addr) {
      if (!node[trit]) node[trit] = {};
      node = node[trit];
    }
    if (!node._indices) node._indices = [];
    node._indices.push(idx);
  });
  return root;
}

/**
 * Exact trie lookup — O(3k) comparisons, independent of database size.
 * Returns the indices of items at the same leaf as (sk, st, se).
 */
export function trieQuery(root, sk, st, se, k = 6) {
  const addr = ternaryAddress(sk, st, se, k);
  let node = root;
  for (const trit of addr) {
    if (!node[trit]) return [];
    node = node[trit];
  }
  return node._indices || [];
}

/**
 * Epsilon-shrinking probe operator Π_P.
 *
 * Algorithm:
 *  1. Compute purpose-distance d(i, P) for every item.
 *  2. Sort ascending.
 *  3. Iterate: threshold_n = d_min + eps0 * 0.82^n
 *     Keep items inside threshold; stop when |cell| < minSize.
 *
 * Returns { stable, history } where stable is the final cell array (items
 * annotated with _dist) and history is [{n, size, meanDist}].
 *
 * @param {Array}  data    - Array of objects with {sk, st, se}
 * @param {Array}  purpose - [pk, pt, pe] purpose point
 * @param {Object} opts    - { maxIter=25, eps0=null, minSize=2, rate=0.82 }
 */
export function probeIterate(data, purpose, opts = {}) {
  const { maxIter = 25, eps0 = null, minSize = 2, rate = 0.82 } = opts;

  const annotated = data.map((item, idx) => ({
    ...item,
    _idx: idx,
    _dist: purposeDistance(item.sk, item.st, item.se, purpose),
  }));

  annotated.sort((a, b) => a._dist - b._dist);

  const dists = annotated.map((d) => d._dist);
  const dMin = dists[0];

  // eps0 defaults to std-dev of all distances (makes threshold data-adaptive)
  let eps;
  if (eps0 !== null) {
    eps = eps0;
  } else {
    const mean = dists.reduce((s, d) => s + d, 0) / dists.length;
    const variance = dists.reduce((s, d) => s + (d - mean) ** 2, 0) / dists.length;
    eps = Math.sqrt(variance);
  }

  let cell = [...annotated];
  const history = [{ n: 0, size: cell.length, meanDist: dists.reduce((s, d) => s + d, 0) / dists.length }];

  for (let n = 1; n <= maxIter; n++) {
    const threshold = dMin + eps * rate ** n;
    const next = cell.filter((item) => item._dist <= threshold);
    if (next.length < minSize) break;
    cell = next;
    const meanDist = cell.reduce((s, d) => s + d._dist, 0) / cell.length;
    history.push({ n, size: cell.length, meanDist });
  }

  return { stable: cell, history };
}

/**
 * Rank all items by purpose distance and return the top-k closest,
 * without running the full probe convergence. Useful for quick previews.
 */
export function rankByPurpose(data, purpose, topK = 10) {
  return data
    .map((item, idx) => ({
      ...item,
      _idx: idx,
      _dist: purposeDistance(item.sk, item.st, item.se, purpose),
    }))
    .sort((a, b) => a._dist - b._dist)
    .slice(0, topK);
}
