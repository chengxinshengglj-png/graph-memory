/**
 * Clustering primitives used by memory consolidation.
 *
 * Adapted from slow-stack/mneme (MIT License) — `dsh-mneme/src/dream/clustering.js`,
 * https://github.com/slow-stack/mneme. The algorithms are reproduced unchanged;
 * only types and the grouping key are parameterized so they fit this store's
 * turn-memory shape. Original copyright (c) slow-stack/mneme contributors.
 *
 * Kept as a standalone module: nothing here touches the database, so it can be
 * exercised and tested without a store, and the consolidation layer can be
 * dropped without leaving a mark on recall or extraction.
 */
/**
 * Cosine similarity of two equal-length vectors. Returns 0 for empty input,
 * mismatched lengths, or a zero-norm vector (numerically stable rather than
 * NaN, so a degenerate embedding cannot poison a whole run).
 */
export function cosineSimilarity(a, b) {
    if (!a || !b || a.length === 0 || a.length !== b.length)
        return 0;
    let dot = 0;
    let na = 0;
    let nb = 0;
    for (let i = 0; i < a.length; i += 1) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if (na === 0 || nb === 0)
        return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}
/**
 * K-Means over numeric vectors, returning one index array per cluster.
 *
 * k-means++ seeding, and empty clusters are reseeded mid-iteration so a bad
 * draw cannot collapse the result to fewer clusters than requested.
 */
export function kMeans(vectors, k, opts = {}) {
    const n = vectors.length;
    if (n === 0)
        return [];
    const { maxIter = 100, tol = 1e-4, random = Math.random } = opts;
    if (k <= 0 || k >= n)
        return [vectors.map((_, index) => index)];
    const dim = vectors[0].length;
    const randomIndex = () => Math.floor(random() * n);
    const squaredDist = (v, c) => {
        let sum = 0;
        for (let d = 0; d < dim; d += 1)
            sum += (v[d] - c[d]) * (v[d] - c[d]);
        return sum;
    };
    // First centroid uniform, later ones sampled proportional to squared distance.
    const centroids = [Float32Array.from(vectors[randomIndex()])];
    while (centroids.length < k) {
        const dist = new Array(n);
        let sum = 0;
        for (let i = 0; i < n; i += 1) {
            let best = Infinity;
            for (const centroid of centroids)
                best = Math.min(best, squaredDist(vectors[i], centroid));
            dist[i] = best;
            sum += best;
        }
        if (sum === 0) {
            centroids.push(Float32Array.from(vectors[randomIndex()]));
        }
        else {
            let r = random() * sum;
            let pick = 0;
            for (let i = 0; i < n; i += 1) {
                r -= dist[i];
                if (r <= 0) {
                    pick = i;
                    break;
                }
            }
            centroids.push(Float32Array.from(vectors[pick]));
        }
    }
    const assignment = new Array(n).fill(0);
    for (let iter = 0; iter < maxIter; iter += 1) {
        let changed = false;
        for (let i = 0; i < n; i += 1) {
            let best = 0;
            let bestD = Infinity;
            for (let j = 0; j < k; j += 1) {
                const d = squaredDist(vectors[i], centroids[j]);
                if (d < bestD) {
                    bestD = d;
                    best = j;
                }
            }
            if (assignment[i] !== best) {
                assignment[i] = best;
                changed = true;
            }
        }
        if (!changed)
            break;
        const sums = Array.from({ length: k }, () => new Float32Array(dim));
        const counts = new Array(k).fill(0);
        for (let i = 0; i < n; i += 1) {
            const cluster = assignment[i];
            counts[cluster] += 1;
            for (let d = 0; d < dim; d += 1)
                sums[cluster][d] += vectors[i][d];
        }
        let maxMove = 0;
        for (let j = 0; j < k; j += 1) {
            if (counts[j] === 0) {
                centroids[j] = Float32Array.from(vectors[randomIndex()]);
                maxMove = Infinity;
                continue;
            }
            for (let d = 0; d < dim; d += 1) {
                const mean = sums[j][d] / counts[j];
                const move = mean - centroids[j][d];
                if (move * move > maxMove)
                    maxMove = move * move;
                centroids[j][d] = mean;
            }
        }
        if (Math.sqrt(maxMove) < tol)
            break;
    }
    const clusters = Array.from({ length: k }, () => []);
    for (let i = 0; i < n; i += 1)
        clusters[assignment[i]].push(i);
    return clusters.filter(cluster => cluster.length > 0);
}
/** Cluster items by their vectors, preserving the original object references. */
export function clusterByVector(items, vectors, k) {
    return kMeans(vectors, k).map(cluster => cluster.map(index => items[index]));
}
/**
 * Pairs that are highly similar yet might contradict each other.
 *
 * Similarity is only a candidate signal: `0.9` means "read these two together",
 * never "these two disagree". Only items sharing the same group key are paired,
 * because a merge or conflict between unrelated kinds of memory is meaningless.
 */
export function findPotentialConflicts(items, vectors, keyOf, threshold = 0.85) {
    const pairs = [];
    for (let i = 0; i < items.length; i += 1) {
        for (let j = i + 1; j < items.length; j += 1) {
            if (keyOf(items[i]) !== keyOf(items[j]))
                continue;
            const similarity = cosineSimilarity(vectors[i], vectors[j]);
            if (similarity > threshold)
                pairs.push({ a: items[i], b: items[j], similarity });
        }
    }
    return pairs;
}
