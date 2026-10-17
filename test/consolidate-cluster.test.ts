import { describe, expect, it } from "vitest";

import {
  clusterByVector,
  cosineSimilarity,
  findPotentialConflicts,
  kMeans,
} from "../src/consolidate/cluster.ts";

const vec = (...values: number[]) => Float32Array.from(values);

describe("cosine similarity", () => {
  it("returns 1 for identical direction and 0 for orthogonal input", () => {
    expect(cosineSimilarity(vec(1, 0), vec(1, 0))).toBeCloseTo(1, 6);
    expect(cosineSimilarity(vec(1, 0), vec(0, 1))).toBeCloseTo(0, 6);
  });

  it("returns 0 instead of NaN for degenerate input", () => {
    // A zero-norm vector would divide by zero and poison every downstream score.
    expect(cosineSimilarity(vec(0, 0), vec(1, 1))).toBe(0);
    expect(cosineSimilarity(vec(1, 1), vec(1, 1, 1))).toBe(0);
  });
});

describe("k-means", () => {
  it("separates two clearly disjoint groups", () => {
    const vectors = [vec(0, 0), vec(0, 0.1), vec(9, 9), vec(9.1, 9)];
    const clusters = kMeans(vectors, 2, { random: () => 0.5 });
    expect(clusters).toHaveLength(2);
    const sizes = clusters.map(cluster => cluster.length).sort();
    expect(sizes).toEqual([2, 2]);
    // Every cluster must be internally tight: no group may mix the two blobs.
    for (const cluster of clusters) {
      const values = cluster.map(index => vectors[index][0]);
      expect(Math.max(...values) - Math.min(...values)).toBeLessThan(1);
    }
  });

  it("collapses to a single cluster when k exceeds the input size", () => {
    const clusters = kMeans([vec(1, 1), vec(2, 2)], 5);
    expect(clusters).toEqual([[0, 1]]);
  });

  it("returns an empty result for no input", () => {
    expect(kMeans([], 3)).toEqual([]);
  });

  it("preserves item identity when clustering objects", () => {
    const items = [{ id: "a" }, { id: "b" }];
    const grouped = clusterByVector(items, [vec(0, 0), vec(0, 0.1)], 1);
    expect(grouped).toHaveLength(1);
    expect(grouped[0].map(item => item.id).sort()).toEqual(["a", "b"]);
  });
});

describe("potential conflicts", () => {
  it("reports a near-duplicate pair once, lower index first", () => {
    const items = [{ key: "x", id: 1 }, { key: "x", id: 2 }];
    const pairs = findPotentialConflicts(items, [vec(1, 0), vec(1, 0.01)], item => item.key);
    expect(pairs).toHaveLength(1);
    expect(pairs[0].a.id).toBe(1);
    expect(pairs[0].b.id).toBe(2);
    expect(pairs[0].similarity).toBeGreaterThan(0.99);
  });

  it("never pairs items that belong to different groups", () => {
    const items = [{ key: "x", id: 1 }, { key: "y", id: 2 }];
    // Identical vectors, different groups: a merge or conflict across kinds of
    // memory would be meaningless, so the group key gates the pair.
    expect(findPotentialConflicts(items, [vec(1, 0), vec(1, 0)], item => item.key)).toEqual([]);
  });

  it("respects the similarity threshold", () => {
    const items = [{ key: "x", id: 1 }, { key: "x", id: 2 }];
    expect(findPotentialConflicts(items, [vec(1, 0), vec(0.7, 0.7)], item => item.key, 0.95)).toEqual([]);
  });
});
