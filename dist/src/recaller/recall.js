/**
 * graph-memory — 跨对话召回
 *
 * By: adoresever
 * Email: Wywelljob@gmail.com
 *
 * Query recall is deliberately relevance-first: vector ranking is preserved,
 * with FTS5 as an exact-term/failure fallback. Graph centrality is useful for
 * offline graph inspection, but must not displace the memories most similar
 * to the current user question.
 */
import { createHash } from "crypto";
import { searchNodes, vectorSearchWithScore, graphWalk, saveVector, getVectorHash, searchTurnMemories, turnMemoryVectorSearchWithScore, nodesForTurnMemories, saveTurnVector, getTurnVectorHash, getNavigationTriplesForMemories, findNavigationSeedTermIds, navigationCandidateTermIds, rankTurnMemoryIdsByNavigation, getTurnMemoriesByIds, hasTurnMemories, } from "../store/store.js";
import { personalizedNavigationPageRank } from "../graph/pagerank.js";
/**
 * Recency is a tie-breaker, not a relevance signal. A memory that is slightly
 * less similar but four months older should not outrank the one that describes
 * what the user is doing this week, yet it must not jump ahead of a clearly
 * better match either. A 30-day half life with a 0.15 ceiling sits below one
 * reciprocal-rank step (1/1 - 1/2 = 0.5), so it can reorder near-ties without
 * overriding the similarity ranks that the embedding provider produced.
 */
const RECENCY_HALF_LIFE_MS = 30 * 24 * 60 * 60 * 1000;
const RECENCY_WEIGHT = 0.15;
/** 1 for a memory written now, 0.5 at thirty days, 0.25 at sixty. */
function recencyOf(memory, now) {
    const updatedAt = Number(memory.updatedAt);
    if (!Number.isFinite(updatedAt) || updatedAt <= 0)
        return 0;
    return Math.pow(0.5, Math.max(0, now - updatedAt) / RECENCY_HALF_LIFE_MS);
}
export class Recaller {
    db;
    cfg;
    embed = null;
    embeddingFingerprint = "";
    constructor(db, cfg) {
        this.db = db;
        this.cfg = cfg;
    }
    setEmbedFn(fn, fingerprint = "") {
        this.embed = fn;
        this.embeddingFingerprint = fingerprint;
    }
    /**
     * @param options.overFetch Retrieval pool multiplier. Defaults to 1, which
     * keeps the historical behaviour: `recallMaxNodes` bounds both retrieval and
     * what the caller may render. A host that post-filters results (for example
     * to drop memories already visible in its fresh window) passes a multiplier
     * so that filtering cannot empty the whole block: a three-slot pool where all
     * three hits belong to the current session used to yield nothing at all, and
     * the caller then injected no memory for that turn.
     */
    async recall(query, options = {}) {
        const limit = this.cfg.recallMaxNodes;
        const overFetch = Math.max(1, Math.floor(options.overFetch ?? 1));
        const poolLimit = limit * overFetch;
        let queryVector;
        if (this.embed) {
            try {
                queryVector = await this.embed(query, "query");
            }
            catch {
                // The lexical path remains available when the embedding provider is
                // temporarily unavailable.
            }
        }
        const directMemories = this.recallTurnMemories(query, poolLimit, queryVector);
        const seedIds = findNavigationSeedTermIds(this.db, query, directMemories.map(memory => memory.id));
        let graphMemories = [];
        let navigationScores = new Map();
        if (seedIds.length) {
            const candidateTermIds = navigationCandidateTermIds(this.db, seedIds);
            navigationScores = personalizedNavigationPageRank(this.db, seedIds, candidateTermIds, this.cfg).scores;
            graphMemories = getTurnMemoriesByIds(this.db, rankTurnMemoryIdsByNavigation(this.db, navigationScores));
        }
        const turnMemories = this.mergeTurnMemoryRanks(directMemories, graphMemories, poolLimit);
        if (turnMemories.length) {
            const memoryIds = turnMemories.map(memory => memory.id);
            const nodes = nodesForTurnMemories(this.db, memoryIds, poolLimit);
            const { edges } = graphWalk(this.db, nodes.map(node => node.id), 0);
            return {
                nodes,
                edges,
                turnMemories,
                triples: getNavigationTriplesForMemories(this.db, memoryIds, navigationScores),
            };
        }
        // Databases created before the turn-memory migration remain searchable.
        // The same confidence rule applies; a legacy node match cannot bypass it.
        return this.recallPrecise(query, poolLimit, queryVector, hasTurnMemories(this.db));
    }
    /**
     * Fuse independent summary and graph ranks without mixing incomparable
     * cosine and PageRank score scales. A memory supported by both routes rises;
     * exact summary matches keep tie priority over graph-only expansion.
     */
    mergeTurnMemoryRanks(direct, graph, limit) {
        const candidates = new Map();
        direct.forEach((memory, index) => {
            candidates.set(memory.id, {
                memory,
                score: 1 / (index + 1),
                directRank: index,
            });
        });
        graph.forEach((memory, index) => {
            const existing = candidates.get(memory.id);
            if (existing) {
                existing.score += 1 / (index + 1);
            }
            else {
                candidates.set(memory.id, {
                    memory,
                    score: 1 / (index + 1),
                    directRank: Number.POSITIVE_INFINITY,
                });
            }
        });
        return Array.from(candidates.values())
            .map(candidate => ({
            ...candidate,
            score: candidate.score + RECENCY_WEIGHT * recencyOf(candidate.memory, Date.now()),
        }))
            .sort((left, right) => right.score - left.score
            || left.directRank - right.directRank
            || right.memory.updatedAt - left.memory.updatedAt
            || left.memory.id.localeCompare(right.memory.id))
            .slice(0, limit)
            .map(candidate => candidate.memory);
    }
    recallTurnMemories(query, limit, queryVector) {
        const lexical = searchTurnMemories(this.db, query, limit);
        const threshold = this.cfg.semanticScoreThreshold;
        const semantic = queryVector && threshold !== undefined
            ? turnMemoryVectorSearchWithScore(this.db, queryVector, limit, threshold)
            : [];
        const selected = [];
        const seen = new Set();
        const append = (memory) => {
            if (selected.length >= limit || seen.has(memory.id))
                return;
            seen.add(memory.id);
            selected.push(memory);
        };
        for (const { memory } of semantic)
            append(memory);
        for (const memory of lexical)
            append(memory);
        return selected;
    }
    /**
     * Preserve semantic rank. FTS5 contributes exact terms and is the complete
     * fallback when the embedding provider is absent or temporarily fails.
     */
    async recallPrecise(query, limit, queryVector, legacyOnly = false) {
        const lexical = searchNodes(this.db, query, limit, legacyOnly);
        const threshold = this.cfg.semanticScoreThreshold;
        const semantic = queryVector && threshold !== undefined
            ? vectorSearchWithScore(this.db, queryVector, limit, threshold, legacyOnly)
            : [];
        const selected = [];
        const selectedIds = new Set();
        const append = (node) => {
            if (selected.length >= limit || selectedIds.has(node.id))
                return;
            selected.push(node);
            selectedIds.add(node.id);
        };
        for (const { node } of semantic)
            append(node);
        for (const node of lexical)
            append(node);
        if (!selected.length)
            return { nodes: [], edges: [], turnMemories: [], triples: [] };
        // Depth zero asks the store only for edges whose two endpoints are direct
        // query matches. No unrelated graph hub is allowed to enter the prompt.
        const { edges } = graphWalk(this.db, selected.map(node => node.id), 0);
        return { nodes: selected, edges, turnMemories: [], triples: [] };
    }
    /** 异步同步 embedding，不阻塞主流程 */
    async syncEmbed(node) {
        if (!this.embed)
            return;
        const temporal = Object.keys(node.temporal).length ? `\n时间语义: ${JSON.stringify(node.temporal)}` : "";
        const text = `${node.name}: ${node.description}\n${node.content}${temporal}`;
        const hashInput = this.embeddingFingerprint ? `${this.embeddingFingerprint}\0${text}` : text;
        const hash = createHash("md5").update(hashInput).digest("hex");
        if (getVectorHash(this.db, node.id) === hash)
            return;
        try {
            const vec = await this.embed(text, "db");
            if (vec.length)
                saveVector(this.db, node.id, hashInput, vec);
        }
        catch { /* 不影响主流程 */ }
    }
    /** Keep the compact episodic layer independently searchable. */
    async syncTurnMemoryEmbed(memory) {
        if (!this.embed)
            return;
        const text = `${memory.outcome}: ${memory.summary}`;
        const hashInput = this.embeddingFingerprint ? `${this.embeddingFingerprint}\0${text}` : text;
        const hash = createHash("md5").update(hashInput).digest("hex");
        if (getTurnVectorHash(this.db, memory.id) === hash)
            return;
        try {
            const vec = await this.embed(text, "db");
            if (vec.length)
                saveTurnVector(this.db, memory.id, hashInput, vec);
        }
        catch { /* 不影响主流程 */ }
    }
}
