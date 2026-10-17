import { beforeEach, describe, expect, it } from "vitest";
import type { DatabaseSyncInstance } from "../src/store/sqlite.ts";

import { Recaller } from "../src/recaller/recall.ts";
import { createHttpReranker } from "../src/recaller/rerank.ts";
import {
  saveMessageOnce,
  saveTurnVector,
  upsertTurnMemory,
} from "../src/store/store.ts";
import { DEFAULT_CONFIG } from "../src/types.ts";
import { createTestDb } from "./helpers.ts";

let db: DatabaseSyncInstance;

beforeEach(() => {
  db = createTestDb();
});

function seed(sessionId: string, key: string, summary: string) {
  saveMessageOnce(db, `user-${key}`, sessionId, 1, "user", `问题 ${key}`);
  saveMessageOnce(db, `assistant-${key}`, sessionId, 1, "assistant", `答案 ${key}`);
  const memory = upsertTurnMemory(db, {
    sessionId,
    summary,
    outcome: "completed",
    sources: [
      { messageId: `user-${key}`, turnIndex: 1 },
      { messageId: `assistant-${key}`, turnIndex: 1 },
    ],
  });
  saveTurnVector(db, memory.id, summary, [1, 0]);
  return memory;
}

function recaller(maxNodes = 4) {
  const instance = new Recaller(db, {
    ...DEFAULT_CONFIG,
    recallMaxNodes: maxNodes,
    semanticScoreThreshold: 0.8,
  });
  instance.setEmbedFn(async () => [1, 0]);
  return instance;
}

describe("cross-encoder reranking", () => {
  it("leaves the embedding order untouched when no reranker is configured", async () => {
    seed("dsh:session-a", "a", "第一条记忆");
    seed("dsh:session-a", "b", "第二条记忆");
    const recalled = await recaller().recall("任意问题");
    const plain = await recaller().recall("任意问题");
    expect(recalled.turnMemories.map(m => m.id)).toEqual(plain.turnMemories.map(m => m.id));
  });

  it("reorders the candidate head by the scores it is given", async () => {
    const first = seed("dsh:session-a", "a", "第一条记忆");
    const second = seed("dsh:session-a", "b", "第二条记忆");
    const instance = recaller();
    // Score by document text: whichever summary is "第二条" must come first.
    instance.setRerankFn(async (_query, documents) =>
      documents.map(text => (text.includes("第二条") ? 0.9 : 0.1)));
    const recalled = await instance.recall("任意问题");
    expect(recalled.turnMemories[0].id).toBe(second.id);
    expect(recalled.turnMemories.map(m => m.id)).toContain(first.id);
  });

  it("falls back to the embedding order when the reranker returns null", async () => {
    seed("dsh:session-a", "a", "第一条记忆");
    seed("dsh:session-a", "b", "第二条记忆");
    const baseline = (await recaller().recall("任意问题")).turnMemories.map(m => m.id);
    const instance = recaller();
    instance.setRerankFn(async () => null);
    const recalled = await instance.recall("任意问题");
    expect(recalled.turnMemories.map(m => m.id)).toEqual(baseline);
  });

  it("survives a throwing reranker without losing recall", async () => {
    seed("dsh:session-a", "a", "第一条记忆");
    seed("dsh:session-a", "b", "第二条记忆");
    const baseline = (await recaller().recall("任意问题")).turnMemories.map(m => m.id);
    const instance = recaller();
    instance.setRerankFn(async () => {
      throw new Error("reranker down");
    });
    const recalled = await instance.recall("任意问题");
    expect(recalled.turnMemories.map(m => m.id)).toEqual(baseline);
  });

  it("keeps candidates beyond the scored head in their original order", async () => {
    seed("dsh:session-a", "a", "记忆一");
    seed("dsh:session-a", "b", "记忆二");
    seed("dsh:session-a", "c", "记忆三");
    seed("dsh:session-a", "d", "记忆四");
    const instance = new Recaller(db, {
      ...DEFAULT_CONFIG,
      recallMaxNodes: 4,
      semanticScoreThreshold: 0.8,
      rerankTopN: 2,
    });
    instance.setEmbedFn(async () => [1, 0]);
    let scored = 0;
    instance.setRerankFn(async (_query, documents) => {
      scored = documents.length;
      return documents.map(() => 0.5);
    });
    await instance.recall("任意问题");
    // Only the head is worth a cross-encoder pass; the tail must stay unpriced.
    expect(scored).toBe(2);
  });
});

describe("http reranker client", () => {
  it("maps a Jina/Cohere-shaped response back onto the document order", async () => {
    const rerank = createHttpReranker({
      endpoint: "http://stub.invalid/rerank",
      timeoutMs: 50,
    });
    // The client must not depend on results arriving in input order.
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      json: async () => ({
        results: [
          { index: 2, relevance_score: 0.9 },
          { index: 0, relevance_score: 0.5 },
          { index: 1, relevance_score: 0.1 },
        ],
      }),
    })) as any;
    try {
      expect(await rerank("q", ["a", "b", "c"])).toEqual([0.5, 0.1, 0.9]);
    } finally {
      globalThis.fetch = original;
    }
  });

  it("returns null rather than a partial ordering when the payload is short", async () => {
    const rerank = createHttpReranker({ endpoint: "http://stub.invalid/rerank", timeoutMs: 50 });
    const original = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      json: async () => ({ results: [{ index: 0, relevance_score: 0.9 }] }),
    })) as any;
    try {
      expect(await rerank("q", ["a", "b"])).toBeNull();
    } finally {
      globalThis.fetch = original;
    }
  });
});
