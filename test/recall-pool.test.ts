import { beforeEach, describe, expect, it } from "vitest";
import type { DatabaseSyncInstance } from "../src/store/sqlite.ts";

import { Recaller } from "../src/recaller/recall.ts";
import {
  saveMessageOnce,
  saveTurnVector,
  searchTurnMemories,
  upsertTurnMemory,
} from "../src/store/store.ts";
import { DEFAULT_CONFIG } from "../src/types.ts";
import { createTestDb } from "./helpers.ts";

let db: DatabaseSyncInstance;

beforeEach(() => {
  db = createTestDb();
});

function seedMemory(sessionId: string, key: string, summary: string, vector: number[]) {
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
  saveTurnVector(db, memory.id, summary, vector);
  return memory;
}

function recaller(maxNodes: number) {
  const instance = new Recaller(db, {
    ...DEFAULT_CONFIG,
    recallMaxNodes: maxNodes,
    semanticScoreThreshold: 0.8,
  });
  instance.setEmbedFn(async () => [1, 0]);
  return instance;
}

describe("recall pool sizing", () => {
  it("bounds retrieval by recallMaxNodes when no over-fetch is requested", async () => {
    for (let index = 1; index <= 5; index += 1) {
      seedMemory("dsh:session-a", `m${index}`, `第 ${index} 条记忆的内容`, [1, 0]);
    }
    const recalled = await recaller(2).recall("任意问题");
    expect(recalled.turnMemories).toHaveLength(2);
  });

  it("returns the wider pool the host needs to survive its visibility filter", async () => {
    for (let index = 1; index <= 5; index += 1) {
      seedMemory("dsh:session-a", `m${index}`, `第 ${index} 条记忆的内容`, [1, 0]);
    }
    // A three-slot pool whose hits all belong to the live session used to be
    // emptied by filterDshRecallMemories, injecting nothing for that turn.
    const narrow = await recaller(2).recall("任意问题");
    const wide = await recaller(2).recall("任意问题", { overFetch: 3 });
    expect(wide.turnMemories.length).toBeGreaterThan(narrow.turnMemories.length);
    expect(wide.turnMemories).toHaveLength(5);
  });

  it("treats a multiplier below one as the historical behaviour", async () => {
    for (let index = 1; index <= 4; index += 1) {
      seedMemory("dsh:session-a", `m${index}`, `第 ${index} 条记忆的内容`, [1, 0]);
    }
    const recalled = await recaller(2).recall("任意问题", { overFetch: 0 });
    expect(recalled.turnMemories).toHaveLength(2);
  });
});

describe("lexical turn-memory matching", () => {
  it("finds a summary from a multi-term query instead of requiring the sentence", () => {
    const target = seedMemory("dsh:session-a", "a", "客必得账号关联任务已全部完成", [1, 0]);
    seedMemory("dsh:session-a", "b", "配置文件已更新", [0, 1]);

    const hits = searchTurnMemories(db, "客必得 账号 关联", 5);
    expect(hits.map(hit => hit.id)).toEqual([target.id]);
  });

  it("strips a question tail so a natural question still matches", () => {
    const target = seedMemory("dsh:session-a", "a", "体系7审计已通过", [1, 0]);
    const hits = searchTurnMemories(db, "体系7审计好了吗？", 5);
    expect(hits.map(hit => hit.id)).toEqual([target.id]);
  });

  it("rejects a match that shares only one term of a two-term query", () => {
    seedMemory("dsh:session-a", "a", "发布端口已修复为9090", [1, 0]);
    seedMemory("dsh:session-a", "b", "配置文件已更新", [0, 1]);

    // Sharing "配置文件" alone is not evidence for "发布端口，配置文件":
    // this exact query used to return five summaries that never mentioned the
    // port at all.
    expect(searchTurnMemories(db, "发布端口，配置文件", 5)).toEqual([]);
  });
});
