/**
 * graph-memory — concept layer persistence
 *
 * The extraction contract lets a turn declare durable concepts; these cases
 * pin down how that declaration becomes rows, and specifically the two rules a
 * recurring mention must not break: an established description is not silently
 * rewritten, and its provenance is only ever added to.
 */

import { describe, it, expect, beforeEach } from "vitest";
import type { DatabaseSyncInstance } from "../src/store/sqlite.ts";
import { createTestDb } from "./helpers.ts";
import {
  applyConceptExtraction,
  findByName,
  getNodeSources,
  saveMessageOnce,
} from "../src/store/store.ts";

const SID = "dsh:session-concept";

let db: DatabaseSyncInstance;

/** Persist one durable Q/A pair and return the provenance refs for it. */
function seedTurn(turn: number): Array<{ messageId: string; turnIndex: number }> {
  saveMessageOnce(db, `${SID}:${turn}:u`, SID, turn, "user", "问题");
  saveMessageOnce(db, `${SID}:${turn}:a`, SID, turn, "assistant", "回答");
  return [`${SID}:${turn}:u`, `${SID}:${turn}:a`].map(messageId => ({ messageId, turnIndex: turn }));
}

beforeEach(() => {
  db = createTestDb();
});

describe("concept extraction persistence", () => {
  it("stores declared concepts and links them to the turn evidence", () => {
    const sources = seedTurn(1);
    const result = applyConceptExtraction(db, SID, sources,
      [
        { name: "抽取路由", type: "SKILL", description: "抽取请求使用的模型路由配置" },
        { name: "本地 27B", type: "SKILL", description: "本机 llama-swap 上的主模型" },
      ],
      [{ from: "抽取路由", to: "本地 27B", type: "REQUIRES", instruction: "路由改为依赖本地模型" }],
    );

    expect(result).toEqual({ nodes: 2, edges: 1 });
    const route = findByName(db, "抽取路由");
    expect(route).not.toBeNull();
    expect(route!.type).toBe("SKILL");
    // Provenance is what makes a concept reachable from recall.
    expect(getNodeSources(db, [route!.id]).length).toBe(2);
  });

  it("confirms a recurring concept instead of rewriting its established claim", () => {
    const first = seedTurn(1);
    applyConceptExtraction(db, SID, first,
      [{ name: "抽取路由", type: "SKILL", description: "初版描述" }], []);
    const original = findByName(db, "抽取路由")!;

    const second = seedTurn(2);
    const result = applyConceptExtraction(db, SID, second,
      [{ name: "抽取路由", type: "SKILL", description: "后来的一句话" }], []);

    const after = findByName(db, "抽取路由")!;
    expect(result.nodes).toBe(0);
    expect(after.id).toBe(original.id);
    expect(after.content).toBe(original.content);
    expect(after.validatedCount).toBe(original.validatedCount + 1);
    // Confirmation adds evidence; it never detaches the earlier turn.
    expect(getNodeSources(db, [after.id]).length).toBe(4);
  });

  it("drops a relation whose endpoints were not declared as concepts", () => {
    const sources = seedTurn(1);
    const result = applyConceptExtraction(db, SID, sources,
      [{ name: "抽取路由", type: "SKILL", description: "路由配置" }],
      [{ from: "抽取路由", to: "未声明的东西", type: "RELATES", instruction: "指向未声明概念" }],
    );

    expect(result).toEqual({ nodes: 1, edges: 0 });
  });

  it("re-observing the same relation does not duplicate the edge", () => {
    const entities = [
      { name: "抽取路由", type: "SKILL" as const, description: "路由配置" },
      { name: "本地 27B", type: "SKILL" as const, description: "本地模型" },
    ];
    const relations = [
      { from: "抽取路由", to: "本地 27B", type: "REQUIRES" as const, instruction: "依赖本地模型" },
    ];
    const first = applyConceptExtraction(db, SID, seedTurn(1), entities, relations);
    const second = applyConceptExtraction(db, SID, seedTurn(2), entities, relations);

    expect(first).toEqual({ nodes: 2, edges: 1 });
    // The second pass confirms the concepts and updates the edge in place.
    expect(second).toEqual({ nodes: 0, edges: 1 });
    const rows = db.prepare("SELECT * FROM gm_edges").all() as any[];
    expect(rows.length).toBe(1);
    expect(rows[0].instruction).toBe("依赖本地模型");
  });

  it("ignores a self-referencing relation", () => {
    const result = applyConceptExtraction(db, SID, seedTurn(1),
      [{ name: "抽取路由", type: "SKILL", description: "路由配置" }],
      [{ from: "抽取路由", to: "抽取路由", type: "RELATES", instruction: "自指" }],
    );
    expect(result).toEqual({ nodes: 1, edges: 0 });
  });

  it("is a no-op when the turn declared no concepts", () => {
    expect(applyConceptExtraction(db, SID, seedTurn(1), [], [])).toEqual({ nodes: 0, edges: 0 });
  });

  // TypeBox's runtime check enforces shape and length but not `enum`, so the
  // kind sets are re-checked at persistence time rather than trusted.
  it("skips a concept whose kind is outside the declared set", () => {
    const result = applyConceptExtraction(db, SID, seedTurn(1),
      [{ name: "抽取路由", type: "CONCEPT" as any, description: "未声明类型" }], []);
    expect(result).toEqual({ nodes: 0, edges: 0 });
    expect(findByName(db, "抽取路由")).toBeNull();
  });

  it("skips a relation whose kind is outside the declared set", () => {
    const result = applyConceptExtraction(db, SID, seedTurn(1),
      [
        { name: "甲", type: "SKILL", description: "甲" },
        { name: "乙", type: "SKILL", description: "乙" },
      ],
      [{ from: "甲", to: "乙", type: "LIKES" as any, instruction: "未声明关系" }]);
    expect(result).toEqual({ nodes: 2, edges: 0 });
  });
});
