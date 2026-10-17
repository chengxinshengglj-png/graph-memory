import { Type } from "@sinclair/typebox";
import { Value } from "@sinclair/typebox/value";
const OutcomeSchema = Type.String({
    enum: ["completed", "partial", "failed", "informational", "unknown"],
    description: "Required final status of this completed dialogue turn.",
});
const NavigationTripleSchema = Type.Object({
    subject: Type.String({ minLength: 1, description: "Concrete searchable subject phrase." }),
    predicate: Type.String({ minLength: 1, description: "Short relation phrase." }),
    object: Type.String({ minLength: 1, description: "Concrete searchable object phrase." }),
}, { additionalProperties: false });
const ConceptEntitySchema = Type.Object({
    name: Type.String({
        minLength: 1,
        maxLength: 24,
        description: "Reusable noun phrase naming one concept; never a sentence, a status word, or a bare quantity.",
    }),
    type: Type.String({
        enum: ["EVENT", "SKILL", "TASK"],
        description: "Concept kind: EVENT happened, SKILL reusable capability, TASK goal or pending work.",
    }),
    description: Type.String({
        minLength: 1,
        description: "One sentence stating what this concept is, using only evidence from this turn.",
    }),
}, { additionalProperties: false });
const ConceptRelationSchema = Type.Object({
    from: Type.String({ minLength: 1, description: "Subject name; must equal one entities[].name value." }),
    to: Type.String({ minLength: 1, description: "Object name; must equal one entities[].name value." }),
    type: Type.String({
        enum: ["RELATES", "SUPERSEDES", "USED_SKILL", "SOLVED_BY", "REQUIRES", "PATCHES", "CONFLICTS_WITH"],
        description: "Declared relation kind between the two concepts.",
    }),
    instruction: Type.String({ minLength: 1, description: "Short natural-language statement of the relation." }),
}, { additionalProperties: false });
/**
 * Provider-facing extraction contract.
 *
 * Keep every required value at the top level. DSH's public ToolSchema does
 * not currently expose pi-ai constrained sampling, so custom OpenAI-compatible
 * routes receive an advisory schema. A flat shape and a single JSON `enum`
 * are followed more reliably across providers than nested required objects and
 * an `anyOf` of literals, while the runtime still validates every field.
 */
export const GRAPH_EXTRACTION_SCHEMA = Type.Object({
    summary: Type.String({
        minLength: 1,
        description: "Required one-sentence, self-contained summary of the current dialogue turn.",
    }),
    outcome: OutcomeSchema,
    triples: Type.Array(NavigationTripleSchema, {
        description: "Required SPO relations derived only from summary; use an empty array when none are explicit.",
    }),
    entities: Type.Optional(Type.Array(ConceptEntitySchema, {
        description: "Optional durable concept list for this turn; omit or use [] when the turn yields no reusable concept.",
    })),
    relations: Type.Optional(Type.Array(ConceptRelationSchema, {
        description: "Optional typed links between entries of entities; both endpoints must be declared concepts.",
    })),
}, { additionalProperties: false });
export const GRAPH_EXTRACTION_TOOL_NAME = "submit_result";
export const GRAPH_EXTRACTION_TOOL = Object.freeze({
    name: GRAPH_EXTRACTION_TOOL_NAME,
    description: "Submit exactly the three required fields summary, outcome, and triples. Derive zero or more subject-predicate-object triples only from the summary; triples must be [] when no relation is explicit. Optionally add entities and relations when the turn names reusable concepts. Emit no text.",
    parameters: GRAPH_EXTRACTION_SCHEMA,
});
/** Fail closed before normalization or persistence when the contract is incomplete. */
export function assertGraphExtractionContract(value) {
    if (Value.Check(GRAPH_EXTRACTION_SCHEMA, value))
        return;
    const errors = Array.from(Value.Errors(GRAPH_EXTRACTION_SCHEMA, value))
        .slice(0, 3)
        .map(error => `${error.path || "/"}: ${error.message}`)
        .join("; ");
    throw new TypeError(`graph extraction contract violation${errors ? `: ${errors}` : ""}`);
}
