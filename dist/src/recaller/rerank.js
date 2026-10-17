/**
 * Cross-encoder reranking over recall candidates.
 *
 * The integration shape (best-effort, opt-in, a failed reranker may only
 * disable reranking and never search itself) follows slow-stack/mneme's
 * reranker wiring (MIT License), https://github.com/slow-stack/mneme. The
 * transport is different: that project loads an ONNX model in-process, while
 * this one talks to an HTTP endpoint so the model can stay out of the agent
 * process and out of VRAM.
 *
 * Why reranking is worth a network hop: a bi-encoder embeds query and document
 * separately, so it cannot see term-level interaction. Measured on this
 * machine's corpus, a cross-encoder separated a true match at 0.9974 from the
 * next candidate at 0.3435 — an ordering the embedding scores did not produce.
 */
/**
 * Build a reranker backed by an HTTP endpoint that speaks the
 * `{ query, documents } -> { results: [{ index, relevance_score }] }` shape
 * (Jina / Cohere compatible, also what llama.cpp-style local servers expose).
 *
 * Every failure mode returns null rather than throwing: a reranker outage, a
 * malformed payload, a mismatched result count, or a timeout all mean "use the
 * embedding order", never "fail the recall".
 */
export function createHttpReranker(options) {
    const { endpoint, model, timeoutMs = 1500, headers } = options;
    return async (query, documents) => {
        if (!documents.length)
            return [];
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), timeoutMs);
        try {
            const body = { query, documents };
            if (model)
                body.model = model;
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "content-type": "application/json", ...headers },
                body: JSON.stringify(body),
                signal: controller.signal,
            });
            if (!response.ok)
                return null;
            const payload = await response.json();
            const results = Array.isArray(payload?.results) ? payload.results : [];
            if (results.length !== documents.length)
                return null;
            const scores = new Array(documents.length).fill(0);
            for (const item of results) {
                const index = Number(item?.index);
                const score = Number(item?.relevance_score ?? item?.score);
                if (!Number.isInteger(index) || index < 0 || index >= scores.length)
                    return null;
                if (!Number.isFinite(score))
                    return null;
                scores[index] = score;
            }
            return scores;
        }
        catch {
            return null;
        }
        finally {
            clearTimeout(timer);
        }
    };
}
