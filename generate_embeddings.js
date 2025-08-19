// generate_embeddings.js  (RAG 루트에 위치)
// 사용법:
//   mac/linux) OPENAI_API_KEY=... node generate_embeddings.js data/060kc_qa.jsonl embeddings.json
//   windows)   $env:OPENAI_API_KEY="..." ; node generate_embeddings.js data/060kc_qa.jsonl embeddings.json
//
// ENV 옵션:
//   EMBED_MODEL=text-embedding-3-small
//   EMBED_MODE=QA | Q      // 기본: QA (Q+A 합쳐서 임베딩)
//   EMBED_BATCH=100

import fs from "fs";
import path from "path";
import "dotenv/config";
import OpenAI from "openai";

if (!process.env.OPENAI_API_KEY) {
  throw new Error("OPENAI_API_KEY missing in .env");
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const inPath  = process.argv[2] || path.join("data", "060kc_qa.jsonl");
const outPath = process.argv[3] || path.join("embeddings.json"); // ✅ 루트에 저장!
const MODEL   = process.env.EMBED_MODEL || "text-embedding-3-small";
const MODE    = (process.env.EMBED_MODE || "QA").toUpperCase();  // "QA" | "Q"
const BATCH   = Number(process.env.EMBED_BATCH || 100);

function loadJsonl(p) {
  const lines = fs.readFileSync(p, "utf8")
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);
  return lines.map(l => JSON.parse(l));
}

function norm(item) {
  const q =
    item?.question ??
    item?.q ??
    item?.messages?.find(m => m.role === "user")?.content ??
    "";
  const a =
    item?.answer ??
    item?.a ??
    item?.messages?.find(m => m.role === "assistant")?.content ??
    "";
  // Q/A 둘 다 있는 것이 이상적이지만, 최소 q만 있으면 보관
  return { q: String(q || "").trim(), a: String(a || "").trim() };
}

function toEmbedText({ q, a }) {
  if (MODE === "Q" || !a) return q;              // 질문만
  return `Q: ${q}\nA: ${a}`;                      // ✅ 기본: Q+A 합침
}

async function embedBatch(texts) {
  const res = await client.embeddings.create({ model: MODEL, input: texts });
  return res.data.map(d => d.embedding);
}

async function main() {
  console.log("📂 Load:", inPath);
  const raws = loadJsonl(inPath);
  const qa0  = raws.map(norm).filter(x => x.q);   // q는 필수, a는 선택

  // ✅ 질문(q) 기준 중복 제거
  const seen = new Set();
  const qa = [];
  for (const it of qa0) {
    const k = it.q.replace(/\s+/g, " ").toLowerCase();
    if (seen.has(k)) continue;
    seen.add(k);
    qa.push(it);
  }

  console.log(`🧮 Total QAs: ${qa.length} (deduped)`);

  const index = [];
  for (let i = 0; i < qa.length; i += BATCH) {
    const batch = qa.slice(i, i + BATCH);
    const inputs = batch.map(toEmbedText);
    const embeds = await embedBatch(inputs);

    // 차원 경고(모델 변경 시 바로 알림)
    const dim = embeds[0]?.length ?? 0;
    if (dim && dim !== 1536) {
      console.warn(`⚠️  Embedding dim=${dim} (expected 1536 for ${MODEL})`);
    }

    for (let j = 0; j < batch.length; j++) {
      index.push({ q: batch[j].q, a: batch[j].a, e: embeds[j] });
    }
    console.log(`✅ Embedded ${Math.min(i + batch.length, qa.length)} / ${qa.length}`);
  }

  fs.writeFileSync(outPath, JSON.stringify(index, null, 2), "utf8");
  console.log("💾 Saved:", outPath);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
