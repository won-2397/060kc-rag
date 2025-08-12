// build_index.js (교체본)
import "dotenv/config";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import OpenAI from "openai";
import { loadJsonl } from "./utils/loadJsonl.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";

// 경로 고정 (cwd 영향 제거)
const QA_PATH  = path.join(__dirname, "data", "060kc_qa.jsonl");
const OUT_PATH = path.join(__dirname, "data", "embeddings.json");

// messages포맷/QA포맷 모두 지원 + 유효성 검사
function toQA(item, i) {
  // ① {"messages":[...]} 포맷
  if (Array.isArray(item?.messages) && item.messages.length >= 2) {
    const norm = m => (m?.role ?? "").toString().toLowerCase().trim();
    const userMsg = item.messages.find(m => norm(m) === "user") || item.messages[0];
    const asstMsg = item.messages.find(m => norm(m) === "assistant") || item.messages[1];

    const user = userMsg?.content?.toString().trim();
    const asst = asstMsg?.content?.toString().trim();

    if (user && asst) return { q: user, a: asst };
  }

  // ② {"question":"...","answer":"..."} 포맷
  if (typeof item?.question === "string" && typeof item?.answer === "string") {
    const q = item.question.trim();
    const a = item.answer.trim();
    if (q && a) return { q, a };
  }

  console.warn(`⚠️ invalid line (skip) at #${i + 1}`);
  return null;
}

// 임베딩 API 재시도(429/5xx) 포함
async function embedBatch(texts, attempt = 1) {
  try {
    const res = await client.embeddings.create({ model: EMBED_MODEL, input: texts });
    return res.data.map(d => d.embedding);
  } catch (err) {
    const retriable = err?.status === 429 || err?.status >= 500;
    if (retriable && attempt < 5) {
      const wait = 300 * Math.pow(2, attempt); // 0.3s → 2.4s
      console.warn(`⏳ retry ${attempt} in ${wait}ms (status=${err?.status})`);
      await new Promise(r => setTimeout(r, wait));
      return embedBatch(texts, attempt + 1);
    }
    throw err;
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) throw new Error("OPENAI_API_KEY missing");

  console.log("📂 Load:", QA_PATH);
  const itemsRaw = loadJsonl(QA_PATH);        // 기존 함수 그대로 사용
  const qa = itemsRaw.map(toQA).filter(Boolean);

  if (qa.length === 0) throw new Error("No valid QA lines. Check JSONL format.");
  console.log(`🧮 Total valid QAs: ${qa.length}`);

  const batchSize = 100;
  const index = [];

  for (let i = 0; i < qa.length; i += batchSize) {
    const batch = qa.slice(i, i + batchSize);
    const embeds = await embedBatch(batch.map(x => x.q));
    for (let j = 0; j < batch.length; j++) {
      index.push({ q: batch[j].q, a: batch[j].a, e: embeds[j] });
    }
    console.log(`✅ Embedded ${Math.min(i + batch.length, qa.length)} / ${qa.length}`);
  }

  fs.writeFileSync(OUT_PATH, JSON.stringify(index), "utf8");
  console.log("💾 Saved:", OUT_PATH);
}

main().catch(err => { console.error(err); process.exit(1); });
