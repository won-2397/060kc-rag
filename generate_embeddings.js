// C:\060kc-rag\generate_embeddings.js
import fs from "fs";
import path from "path";
import "dotenv/config";
import OpenAI from "openai";

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const dataPath = path.join("data", "060kc_qa.jsonl");
const outPath  = path.join("data", "embeddings.json");
const MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";

function loadJsonl(p) {
  const lines = fs.readFileSync(p, "utf8")
    .split("\n")
    .map(l => l.trim())
    .filter(Boolean);
  return lines.map(l => JSON.parse(l));
}

async function embedBatch(texts) {
  const res = await client.embeddings.create({ model: MODEL, input: texts });
  return res.data.map(d => d.embedding);
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY missing in .env");
  }
  console.log("📂 Load:", dataPath);
  const items = loadJsonl(dataPath);

  // JSONL 구조: {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]}
  const qa = items.map((it, i) => {
    const user = it?.messages?.find(m => m.role === "user")?.content;
    const assistant = it?.messages?.find(m => m.role === "assistant")?.content;
    if (!user || !assistant) {
      console.warn(`⚠️  Line ${i+1}: invalid structure, skipping`);
    }
    return { q: user, a: assistant };
  }).filter(x => x.q && x.a);

  console.log(`🧮 Total QAs: ${qa.length}`);

  const batchSize = 100;
  const index = [];
  for (let i = 0; i < qa.length; i += batchSize) {
    const batch = qa.slice(i, i + batchSize);
    const embeds = await embedBatch(batch.map(x => x.q));
    for (let j = 0; j < batch.length; j++) {
      index.push({ q: batch[j].q, a: batch[j].a, e: embeds[j] });
    }
    console.log(`✅ Embedded ${Math.min(i + batch.length, qa.length)} / ${qa.length}`);
    // (필요시) await new Promise(r => setTimeout(r, 100));
  }

  fs.writeFileSync(outPath, JSON.stringify(index), "utf8");
  console.log("💾 Saved:", outPath);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
