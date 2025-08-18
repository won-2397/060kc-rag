// RAG server.js (Render 배포용)
import "dotenv/config";
import fs from "fs";
import express from "express";
import OpenAI from "openai";
import cors from "cors";
import { cosineSimilarity } from "./utils/similarity.js";

console.log("🚀 060KC RAG boot :: /ask route with hits.text");

const app = express();

// CORS
app.use(cors({
  origin: [
    "https://www.060kc.com",
    "https://060kc.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
  ],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"],
  credentials: false
}));
app.options("*", cors());

app.use(express.json({ limit: "1mb" }));
app.use((_,res,next)=>{ res.set("Cache-Control","no-store"); next(); });

// ENV
if (!process.env.OPENAI_API_KEY) {
  console.error("❌ OPENAI_API_KEY missing");
  process.exit(1);
}
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const PORT = Number(process.env.PORT);
if (!PORT) {
  console.error("❌ PORT env missing (Render는 PORT 필수)");
  process.exit(1);
}
const THRESHOLD = Number(process.env.THRESHOLD || 0.78);

// Health
app.get("/health", (_req, res) => res.json({ ok: true, ts: Date.now() }));

// Meta (점검용)
let index = [];
try {
  index = JSON.parse(fs.readFileSync("./data/embeddings.json", "utf8"));
  console.log("[EMB] loaded", { count: index.length, dim: index[0]?.e?.length });
} catch (err) {
  console.error("❌ embeddings.json load fail:", err.message);
  index = [];
}
app.get("/meta", (_req,res)=>{
  res.json({
    count: index.length,
    dim: index[0]?.e?.length || 0,
    threshold: THRESHOLD,
    embed_model: EMBED_MODEL
  });
});

// 유틸
async function embed(text) {
  const r = await client.embeddings.create({ model: EMBED_MODEL, input: text });
  return r.data[0].embedding;
}

function topK(qv, k = 15) {
  const scored = index.map(item => {
    const score = cosineSimilarity(qv, item.e);
    return {
      q: item.q,
      a: item.a,
      // ⬇️ 프론트가 읽는 본문 키 (필수)
      text: `Q: ${item.q}\nA: ${item.a}`,
      score,
      // 메타(있으면 사용)
      doc: item.doc || "faq",
      section: item.section || "일반",
      rev: item.rev || null
    };
  });
  return scored.sort((a,b)=>b.score-a.score).slice(0, k);
}

// /ask
app.post("/ask", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    if (!question) return res.status(400).json({ error: "question required" });

    if (!index.length) {
      return res.json({ answer: "자료에 없음", hits: [], bestScore: 0, found: false });
    }

    const qv = await embed(question);
    const hits = topK(qv, 15);
    const best = hits[0];
    const bestScore = best?.score ?? 0;

    if (!best || bestScore < THRESHOLD) {
      // 게이트 미달 시 hits 비워서 프론트가 확실히 사람이음으로
      return res.json({ answer: "자료에 없음", hits: [], bestScore, found: false });
    }

    // 참고용 answer (프론트는 hits[].text로 컨텍스트 구성)
    res.json({ answer: best.a, hits, bestScore, found: true });
  } catch (e) {
    console.error("❌ /ask error:", e?.message || e);
    res.status(500).json({ error: "server error" });
  }
});

// Listen
app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ RAG ONLINE on 0.0.0.0:${PORT} (TH=${THRESHOLD})`);
});
