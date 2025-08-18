// server.js (ESM)
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 10000;
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.35);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMB_PATH = process.env.EMB_PATH || path.join(__dirname, "embeddings.json");

app.use(cors({
  origin: ["https://www.060kc.com","https://060kc.com","http://localhost:8080","http://127.0.0.1:8080"],
  methods: ["POST","GET","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"]
}));
app.use(express.json({ limit: "2mb" }));

let EMB = []; let EMB_DIM = 0;

async function loadEmbeddings() {
  try {
    const raw = await fs.readFile(EMB_PATH, "utf-8");
    EMB = JSON.parse(raw);
    const first = EMB.find(x => Array.isArray(x.vector))?.vector || [];
    EMB_DIM = first.length || 0;
    console.log(`[EMB] loaded { count: ${EMB.length}, dim: ${EMB_DIM} }`);
  } catch (e) {
    console.error("❌ failed to load embeddings:", e.message || e);
    EMB = []; EMB_DIM = 0;
  }
}

function cosineSimilarity(a, b) {
  if (!a?.length || !b?.length || a.length !== b.length) return 0;
  let dot=0, na=0, nb=0;
  for (let i=0;i<a.length;i++){ const x=a[i]||0, y=b[i]||0; dot+=x*y; na+=x*x; nb+=y*y; }
  const den = Math.sqrt(na)*Math.sqrt(nb);
  return den ? dot/den : 0;
}
const cheapTextScore = (q, it) => {
  const s = v => String(v||"").toLowerCase();
  const Q = s(q), T = s(it.question)+" "+s(it.text);
  return Math.min(Q.split(/\s+/).filter(Boolean).reduce((acc,k)=>acc+(T.includes(k)?0.1:0),0), 0.6);
};

function search({ question, qvec }) {
  const scored = EMB.map(it => {
    const sim = (Array.isArray(qvec) && Array.isArray(it.vector) && it.vector.length===EMB_DIM)
      ? cosineSimilarity(qvec, it.vector)
      : cheapTextScore(question, it);
    return { ...it, score: sim };
  }).sort((a,b)=>b.score-a.score);

  const top = scored.slice(0,5);
  const bestScore = top[0]?.score ?? 0;
  const found = bestScore >= RAG_THRESHOLD;
  const answer = found ? (top[0].answer || top[0].text || "자료에 없음") : "자료에 없음";
  const hits = top.map(({ id, question, answer, text, score }) => ({ id, question, answer, text, score }));
  return { answer, hits, bestScore, found };
}

app.get("/health", (_req,res)=>res.json({ ok:true, ts: Date.now() }));

// 표준 엔드포인트: gpt-server가 기대하는 스키마
app.post("/ask", async (req,res)=>{
  try{
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector;
    if (!question) return res.status(400).json({ error: "question required" });
    const r = search({ question, qvec });
    res.json(r); // { answer, hits, bestScore, found }
  }catch(e){
    console.error("[/ask] error:", e.message || e);
    res.status(500).json({ error: "RAG error" });
  }
});

await loadEmbeddings();
app.listen(PORT,"0.0.0.0",()=>console.log(`✅ RAG ONLINE on 0.0.0.0:${PORT} (TH=${RAG_THRESHOLD})`));
