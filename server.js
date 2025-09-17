// server.js (RAG · ESM 최종본) - 포트 3000 수정
// 기능: embeddings.json 로드(q/a/e 또는 question/answer/vector 자동 지원) → POST /ask 응답
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000; // ✅ 3000으로 변경
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.35);

// 파일 경로 설정: EMB_PATH 없으면 레포 루트의 embeddings.json 사용
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMB_PATH = process.env.EMB_PATH
  ? path.resolve(__dirname, process.env.EMB_PATH)
  : path.join(__dirname, "embeddings.json");

console.log("[BOOT] EMB_PATH =", EMB_PATH);

// CORS
app.use(cors({
  origin: [
    "https://www.060kc.com",
    "https://060kc.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080"
  ],
  methods: ["POST", "GET", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"]
}));
app.use(express.json({ limit: "2mb" }));

let EMB = [];
let EMB_DIM = 0;

// ---- 임베딩 로드(+스키마 정규화) ----
async function loadEmbeddings() {
  try {
    const raw = await fs.readFile(EMB_PATH, "utf-8");
    EMB = JSON.parse(raw);

    // ⬇️ 스키마 정규화: q/a/e → question/answer/vector/text
    EMB = (Array.isArray(EMB) ? EMB : []).map((it, idx) => {
      const question = it.question ?? it.q ?? "";
      const answer   = it.answer   ?? it.a ?? "";
      const text     = it.text     ?? `${question} ${answer}`;
      const vector   = it.vector   ?? it.e ?? [];
      const id       = it.id       ?? `q${idx + 1}`;
      return { id, question, answer, text, vector };
    });

    const first = EMB.find(x => Array.isArray(x.vector))?.vector || [];
    EMB_DIM = first.length || 0;

    console.log(`[EMB] loaded { count: ${EMB.length}, dim: ${EMB_DIM} }`);
  } catch (e) {
    console.error("❌ failed to load embeddings:", e.message || e);
    EMB = [];
    EMB_DIM = 0;
  }
}

// ---- 유사도 / 간이 점수 ----
function cosineSimilarity(a, b) {
  if (!a?.length || !b?.length || a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0, y = b[i] ?? 0;
    dot += x * y; na += x * x; nb += y * y;
  }
  const den = Math.sqrt(na) * Math.sqrt(nb);
  return den ? dot / den : 0;
}

const cheapTextScore = (q, it) => {
  const s = v => String(v || "").toLowerCase();
  const Q = s(q), T = `${s(it.question)} ${s(it.text)}`;
  const keys = Q.split(/\s+/).filter(Boolean);
  const score = keys.reduce((acc, k) => acc + (T.includes(k) ? 0.1 : 0), 0);
  return Math.min(score, 0.6); // 상한
};

// ---- 검색 ----
function search({ question, qvec }) {
  const scored = EMB.map(it => {
    const sim = (Array.isArray(qvec) && Array.isArray(it.vector) && it.vector.length === EMB_DIM)
      ? cosineSimilarity(qvec, it.vector)
      : cheapTextScore(question, it);
    return { ...it, score: sim };
  }).sort((a, b) => b.score - a.score);

  const top = scored.slice(0, 5);
  const bestScore = top[0]?.score ?? 0;
  const found = bestScore >= RAG_THRESHOLD;
  const answer = found ? (top[0].answer || top[0].text || "자료에 없음") : "자료에 없음";

  const hits = top.map(({ id, question, answer, text, score }) => ({ id, question, answer, text, score }));
  return { answer, hits, bestScore, found };
}

// ---- 라우트 ----
app.get("/health", (_req, res) => res.json({ ok: true, ts: Date.now() }));

// 표준 엔드포인트: gpt-server가 기대하는 스키마
app.post("/ask", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector; // 옵션: 질문 임베딩 벡터
    if (!question) return res.status(400).json({ error: "question required" });

    const result = search({ question, qvec });
    return res.json(result); // { answer, hits, bestScore, found }
  } catch (e) {
    console.error("[/ask] error:", e.message || e);
    return res.status(500).json({ error: "RAG error" });
  }
});

// 디버그용(선택): 상위 N개 미리보기
app.post("/ask/debug", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector;
    const N = Math.max(1, Math.min(20, Number(req.body?.top || 5)));
    if (!question) return res.status(400).json({ error: "question required" });

    const { hits, bestScore, found } = search({ question, qvec });
    return res.json({ top: hits.slice(0, N), bestScore, found, count: EMB.length, dim: EMB_DIM });
  } catch (e) {
    console.error("[/ask/debug] error:", e.message || e);
    return res.status(500).json({ error: "RAG error" });
  }
});

// ---- 부팅 ----
await loadEmbeddings();
app.listen(PORT, "0.0.0.0", () =>
  console.log(`✅ RAG ONLINE on 0.0.0.0:${PORT} (TH=${RAG_THRESHOLD})`)
);
