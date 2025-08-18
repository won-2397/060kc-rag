// RAG server.js — CommonJS (Render/사내 공통)
// 기능: embeddings.json 로드 → POST /ask 로 질의 응답
const express = require("express");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
require("dotenv").config();

const app = express();
const PORT = Number(process.env.PORT) || 10000;
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.35);

// ---- 1) 로딩: 임베딩 파일 (기본: ./embeddings.json, 환경변수로 변경 가능) ----
const EMB_PATH = process.env.EMB_PATH || path.join(process.cwd(), "embeddings.json");
/**
 * 기대 구조 예시 (배열):
 * [
 *   {
 *     "id": "q1",
 *     "question": "설치비가 있나요?",
 *     "answer": "초기 설치비는 없으며, 최초 1회에 한해 프로그램비를 납부하셔야 합니다.",
 *     "text": "설치비/프로그램비 안내 ...",         // 검색용 본문
 *     "vector": [0.0123, -0.0045, ...]             // 길이 1536
 *   },
 *   ...
 * ]
 */
let EMB = [];
let EMB_DIM = 0;

function safeLoadEmbeddings() {
  if (!fs.existsSync(EMB_PATH)) {
    console.error(`❌ embeddings file not found: ${EMB_PATH}`);
    EMB = [];
    EMB_DIM = 0;
    return;
  }
  try {
    const raw = fs.readFileSync(EMB_PATH, "utf-8");
    const arr = JSON.parse(raw);
    EMB = Array.isArray(arr) ? arr : [];
    const firstVec = EMB.find(x => Array.isArray(x.vector))?.vector || [];
    EMB_DIM = firstVec.length || 0;
    console.log(`[EMB] loaded { count: ${EMB.length}, dim: ${EMB_DIM} }`);
  } catch (e) {
    console.error("❌ failed to load embeddings:", e?.message || e);
    EMB = [];
    EMB_DIM = 0;
  }
}

// ---- 2) 유틸: 코사인 유사도 ----
function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length || a.length === 0) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] || 0;
    const y = b[i] || 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom ? (dot / denom) : 0;
}

// ---- 3) (옵션) 쿼리 벡터화: 프리컴퓨팅/외부 임베딩 없이 동작하려면 불가 →
//      실제 운영에선 질문 임베딩을 미리 생성해서 전달하거나, 여기서 OpenAI로 임베딩 생성 필요.
//      임시로는 '질문' 필드와 텍스트의 매칭 점수를 사용 (간이 매칭) 또는 질문 벡터를 함께 보내도록 설계.
//      아래는 간이 전략: 질문 문자열과 항목 question/text 간 단순 키워드 겹침 점수 + (벡터가 있다면 코사인)
function cheapTextScore(q, item) {
  const s = (txt) => String(txt || "").toLowerCase();
  const Q = s(q);
  const T = s(item.question) + " " + s(item.text);
  let score = 0;
  // 매우 단순 키워드 매칭
  const keys = Q.split(/\s+/).filter(Boolean);
  for (const k of keys) if (T.includes(k)) score += 0.1;
  return Math.min(score, 0.6); // 캡
}

/**
 * 질문 임베딩을 클라이언트가 함께 보낼 수 있도록 확장:
 * - body.vector: [..] 이 있으면 코사인 유사도 사용
 */
function searchBySimilarity({ question, qvec }) {
  const scored = EMB.map((it) => {
    let sim = 0;
    if (Array.isArray(qvec) && Array.isArray(it.vector) && it.vector.length && EMB_DIM) {
      sim = cosineSimilarity(qvec, it.vector);
    } else {
      // 임시: 텍스트 스코어
      sim = cheapTextScore(question, it);
    }
    return { ...it, score: sim };
  }).sort((a, b) => b.score - a.score);

  const top = scored.slice(0, 5);
  const best = top[0];
  const bestScore = best?.score ?? 0;
  const found = bestScore >= RAG_THRESHOLD;

  // answer 우선순위: answer 필드 > text
  const answer = found ? (best.answer || best.text || "자료에 없음") : "자료에 없음";

  // hits 형태 간단화
  const hits = top.map(({ id, question, answer, text, score }) => ({
    id, question, answer, text, score
  }));

  return { answer, hits, bestScore, found };
}

// ---- 4) 서버/라우팅 ----
app.use(cors({
  origin: [
    "https://www.060kc.com",
    "https://060kc.com",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
  ],
  methods: ["POST","GET","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"],
}));
app.use(express.json({ limit: "2mb" }));

app.get("/health", (_req, res) => res.json({ ok: true, ts: Date.now() }));

// 표준 RAG 엔드포인트 — 웹 서버가 기대하는 스키마로 응답
app.post("/ask", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector; // 선택: 질문 임베딩 벡터를 클라이언트가 같이 줄 수도 있음
    if (!question) return res.status(400).json({ error: "question required" });

    const { answer, hits, bestScore, found } = searchBySimilarity({ question, qvec });
    return res.json({ answer, hits, bestScore, found });
  } catch (e) {
    console.error("[/ask] error:", e?.message || e);
    return res.status(500).json({ error: "RAG error" });
  }
});

// (선택) 디버그: 상위 N개 미리보기
app.post("/ask/debug", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector;
    const N = Number(req.body?.top || 5);
    if (!question) return res.status(400).json({ error: "question required" });

    const { hits, bestScore, found } = searchBySimilarity({ question, qvec });
    return res.json({ top: hits.slice(0, N), bestScore, found, count: EMB.length });
  } catch (e) {
    console.error("[/ask/debug] error:", e?.message || e);
    return res.status(500).json({ error: "RAG error" });
  }
});

// ---- 부팅 ----
safeLoadEmbeddings();
app.listen(PORT, "0.0.0.0", () => {
  console.log(`✅ RAG ONLINE on 0.0.0.0:${PORT} (TH=${RAG_THRESHOLD})`);
});
