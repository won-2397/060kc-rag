// server.js (RAG · ESM 최종본) - 개선된 검색 로직 통합
// 기능: embeddings.json 로드(q/a/e 또는 question/answer/vector 자동 지원) → POST /ask 응답
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.35); // 다시 높은 값으로

// 파일 경로 설정: EMB_PATH 없으면 data/embeddings.json 사용
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const EMB_PATH = process.env.EMB_PATH
  ? path.resolve(__dirname, process.env.EMB_PATH)
  : path.join(__dirname, "data", "embeddings.json");

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

    // 스키마 정규화: q/a/e → question/answer/vector/text
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

// ---- 유사도 계산 ----
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

// ---- 개선된 텍스트 유사도 계산 ----
function improvedTextScore(question, item) {
  const normalize = (text) => String(text || "").toLowerCase().trim();
  const Q = normalize(question);
  const T = `${normalize(item.question)} ${normalize(item.answer)} ${normalize(item.text)}`;
  
  if (!Q || !T) return 0;
  
  let score = 0;
  const questionWords = Q.split(/\s+/).filter(Boolean);
  
  // 1. 완전 일치 검사 (높은 점수)
  if (T.includes(Q)) {
    score += 0.8;
  }
  
  // 2. 개별 키워드 매칭 (중간 점수)
  let matchedWords = 0;
  for (const word of questionWords) {
    if (T.includes(word)) {
      matchedWords++;
      score += 0.4; // 키워드당 0.4점
    }
  }
  
  // 3. 키워드 매칭 비율 보너스
  const matchRatio = matchedWords / questionWords.length;
  score += matchRatio * 0.3;
  
  // 4. 유사 단어 검사 (동의어, 관련어)
  const synonyms = {
    '경마': ['경마', '경주', '마권', '경마장', '경마정보'],
    '증권': ['증권', '주식', '투자', '금융', '주가'],
    '운세': ['운세', '점', '사주', '타로', '운명'],
    '설치': ['설치', '설정', '구축', '개통', '연결'],
    '비용': ['비용', '요금', '가격', '료', '수수료', '비'],
    '060': ['060', '유료', '전화', '상담'],
    '할인': ['할인', '선불', '저렴', '싸게'],
    '후불': ['후불', '후불결제', '나중결제'],
    '상담': ['상담', '전화', '통화', '대화'],
    '서비스': ['서비스', '업무', '사업']
  };
  
  for (const [key, variants] of Object.entries(synonyms)) {
    if (questionWords.includes(key)) {
      for (const variant of variants) {
        if (T.includes(variant) && !questionWords.includes(variant)) {
          score += 0.2; // 유사어 매칭 보너스
        }
      }
    }
  }
  
  // 5. 문장 구조 유사성 (간단한 n-gram)
  const qBigrams = [];
  const tBigrams = [];
  
  for (let i = 0; i < questionWords.length - 1; i++) {
    qBigrams.push(questionWords[i] + questionWords[i + 1]);
  }
  
  const tWords = T.split(/\s+/);
  for (let i = 0; i < tWords.length - 1; i++) {
    tBigrams.push(tWords[i] + tWords[i + 1]);
  }
  
  const bigramMatches = qBigrams.filter(bg => tBigrams.includes(bg)).length;
  if (bigramMatches > 0) {
    score += bigramMatches * 0.15;
  }
  
  return Math.min(score, 1.0); // 최대 1.0점
}

// ---- 개선된 검색 함수 ----
function search({ question, qvec }) {
  const scored = EMB.map(item => {
    let score;
    
    if (Array.isArray(qvec) && Array.isArray(item.vector) && item.vector.length === EMB_DIM) {
      // 벡터가 있으면 코사인 유사도 사용
      score = cosineSimilarity(qvec, item.vector);
    } else {
      // 벡터가 없으면 개선된 텍스트 점수 사용
      score = improvedTextScore(question, item);
    }
    
    return { ...item, score };
  }).sort((a, b) => b.score - a.score);

  const top = scored.slice(0, 5);
  const bestScore = top[0]?.score ?? 0;
  
  // 적응적 임계값: 텍스트 검색은 낮은 임계값, 벡터 검색은 높은 임계값
  const adaptiveThreshold = Array.isArray(qvec) ? RAG_THRESHOLD : Math.max(0.3, RAG_THRESHOLD * 0.7);
  const found = bestScore >= adaptiveThreshold;
  
  const answer = found ? (top[0].answer || top[0].text || "자료에 없음") : "자료에 없음";
  const hits = top.map(({ id, question, answer, text, score }) => ({ id, question, answer, text, score }));
  
  return { answer, hits, bestScore, found, threshold: adaptiveThreshold };
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
