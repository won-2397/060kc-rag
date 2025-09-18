// server.js (RAG · ESM 최종본) - 포트 3000 수정
// 기능: embeddings.json 로드(q/a/e 또는 question/answer/vector 자동 지원) → POST /ask 응답
// 추가: 키워드 기반 안내 시스템
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000; // ✅ 3000으로 변경
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.25);

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

// ---- 키워드 기반 안내 시스템 ----
const KEYWORD_GUIDE_MAP = {
  "060": {
    keywords: ["060", "060서비스", "프리미엄", "통화"],
    response: "네. 060 서비스의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "경마": {
    keywords: ["경마", "경마장", "경주", "베팅", "마권"],
    response: "네. 경마의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "운세": {
    keywords: ["운세", "점", "사주", "타로", "신점"],
    response: "네. 운세의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "게임": {
    keywords: ["게임", "온라인게임", "모바일게임", "카지노"],
    response: "네. 게임의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "채팅": {
    keywords: ["채팅", "대화", "톡", "메신저"],
    response: "네. 채팅 서비스의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "결제": {
    keywords: ["결제", "요금", "비용", "수수료", "가격", "얼마"],
    response: "네. 결제 관련해서 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "가입": {
    keywords: ["가입", "회원가입", "등록", "신청"],
    response: "네. 가입 절차의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  },
  "이용": {
    keywords: ["이용", "사용", "이용방법", "사용법"],
    response: "네. 이용 방법의 어떤 점이 궁금하신가요? 보다 정확히 질문해주시면 자세히 답변해드리겠습니다."
  }
};

// 키워드 가이드 체크 함수
function checkKeywordGuide(question) {
  const normalizedQ = question.toLowerCase().trim();
  const words = normalizedQ.split(/\s+/);
  
  // 단어가 1-2개이고 짧은 질문인 경우에만 키워드 가이드 적용
  if (words.length <= 2 && question.length <= 10) {
    for (const [category, config] of Object.entries(KEYWORD_GUIDE_MAP)) {
      for (const keyword of config.keywords) {
        if (normalizedQ.includes(keyword.toLowerCase())) {
          return {
            isKeywordGuide: true,
            category,
            response: config.response
          };
        }
      }
    }
  }
  
  return { isKeywordGuide: false };
}

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
  const Q = s(q);
  const question = s(it.question);
  const answer = s(it.answer);
  
  // 텍스트 정규화 (기존 코드 유지)
  const normalize = (text) => {
    return text
      .replace(/잇나여|잇나요|있냐|잇냐|잇나|있나\?/g, '있나요')
      .replace(/뭐에요|뭐야|뭔가요|뭔가여|머에요/g, '무엇인가요')
      .replace(/어떻게요|어케|어떠케|어떻케/g, '어떻게')
      .replace(/얼마에요|얼마야|얼마냐|얼마인가요/g, '얼마')
      .replace(/이용하고싶다|이용하고싶어요|사용하고싶다/g, '이용방법')
      .replace(/\s+/g, ' ')
      .trim();
  };
  
  const normalizedQ = normalize(Q);
  const normalizedQuestion = normalize(question);
  
  let score = 0;
  
  // 1. 질문 의도 분류
  const getQuestionIntent = (text) => {
    if (/이용|사용|신청|시작|운영|개통/.test(text)) return 'usage';
    if (/자격증|서약서|필요|조건|요건/.test(text)) return 'requirement';
    if (/비용|요금|수수료|가격|얼마/.test(text)) return 'cost';
    if (/절차|방법|어떻게|순서/.test(text)) return 'process';
    return 'general';
  };
  
  const qIntent = getQuestionIntent(normalizedQ);
  const targetIntent = getQuestionIntent(normalizedQuestion);
  
  // 2. 의도가 다르면 점수 크게 감점
  if (qIntent !== 'general' && targetIntent !== 'general' && qIntent !== targetIntent) {
    score -= 0.5;
  }
  
  // 3. 핵심 키워드 매칭
  const qKeywords = normalizedQ.split(/\s+/).filter(w => w.length >= 2);
  qKeywords.forEach(keyword => {
    if (normalizedQuestion.includes(keyword)) score += 0.3;
    if (answer.includes(keyword)) score += 0.2;
  });
  
  // 4. 완전 일치 보너스
  if (normalizedQuestion.includes(normalizedQ) || normalizedQ.includes(normalizedQuestion)) {
    score += 0.4;
  }
  
  // 5. 질문 패턴 매칭
  const questionPatterns = ['있나요', '무엇인가요', '어떻게', '얼마', '이용방법'];
  const qHasPattern = questionPatterns.some(pattern => normalizedQ.includes(pattern));
  const targetHasPattern = questionPatterns.some(pattern => normalizedQuestion.includes(pattern));
  
  if (qHasPattern && targetHasPattern) score += 0.2;
  
  return Math.max(0, Math.min(score, 1.0)); // 음수 방지
};

// ---- 검색 ----
function search({ question, qvec }) {
  // 먼저 키워드 가이드 체크
  const keywordGuide = checkKeywordGuide(question);
  if (keywordGuide.isKeywordGuide) {
    return {
      answer: keywordGuide.response,
      hits: [],
      bestScore: 1.0,
      found: true,
      isKeywordGuide: true,
      category: keywordGuide.category
    };
  }

  // 기존 검색 로직
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
  return { answer, hits, bestScore, found, isKeywordGuide: false };
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
    return res.json(result); // { answer, hits, bestScore, found, isKeywordGuide?, category? }
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

    const { hits, bestScore, found, isKeywordGuide, category } = search({ question, qvec });
    return res.json({ 
      top: hits.slice(0, N), 
      bestScore, 
      found, 
      count: EMB.length, 
      dim: EMB_DIM,
      isKeywordGuide,
      category
    });
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
