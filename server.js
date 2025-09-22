// server.js (RAG · ESM 최종본) - 검색 정확도 향상 버전
// 기능: embeddings.json 로드 + 동적 키워드 가이드 + 검색 최적화
import express from "express";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import dotenv from "dotenv";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;
const RAG_THRESHOLD = Number(process.env.RAG_THRESHOLD || 0.25);

// 파일 경로 설정
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

// ---- 검색 최적화 시스템 ----

// 동의어/유의어 매핑
const SYNONYM_MAP = {
  "이용료": ["요금", "비용", "가격", "수수료", "금액"],
  "신청": ["가입", "등록", "개설", "개통"],
  "방법": ["절차", "과정", "순서", "단계"],
  "문의": ["질문", "상담", "연락"],
  "서비스": ["업무", "상품", "제품"],
  "취소": ["해지", "중단", "종료"],
  "변경": ["수정", "조정", "바꾸기"]
};

// 오타 보정 매핑
const TYPO_CORRECTION = {
  "잇나여": "있나요", "잇나요": "있나요", "있냐": "있나요", "잇냐": "있나요", "잇나": "있나요",
  "뭐에요": "무엇인가요", "뭐야": "무엇인가요", "뭔가요": "무엇인가요", "뭔가여": "무엇인가요", "머에요": "무엇인가요",
  "어떻게요": "어떻게", "어케": "어떻게", "어떠케": "어떻게", "어떻케": "어떻게",
  "얼마에요": "얼마", "얼마야": "얼마", "얼마냐": "얼마", "얼마인가요": "얼마",
  "이용하고싶다": "이용방법", "이용하고싶어요": "이용방법", "사용하고싶다": "이용방법"
};

// 불용어 목록
const STOP_WORDS = ["그", "저", "이", "그런데", "혹시", "그리고", "그래서", "그런", "이런", "저런"];

// 핵심 키워드 가중치
const KEYWORD_WEIGHTS = {
  "060": 2.0, "프리미엄": 2.0, "경마": 2.0, "운세": 2.0, "게임": 2.0,
  "요금": 1.5, "비용": 1.5, "가격": 1.5, "얼마": 1.5,
  "이용": 1.5, "사용": 1.5, "신청": 1.5, "가입": 1.5,
  "방법": 1.3, "절차": 1.3, "어떻게": 1.3
};

// 텍스트 전처리 함수
function preprocessText(text) {
  let processed = text.toLowerCase().trim();
  
  // 1. 오타 보정
  Object.entries(TYPO_CORRECTION).forEach(([typo, correct]) => {
    processed = processed.replace(new RegExp(typo, 'g'), correct);
  });
  
  // 2. 동의어 확장
  Object.entries(SYNONYM_MAP).forEach(([main, synonyms]) => {
    synonyms.forEach(synonym => {
      if (processed.includes(synonym)) {
        processed += ` ${main}`;
      }
    });
  });
  
  // 3. 불용어 제거
  const words = processed.split(/\s+/);
  const filteredWords = words.filter(word => !STOP_WORDS.includes(word) && word.length > 1);
  
  return {
    original: text,
    processed: filteredWords.join(' '),
    keywords: filteredWords.filter(word => word.length >= 2)
  };
}

// 키워드 가이드 체크 함수 (질문/문장 구분)
function checkKeywordGuide(question) {
  const normalizedQ = question.trim();
  
  // 질문 형태 감지 - 더 엄격한 패턴
  const questionMarkers = [
    /\?$/, // 물음표로 끝남
    /어떤/, /어떻게/, /얼마/, /언제/, /어디/, /왜/, /누구/, /무엇/, /몇/, /어느/,
    /뭐야/, /뭐에요/, /뭔가요/, /뭐예요/,
    /인가요/, /있나요/, /하나요/, /나요$/, /까요$/,
    /는$/, /을$/, /를$/, /이$/, /가$/  // 조사로 끝남
  ];
  
  const isQuestion = questionMarkers.some(pattern => pattern.test(normalizedQ));
  
  console.log(`[DEBUG] Question: "${normalizedQ}", isQuestion: ${isQuestion}`);
  
  // 질문이 아닌 단순 키워드/명사구인 경우
  if (!isQuestion) {
    const keyword = normalizedQ;
    const response = `네. ${keyword}에 대해서 어떤것이 궁금하신가요? 저는 챗봇이기 때문에 구체적인 문장을 작성해주셔야 올바른 대답을 해드릴 수 있습니다. 예) 060 케이씨는 어떤 회사인가요?`;
    
    console.log(`[DEBUG] Keyword guide triggered for: "${keyword}"`);
    
    return {
      isKeywordGuide: true,
      keyword: keyword,
      response: response
    };
  }
  
  return { isKeywordGuide: false };
}

// ---- 임베딩 로드 ----
async function loadEmbeddings() {
  try {
    const raw = await fs.readFile(EMB_PATH, "utf-8");
    EMB = JSON.parse(raw);

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

// 향상된 텍스트 스코어링
function enhancedTextScore(question, item) {
  const qProcessed = preprocessText(question);
  const questionProcessed = preprocessText(item.question);
  const answerProcessed = preprocessText(item.answer);
  
  let score = 0;
  
  // 1. 질문 의도 분류 및 매칭
  const getQuestionIntent = (keywords) => {
    if (keywords.some(k => /이용|사용|신청|시작|운영|개통/.test(k))) return 'usage';
    if (keywords.some(k => /자격증|서약서|필요|조건|요건/.test(k))) return 'requirement';
    if (keywords.some(k => /비용|요금|수수료|가격|얼마/.test(k))) return 'cost';
    if (keywords.some(k => /절차|방법|어떻게|순서/.test(k))) return 'process';
    return 'general';
  };
  
  const qIntent = getQuestionIntent(qProcessed.keywords);
  const targetIntent = getQuestionIntent(questionProcessed.keywords);
  
  if (qIntent === targetIntent && qIntent !== 'general') {
    score += 0.3; // 의도 일치 보너스
  } else if (qIntent !== 'general' && targetIntent !== 'general' && qIntent !== targetIntent) {
    score -= 0.4; // 의도 불일치 페널티
  }
  
  // 2. 키워드 매칭 (가중치 적용)
  qProcessed.keywords.forEach(keyword => {
    const weight = KEYWORD_WEIGHTS[keyword] || 1.0;
    
    if (questionProcessed.processed.includes(keyword)) {
      score += 0.25 * weight;
    }
    if (answerProcessed.processed.includes(keyword)) {
      score += 0.15 * weight;
    }
  });
  
  // 3. 완전 문구 일치 보너스
  if (questionProcessed.processed.includes(qProcessed.processed) || 
      qProcessed.processed.includes(questionProcessed.processed)) {
    score += 0.4;
  }
  
  // 4. 키워드 밀도 점수
  const matchedKeywords = qProcessed.keywords.filter(k => 
    questionProcessed.processed.includes(k) || answerProcessed.processed.includes(k)
  );
  const keywordDensity = matchedKeywords.length / Math.max(qProcessed.keywords.length, 1);
  score += keywordDensity * 0.2;
  
  // 5. 답변 품질 점수
  if (item.answer && item.answer.length > 20) score += 0.1; // 충분한 답변 길이
  if (/\d+/.test(item.answer)) score += 0.1; // 구체적 수치 포함
  
  return Math.max(0, Math.min(score, 1.0));
}

// ---- 검색 함수 ----
function search({ question, qvec }) {
  // 키워드 가이드 체크
  const keywordGuide = checkKeywordGuide(question);
  if (keywordGuide.isKeywordGuide) {
    return {
      answer: keywordGuide.response,
      hits: [],
      bestScore: 1.0,
      found: true,
      isKeywordGuide: true,
      keyword: keywordGuide.keyword
    };
  }

  // 검색 실행
  const scored = EMB.map(it => {
    const sim = (Array.isArray(qvec) && Array.isArray(it.vector) && it.vector.length === EMB_DIM)
      ? cosineSimilarity(qvec, it.vector)
      : enhancedTextScore(question, it);
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

app.post("/ask", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector;
    if (!question) return res.status(400).json({ error: "question required" });

    const result = search({ question, qvec });
    return res.json(result);
  } catch (e) {
    console.error("[/ask] error:", e.message || e);
    return res.status(500).json({ error: "RAG error" });
  }
});

app.post("/ask/debug", async (req, res) => {
  try {
    const question = (req.body?.question || "").trim();
    const qvec = req.body?.vector;
    const N = Math.max(1, Math.min(20, Number(req.body?.top || 5)));
    if (!question) return res.status(400).json({ error: "question required" });

    const { hits, bestScore, found, isKeywordGuide, keyword } = search({ question, qvec });
    return res.json({ 
      top: hits.slice(0, N), 
      bestScore, 
      found, 
      count: EMB.length, 
      dim: EMB_DIM,
      isKeywordGuide,
      keyword,
      preprocessed: preprocessText(question)
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
