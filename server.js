import "dotenv/config";
import fs from "fs";
import express from "express";
import OpenAI from "openai";
import { cosineSimilarity } from "./utils/similarity.js";
import cors from "cors";

const app = express();

// ✅ CORS
app.use(cors({
  origin: [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost",
    "http://127.0.0.1",
    "https://www.060kc.com",
    "https://060kc.com",
    // 내부망 테스트가 필요하면 정규식 Origin을 추가하세요:
    // /^http:\/\/192\.168\.0\.\d+(:\d+)?$/
  ],
  methods: ["GET","POST","OPTIONS"],
  allowedHeaders: ["Content-Type","Authorization"],
  credentials: false
}));
app.options("*", cors());

app.use(express.json());
app.use((req,res,next)=>{ res.set("Cache-Control","no-store"); next(); });

if (!process.env.OPENAI_API_KEY) {
  console.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env를 확인하세요.");
  process.exit(1);
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const PORT = Number(process.env.PORT);
if (!PORT) {
  console.error("❌ PORT env missing. Render Web Service는 PORT로만 리슨해야 합니다.");
  process.exit(1);
}
const THRESHOLD = Number(process.env.THRESHOLD || 0.78);

// ✅ 헬스체크
app.get("/health", (req, res) => res.json({ ok: true, ts: Date.now() }));
app.get("/", (req, res) => res.send("OK"));

// ✅ 인덱스 로드
let index = [];
try {
  index = JSON.parse(fs.readFileSync("./data/embeddings.json", "utf8"));
  console.log("[EMB] loaded", { count: index.length, dim: index[0]?.e?.length, cwd: process.cwd() });
} catch (err) {
  console.error("❌ embeddings.json 로드 실패:", err.message);
  index = [];
}

// ★ (1) “질문/답변” 라벨 제거 (강화 버전)
function cleanAnswer(raw = "") {
  let t = (raw || "").trim();

  // 1) "질문: ... 답변:" 블록 통째 제거 (가장 강함)
  t = t.replace(/^\s*질문\s*[:：]\s*[\s\S]*?답변\s*[:：]\s*/i, "");

  // 2) 줄 시작 접두어 제거 (여러 줄 모두)
  t = t.replace(/^\s*(Q|질문)\s*[:：]\s*/gmi, "")
       .replace(/^\s*(A|답변|답)\s*[:：]\s*/gmi, "");

  // 3) 중간에 남은 라벨 토막 제거
  t = t.replace(/\b(질문|답변|답)\s*[:：]\s*/gi, "");

  // 4) 앞쪽 불필요한 마침표/공백 정리
  t = t.replace(/^[.\u3002]+/, "").trim();

  return t;
}

// ★ (2) 질의 정규화(동의어/불용어 처리)
async function normalizeQuery(q) {
  try {
    const chat = await client.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.0,
      messages: [
        {
          role: "system",
          content:
            "아래 한국어 질문을 검색에 적합한 한 줄 쿼리로 바꿔줘. 의미 보존, 동의어/약어 풀어쓰기, 불필요한 말 삭제. 출력은 한 줄 쿼리만."
        },
        { role: "user", content: q }
      ]
    });
    return chat.choices[0]?.message?.content?.trim() || q;
  } catch {
    return q;
  }
}

async function embed(text) {
  const res = await client.embeddings.create({ model: EMBED_MODEL, input: text });
  return res.data[0].embedding;
}

function topK(qv, k = 15) { // 후보 수 확장
  const scored = index.map(item => ({
    q: item.q,
    a: item.a,
    score: cosineSimilarity(qv, item.e)
  }));
  return scored.sort((x, y) => y.score - x.score).slice(0, k);
}

app.post("/ask", async (req, res) => {
  try {
    const { question, rewrite = false } = req.body || {};
    if (!question) return res.status(400).json({ error: "question required" });

    // 인덱스 비어있으면 즉시 폴백
    if (!Array.isArray(index) || index.length === 0) {
      return res.json({ answer: "자료에 없음", hits: [], bestScore: 0, found: false });
    }

    // ★ 질의 정규화 후 임베딩
    const normalized = await normalizeQuery(question);
    const qv = await embed(normalized);

    const hits = topK(qv, 15);
    const best = hits[0];
    const bestScore = best?.score ?? 0;

    // 기준 미달 → 폴백 유도
    if (!best || bestScore < THRESHOLD) {
      return res.json({ answer: "자료에 없음", hits, bestScore, found: false });
    }

    // 기준 통과 → 답변 사용 (포맷 정리)
    let answer = cleanAnswer(best.a);

    // 선택: 자연스러운 한국어로 재작성(“답만 출력” 강제)
    if (rewrite) {
      const chat = await client.chat.completions.create({
        model: "gpt-4o-mini",
        temperature: 0.1,
        messages: [
          {
            role: "system",
            content:
              "너는 060KC Q&A 전용 상담봇이다. 반드시 '답변 본문'만 출력해. '질문:' '답변:' 같은 접두어/설명, 인사말, 새로운 사실 모두 금지."
          },
          { role: "user", content: `다듬을 답변:\n${answer}\n\n출력은 답변 문장만.` }
        ]
      });
      answer = chat.choices[0]?.message?.content?.trim() || answer;
    }

    // ✅ 응답 직전에 다시 한 번 강제 클린
    answer = cleanAnswer(answer);

    res.json({ answer, hits, bestScore, found: true });
  } catch (e) {
    console.error("❌ /ask error:", e);
    res.status(500).json({ error: "server error" });
  }
});

// 기본은 로컬호스트에만 바인드
//app.listen(PORT, () => {
//  console.log(`🚀 060KC RAG API on http://localhost:${PORT} (THRESHOLD=${THRESHOLD})`);
//});

// 내부망에서 접근해야 하면 아래로 교체하세요.
 app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 RAG ONLINE on 0.0.0.0:${PORT} (TH=${THRESHOLD})`);
 });
