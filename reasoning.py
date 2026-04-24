import os
import json
import re
import instructor
from groq import Groq
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def disable_broken_local_proxy():
    proxy_keys = [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]
    for key in proxy_keys:
        value = os.environ.get(key)
        if value and ("127.0.0.1:9" in value or "localhost:9" in value):
            os.environ.pop(key, None)


def load_local_env():
    priority_keys = {"GROQ_API_KEY", "GROQ_KEY"}
    candidates = [
        os.path.join(BASE_DIR, ".env"),
        os.path.join(BASE_DIR, "env"),
        os.path.join(PROJECT_ROOT, ".env"),
        os.path.join(PROJECT_ROOT, "env"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if key in priority_keys:
                    # Prefer project-local key over stale shell-level values.
                    os.environ[key] = value
                elif key not in os.environ:
                    os.environ[key] = value


def get_groq_clients():
    disable_broken_local_proxy()
    load_local_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Groq API key. Add GROQ_API_KEY to scraper/env or set it in your environment."
        )

    raw = Groq(api_key=api_key)
    wrapped = instructor.from_groq(raw)
    return raw, wrapped


raw_client, client = get_groq_clients()
MODEL_CANDIDATES = [
    model for model in [
        os.getenv("GROQ_MODEL"),
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ] if model
]


class RegulatoryConstraint(BaseModel):
    subject_entity: str
    action: str
    constraint_value: Optional[str] = None
    condition: Optional[str] = None
    rule_type: str = Field(description="One of: obligation, prohibition, limit, permission")
    is_mandatory: bool
    source_clause: str
    confidence: float


class CompliancePolicy(BaseModel):
    circular_topic: str
    effective_date: str
    constraints: List[RegulatoryConstraint]


def parse_json_payload(raw_text: str):
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    # Remove markdown code fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()

    # If model adds prose, keep only the first JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model response")

    candidate = text[start:end + 1]
    return json.loads(candidate)


def normalize_optional_text(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null", "n/a", ""}:
        return None
    return value


def normalize_policy(policy: CompliancePolicy):
    for c in policy.constraints:
        c.constraint_value = normalize_optional_text(c.constraint_value)
        c.condition = normalize_optional_text(c.condition)
    return policy


def retrieve_context(query: str, k: int = 8):
    print(f"\n[RAG] Query: {query}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    results = db.similarity_search_with_score(query, k=k)

    context_blocks = []
    for i, (doc, score) in enumerate(results):
        context_blocks.append({
            "id": i,
            "source": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
            "score": float(score),
        })

    return context_blocks


def build_prompt(query, context_blocks):
    context_text = ""

    for block in context_blocks:
        context_text += f"""
[BLOCK ID: {block['id']} | SOURCE: {block['source']}]
{block['content']}
"""

    return f"""
You are a STRICT legal rule extraction engine.

CRITICAL RULES:
- DO NOT hallucinate
- DO NOT infer
- ONLY extract explicitly stated rules
- If unsure → SKIP

MANDATORY REQUIREMENTS:
1. source_clause MUST be the EXACT sentence copied from context
   - DO NOT write "BLOCK ID"
   - DO NOT summarize
   - MUST be verbatim text

2. Extract ALL numeric limits (₹, %, hours, etc.)
   - Missing a numeric rule is considered FAILURE

3. Confidence scoring:
   - 1.0 → exact explicit sentence
   - 0.8–0.9 → slight paraphrase
   - <0.7 → avoid extraction

4. Rule classification:
   - obligation → must / shall
   - prohibition → must not / shall not
   - limit → numeric thresholds
   - permission → may / allowed

   ⚠️ IMPORTANT:
   If rule describes system behavior (not restriction), DO NOT mark as prohibition.

OUTPUT FORMAT:
Return ONLY valid JSON.

{{
  "circular_topic": "...",
  "effective_date": "...",
  "constraints": [
    {{
      "subject_entity": "...",
      "action": "...",
      "constraint_value": "...",
      "condition": "...",
      "rule_type": "...",
      "is_mandatory": true/false,
      "source_clause": "...",
      "confidence": 0.0-1.0
    }}
  ]
}}

User Query:
{query}

Legal Context:
{context_text}
"""

def validate_constraints(policy: CompliancePolicy):
    valid_constraints = []

    for c in policy.constraints:

        # ❌ Reject bad source clause
        if "BLOCK ID" in c.source_clause:
            continue

        # ❌ Reject empty or vague clause
        if len(c.source_clause.strip()) < 20:
            continue

        # ✅ Fix confidence if model cheats
        if c.confidence == 1.0 and c.source_clause:
            c.confidence = 0.95

        # ✅ Clamp confidence
        c.confidence = max(0.0, min(1.0, c.confidence))

        valid_constraints.append(c)

    policy.constraints = valid_constraints
    return policy


def extract_rules(query: str, context_blocks):
    print("[LLM] Extracting structured policy...")

    prompt = build_prompt(query, context_blocks)
    last_error = None

    for model_name in MODEL_CANDIDATES:
        try:
            policy = client.chat.completions.create(
                model=model_name,
                response_model=CompliancePolicy,
                messages=[
                    {"role": "system", "content": "You extract legal rules precisely."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return policy
        except Exception as structured_error:
            last_error = structured_error
            print(f"[WARN] Structured parse failed on {model_name}. Trying JSON fallback...")

            try:
                response = raw_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )

                raw_text = response.choices[0].message.content
                data = parse_json_payload(raw_text)
                return normalize_policy(CompliancePolicy(**data))
            except Exception as fallback_error:
                last_error = fallback_error
                print(f"[WARN] Fallback failed on {model_name}.")

    err_text = str(last_error)
    if "invalid_api_key" in err_text.lower() or "invalid api key" in err_text.lower():
        print("[ERROR] All model attempts failed: Invalid Groq API key.")
        print("[ERROR] Update GROQ_API_KEY in scraper/.env (or set env var) with a valid active key.")
    else:
        print("[ERROR] All model attempts failed:", last_error)
    return None


def post_process(policy: CompliancePolicy):
    policy = normalize_policy(policy)
    policy = validate_constraints(policy)  # 🔥 new step

    seen = set()
    cleaned = []

    for c in policy.constraints:
        key = (c.subject_entity, c.action, c.constraint_value, c.condition)

        if key not in seen:
            seen.add(key)
            cleaned.append(c)

    policy.constraints = cleaned
    return policy


def run_reasoning_pipeline(
    query: str = "e-mandate rules AFA limits thresholds transaction limits recurring payments",
    retrieval_queries: Optional[List[str]] = None,
) -> CompliancePolicy:
    disable_broken_local_proxy()

    if retrieval_queries is None:
        retrieval_queries = [
            "e-mandate AFA rules",
            "transaction limits e-mandate",
            "recurring payments AFA requirements RBI",
            "pre transaction notification rules RBI",
        ]

    context_blocks = []
    for q in retrieval_queries:
        context_blocks.extend(retrieve_context(q, k=3))

    if not context_blocks:
        raise RuntimeError("No context retrieved")

    policy = extract_rules(query, context_blocks)
    if not policy:
        raise RuntimeError("Extraction failed")

    return post_process(policy)


if __name__ == "__main__":
    try:
        policy = run_reasoning_pipeline()
        print("\n===================================")
        print("[OK] FINAL STRUCTURED POLICY")
        print("===================================")
        print(policy.model_dump_json(indent=2))
    except Exception:
        print("[ERROR] Extraction failed")
        raise SystemExit(1)
