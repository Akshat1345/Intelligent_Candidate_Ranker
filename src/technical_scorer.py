"""
technical_scorer.py
-------------------
Computes the technical relevance score for a candidate against the JD.
No embeddings, no LLM calls — deterministic, CPU-only, runs ~15s for 100K.

JD (Senior AI Engineer, Redrob AI) — key signals extracted:
  MUST HAVE:
    - Production embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI)
    - Production vector DB / hybrid search (Pinecone, Qdrant, Milvus, FAISS, OpenSearch, ES)
    - Strong Python + code quality
    - Evaluation frameworks for ranking (NDCG, MRR, MAP, A/B testing)

  NICE TO HAVE:
    - LLM fine-tuning (LoRA, QLoRA, PEFT)
    - Learning-to-rank (XGBoost, neural LTR)
    - HR-tech / marketplace / recruiting experience
    - Distributed systems / large-scale inference

  EXPLICIT DISQUALIFIERS (from JD):
    - Pure consulting career (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini etc)
    - CV/speech/robotics specialist with NO NLP/IR exposure
    - Pure research (no production deployment)
    - "AI experience" = only LangChain tutorials / wrappers < 12 months
    - Non-tech career (Marketing, Accounting, Civil/Mech Eng, HR, Ops, Sales)
    - Title-chasers (job-hopping every 1.5 years for title bumps)

Score = weighted sum of 5 components, then multiplied by behavioral multiplier.
"""

import re
from typing import Any, Dict
from datetime import date, datetime

REFERENCE_DATE = date(2026, 6, 8)

# ---------------------------------------------------------------
# JD skill groups — extracted from actual job_description.md
# Each group = one required capability. Match = 1.0 for that group.
# ---------------------------------------------------------------
REQUIRED_SKILLS: list[tuple[str, list[str]]] = [
    ("embeddings_retrieval", [
        "embedding", "embeddings", "sentence-transformers", "sentence transformers",
        "openai embeddings", "bge", "e5", "text embeddings", "dense retrieval",
        "semantic search", "vector embeddings", "text2vec", "ada embeddings",
        "dense vector", "bi-encoder", "retrieval", "semantic retrieval",
        "neural retrieval", "dense passage retrieval", "dpr",
    ]),
    ("vector_db_search", [
        "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
        "elasticsearch", "chroma", "pgvector", "annoy", "hnsw",
        "vector database", "vector db", "vector search", "hybrid search",
        "approximate nearest neighbor", "ann search", "vector index",
        "vector store", "knn search",
    ]),
    ("ranking_ir", [
        "ranking", "information retrieval", "reranking", "re-ranking",
        "bm25", "hybrid retrieval", "rag", "retrieval augmented",
        "recommendation system", "recommendation systems", "recommender",
        "learning to rank", "ltr", "cross-encoder", "two-stage retrieval",
        "candidate retrieval", "search ranking", "relevance ranking",
        "xgboost ranking", "lightgbm ranking", "pairwise ranking",
        "listwise", "pointwise", "lambdamart",
    ]),
    ("python_engineering", [
        "python", "pytorch", "tensorflow", "sklearn", "scikit-learn",
        "numpy", "pandas", "fastapi", "flask", "pydantic",
        "asyncio", "celery", "ray", "spark", "pyspark",
    ]),
    ("eval_frameworks", [
        "ndcg", "mrr", "map@", "precision@", "recall@",
        "a/b test", "a/b testing", "ab test", "offline evaluation",
        "online evaluation", "offline eval", "ranking metrics",
        "evaluation framework", "experimentation", "mlflow",
        "weights & biases", "wandb", "experiment tracking",
        "offline-to-online", "recall@k", "precision@k",
    ]),
]

PREFERRED_SKILLS: list[tuple[str, list[str]]] = [
    ("llm_finetuning", [
        "fine-tuning", "finetuning", "lora", "qlora", "peft",
        "instruction tuning", "rlhf", "dpo", "sft",
    ]),
    ("ltr_models", [
        "xgboost", "lightgbm", "catboost", "learning to rank",
        "gradient boosting", "gbdt",
    ]),
    ("nlp_depth", [
        "nlp", "natural language processing", "transformers", "hugging face",
        "huggingface", "bert", "gpt", "llm", "language model",
        "text classification", "ner", "named entity", "question answering",
        "summarization", "tokenization",
    ]),
    ("infra_scale", [
        "kafka", "airflow", "kubernetes", "docker", "distributed",
        "microservices", "triton", "model serving", "inference optimization",
        "latency", "throughput", "sharding",
    ]),
]

# JD explicitly names these firms as disqualifiers for pure-career cases
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "ltimindtree", "mindtree", "l&t infotech",
    "kpit", "persistent systems", "mastech",
}

# Titles that are pure non-tech (when entire career = these)
NON_TECH_TITLE_WORDS = {
    "marketing manager", "accountant", "civil engineer", "mechanical engineer",
    "hr manager", "human resources", "operations manager", "customer support",
    "content writer", "sales executive", "sales manager", "graphic designer",
    "project manager", "financial analyst", "supply chain", "procurement",
    "business development", "brand manager", "product marketing",
}

# Tech role words — if ANY role in career has these, not a pure non-tech career
TECH_ROLE_WORDS = [
    "engineer", "developer", "scientist", "ml", "ai", "data", "software",
    "backend", "frontend", "full-stack", "fullstack", "machine learning",
    "nlp", "research engineer", "platform", "devops", "cloud", "architect",
    "applied", "research", "analytics engineer", "data engineer",
]

# Signals that a role description shows PRODUCTION AI/ML work
PRODUCTION_AI_SIGNALS = [
    "shipped", "deployed", "production", "launched", "served",
    "recommendation", "search", "ranking", "retrieval", "embedding",
    "rag", "fine-tun", "vector", "semantic", "nlp", "ml model",
    "machine learning", "a/b test", "feature engineering", "model",
    "training", "inference", "pipeline", "experiment",
]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 /&@]", " ", text.lower())


def _build_pool(candidate: Dict[str, Any]) -> str:
    """Full text bag from all profile sources."""
    parts = []
    for skill in candidate.get("skills", []):
        parts.append(_normalize(skill.get("name", "")))
    for role in candidate.get("career_history", []):
        parts.append(_normalize(role.get("description", "")))
        parts.append(_normalize(role.get("title", "")))
    p = candidate.get("profile", {})
    parts.append(_normalize(p.get("headline", "")))
    parts.append(_normalize(p.get("summary", "")))
    return " ".join(parts)


def _title_pool(candidate: Dict[str, Any]) -> str:
    """Only role titles — more trustworthy than descriptions (which are shuffled)."""
    titles = [_normalize(r.get("title", "")) for r in candidate.get("career_history", [])]
    titles.append(_normalize(candidate.get("profile", {}).get("current_title", "")))
    return " ".join(titles)


def _score_core_skills(candidate: Dict[str, Any]) -> float:
    """
    0.0–1.0+. Each required group = 1/5. Preferred groups give up to +0.20 bonus.
    Uses full text pool (skills + descriptions + titles + summary).
    """
    pool = _build_pool(candidate)

    req_hits = sum(
        1 for _name, syns in REQUIRED_SKILLS
        if any(syn in pool for syn in syns)
    )
    req_score = req_hits / len(REQUIRED_SKILLS)

    pref_hits = sum(
        1 for _name, syns in PREFERRED_SKILLS
        if any(syn in pool for syn in syns)
    )
    pref_bonus = (pref_hits / len(PREFERRED_SKILLS)) * 0.20

    return req_score + pref_bonus


def _score_career_substance(candidate: Dict[str, Any]) -> float:
    """
    0.0–1.0. Rewards product company experience with production AI/ML work.
    Penalizes pure consulting firms. Uses TITLE heavily (descriptions are shuffled).

    Key insight from data: descriptions may be mismatched to roles (dataset trap).
    Trust role TITLES more than descriptions.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    total_months = sum(max(0, r.get("duration_months", 0)) for r in career)
    if total_months == 0:
        return 0.0

    # Check if entire career is at consulting firms
    all_consulting = all(
        any(f in r.get("company", "").lower() for f in CONSULTING_FIRMS)
        for r in career
    )

    # Score each role
    weighted_score = 0.0
    for role in career:
        months = max(0, role.get("duration_months", 0))
        company_lower = role.get("company", "").lower()
        title_lower = role.get("title", "").lower()
        desc_lower = _normalize(role.get("description", ""))
        company_size = role.get("company_size", "")

        # Is this a tech/AI role by TITLE?
        is_tech_title = any(word in title_lower for word in TECH_ROLE_WORDS)
        is_ai_title = any(word in title_lower for word in [
            "ml", "ai", "machine learning", "nlp", "data scientist",
            "research engineer", "applied", "recommendation", "search",
            "ranking", "retrieval", "embedding",
        ])

        # Is this a consulting firm?
        is_consulting = any(f in company_lower for f in CONSULTING_FIRMS)

        # Production AI signals in description
        prod_signal_count = sum(1 for s in PRODUCTION_AI_SIGNALS if s in desc_lower)

        # Company size: smaller = more likely product company
        size_bonus = {
            "1-10": 0.15, "11-50": 0.10, "51-200": 0.05,
            "201-500": 0.0, "501-1000": -0.05, "1001-5000": -0.05,
            "5001-10000": -0.08, "10001+": -0.10,
        }.get(company_size, 0.0)

        # Build role score
        role_score = 0.0
        if is_ai_title:
            role_score += 0.70
        elif is_tech_title:
            role_score += 0.35

        # Production AI signals in description (partial credit since descriptions may be shuffled)
        role_score += min(0.30, prod_signal_count * 0.06)

        # Size adjustment
        role_score = max(0.0, role_score + size_bonus)

        # Consulting penalty
        if is_consulting:
            role_score *= 0.50 if not is_ai_title else 0.70

        weighted_score += role_score * months

    base = weighted_score / total_months

    # Hard penalty: pure consulting career (JD explicit disqualifier)
    if all_consulting:
        base *= 0.40

    return min(1.0, base)


def _score_experience_fit(candidate: Dict[str, Any]) -> float:
    """
    JD says 5-9 years ideal, but "will consider outside the band if other signals strong."
    Sweet spot: 6-8 years (JD says "roughly 6-8 years is the ideal we're imagining").
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if yoe < 2:
        return 0.10
    elif yoe < 3:
        return 0.30
    elif yoe < 4:
        return 0.55
    elif yoe < 5:
        return 0.75
    elif yoe <= 9:
        return 1.00
    elif yoe <= 11:
        return 0.85
    elif yoe <= 14:
        return 0.70
    else:
        return 0.55


def _score_skill_trust(candidate: Dict[str, Any]) -> float:
    """
    Anti-keyword-stuffing: trust skills that have duration + endorsements.
    Salary min>max is data noise, but endorsements and assessment scores are real.
    """
    skills = candidate.get("skills", [])
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    if not skills:
        return 0.0

    proficiency_values = {"beginner": 0.25, "intermediate": 0.50, "advanced": 0.75, "expert": 1.0}

    # Build pool of JD-relevant skill names
    all_jd_synonyms = set()
    for _, syns in REQUIRED_SKILLS + PREFERRED_SKILLS:
        all_jd_synonyms.update(syns)

    trust_scores = []
    for skill in skills:
        sname = skill.get("name", "")
        sname_norm = _normalize(sname)
        is_relevant = any(syn in sname_norm for syn in all_jd_synonyms)
        if not is_relevant:
            continue  # only trust JD-relevant skills

        prof = proficiency_values.get(skill.get("proficiency", "beginner"), 0.25)
        duration = min(72, skill.get("duration_months", 0))
        endorsements = min(60, skill.get("endorsements", 0))

        trust = prof * 0.35 + (duration / 72.0) * 0.40 + (endorsements / 60.0) * 0.25

        # Assessment corroboration with stricter handling for suspicious skills
        if sname in assessments:
            assessed = assessments[sname] / 100.0
            # Strict penalties for expert/advanced claims not backed by assessment
            if prof >= 1.0 and assessed * 100 < 30:
                trust = 0.20
            elif prof >= 0.75 and assessed * 100 < 20:
                trust = 0.25
            elif prof >= 0.75 and duration == 0 and endorsements == 0:
                trust = 0.10
            else:
                if abs(assessed - prof) < 0.30:
                    trust = min(1.0, trust * 1.15)
                elif assessed < prof - 0.35:
                    trust *= 0.75
        else:
            # No assessment: if advanced/expert but zero duration and zero endorsements, likely stuffing
            if prof >= 0.75 and duration == 0 and endorsements == 0:
                trust = 0.10

        trust_scores.append(trust)

    if not trust_scores:
        return 0.0

    trust_scores.sort(reverse=True)
    top_k = trust_scores[: min(8, len(trust_scores))]
    return sum(top_k) / len(top_k)


def _score_education(candidate: Dict[str, Any]) -> float:
    tier_values = {"tier_1": 1.0, "tier_2": 0.85, "tier_3": 0.70, "tier_4": 0.55, "unknown": 0.60}
    education = candidate.get("education", [])
    if not education:
        return 0.60
    best = max(tier_values.get(e.get("tier", "unknown"), 0.60) for e in education)
    return best


def _score_ai_career_depth(candidate: Dict[str, Any]) -> float:
    """
    Compute proportion of career spent in core AI/ML titles using TITLES only.
    Returns a score according to thresholds specified in the prompt.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    core_ai_keywords = [
        "ml engineer", "machine learning engineer", "ai engineer", "nlp engineer",
        "data scientist", "applied scientist", "research engineer", "search engineer",
        "recommendation", "recommendation systems", "senior ai", "lead ai",
        "staff ml",
    ]
    total_months = 0
    ai_months = 0
    for role in career:
        months = max(0, role.get("duration_months", 0))
        total_months += months
        title = role.get("title", "").lower()
        if any(k in title for k in core_ai_keywords):
            ai_months += months

    if total_months == 0:
        return 0.0

    if ai_months >= 48:
        return 1.0
    if ai_months >= 30:
        return 0.75
    if ai_months >= 18:
        return 0.50
    if ai_months >= 6:
        return 0.25
    return 0.0


def _is_langchain_wrapper(candidate: Dict[str, Any]) -> bool:
    """
    Detect thin 'LangChain wrapper' profiles per prompt heuristics.
    """
    pool = _build_pool(candidate)
    headline = _normalize(candidate.get("profile", {}).get("headline", ""))
    summary = _normalize(candidate.get("profile", {}).get("summary", ""))
    text = headline + " " + summary + " " + pool

    wrapper_terms = [
        "langchain", "llamaindex", "openai api", "chatgpt", "gpt-4",
        "prompt engineering", "no-code", "automation",
    ]
    hit_count = sum(1 for t in wrapper_terms if t in text)

    # Production signals in career descriptions
    career = candidate.get("career_history", [])
    prod_count = 0
    for role in career:
        desc = _normalize(role.get("description", ""))
        prod_count += sum(1 for s in PRODUCTION_AI_SIGNALS if s in desc)

    # Check for vector DB / embedding mentions
    vector_terms = set()
    for name, syns in REQUIRED_SKILLS:
        if name in ("embeddings_retrieval", "vector_db_search"):
            vector_terms.update(syns)
    has_vector = any(v in pool for v in vector_terms)

    return hit_count >= 3 and prod_count == 0 and not has_vector


def _score_trajectory(candidate: Dict[str, Any]) -> float:
    """
    Heuristic trajectory scoring: promotions within AI titles.
    """
    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    ai_keywords = ["ml", "ai", "nlp", "machine learning", "research", "data scientist", "recommendation", "search"]

    def seniority(title: str) -> int:
        t = title.lower()
        if any(x in t for x in ["staff", "principal", "lead", "head"]):
            return 4
        if any(x in t for x in ["senior", "sr", "seniior"]):
            return 3
        if any(x in t for x in ["mid", "associate"]):
            return 2
        if any(x in t for x in ["junior", "jr"]):
            return 1
        # default: if contains ai keyword and engineer/scientist, treat as mid
        if any(k in t for k in ai_keywords) and any(x in t for x in ["engineer", "scientist", "developer"]):
            return 2
        return 0

    ai_roles = [r for r in career if any(k in (r.get("title", "").lower()) for k in ai_keywords)]
    if not ai_roles:
        return 0.0

    # Compute seniority progression across AI roles in listed order
    prev = None
    promotions = 0
    for r in ai_roles:
        s = seniority(r.get("title", ""))
        if prev is not None and s > prev:
            promotions += 1
        prev = s

    # Check for full progression junior->senior->staff
    titles_seq = [r.get("title", "").lower() for r in ai_roles]
    has_full = any("junior" in t for t in titles_seq) and any("senior" in t for t in titles_seq) and any(any(x in t for x in ["staff", "lead", "principal"]) for t in titles_seq)

    if has_full:
        return 1.0
    if promotions >= 2:
        return 0.85
    if promotions >= 1:
        return 0.65

    # pivot: recent role AI but earlier roles non-tech
    first_ai_idx = None
    for i, r in enumerate(career):
        if any(k in r.get("title", "").lower() for k in ai_keywords):
            first_ai_idx = i
            break
    if first_ai_idx is not None and first_ai_idx > 0:
        # earlier roles present and not AI
        earlier = career[:first_ai_idx]
        nontech_earlier = sum(1 for r in earlier if not any(k in r.get("title", "").lower() for k in ai_keywords))
        if nontech_earlier >= max(1, len(earlier) // 2):
            return 0.40

    return 0.0


def _is_hard_disqualified(candidate: Dict[str, Any]) -> tuple[bool, str]:
    """
    JD-explicit hard disqualifiers that should score very low.
    Returns (disqualified, reason).
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    pool = _build_pool(candidate)
    title_pool_str = _title_pool(candidate)

    # 1. Entire career = non-tech roles only
    has_any_tech_role = any(
        any(word in (r.get("title", "") + r.get("description", "")).lower()
            for word in TECH_ROLE_WORDS)
        for r in career
    )
    current_title_lower = profile.get("current_title", "").lower()
    current_is_nonttech = any(t in current_title_lower for t in NON_TECH_TITLE_WORDS)

    if not has_any_tech_role and current_is_nonttech:
        return True, f"pure_non_tech_career: {profile.get('current_title')}"

    # 2. CV/speech/robotics specialist with NO NLP/IR signals
    cv_speech_signals = [
        "computer vision", "opencv", "yolo", "object detection",
        "speech recognition", "tts", "text to speech", "robotics", "ros",
        "image segmentation", "pose estimation",
    ]
    nlp_ir_signals = [
        "nlp", "retrieval", "search", "ranking", "embedding", "transformers",
        "language model", "recommendation", "information retrieval", "bm25",
    ]
    cv_count = sum(1 for s in cv_speech_signals if s in pool)
    ir_count = sum(1 for s in nlp_ir_signals if s in pool)

    if cv_count >= 3 and ir_count == 0:
        return True, "pure_cv_speech_no_nlp_ir"

    return False, ""


def _score_title_job_hop_penalty(candidate: Dict[str, Any]) -> float:
    """
    JD explicitly dislikes title-chasers who switch every 1.5 years for title bumps.
    Returns a multiplier 0.7-1.0.
    """
    career = candidate.get("career_history", [])
    if len(career) < 3:
        return 1.0

    # Count very short stints (< 18 months) at different companies
    short_hops = 0
    for role in career:
        months = role.get("duration_months", 36)
        if months < 15 and not role.get("is_current", False):
            short_hops += 1

    if short_hops >= 3:
        return 0.75
    elif short_hops >= 2:
        return 0.88
    return 1.0


def compute_technical_score(
    candidate: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    """
    Returns (composite_tech_score, breakdown_dict).
    Score is unbounded above 1.0 (preferred skills add bonus).
    Multiply by behavioral multiplier in rank.py for final score.
    """
    WEIGHTS = {
        "core_skills_match": 0.30,
        "ai_career_depth": 0.25,
        "career_substance": 0.15,
        "experience_fit": 0.15,
        "skill_trust": 0.10,
        "trajectory": 0.03,
        "education_tier": 0.02,
    }

    disq, reason = _is_hard_disqualified(candidate)
    if disq:
        return 0.05, {
            "core_skills_match": 0.0,
            "career_substance": 0.0,
            "experience_fit": 0.0,
            "skill_trust": 0.0,
            "education_tier": _score_education(candidate),
            "disqualified_reason": reason,
        }

    breakdown = {
        "core_skills_match": _score_core_skills(candidate),
        "ai_career_depth": _score_ai_career_depth(candidate),
        "career_substance": _score_career_substance(candidate),
        "experience_fit": _score_experience_fit(candidate),
        "skill_trust": _score_skill_trust(candidate),
        "trajectory": _score_trajectory(candidate),
        "education_tier": _score_education(candidate),
    }

    composite = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)

    # Apply job-hop penalty
    hop_penalty = _score_title_job_hop_penalty(candidate)
    composite *= hop_penalty
    breakdown["job_hop_penalty"] = hop_penalty

    # LangChain wrapper penalty
    if _is_langchain_wrapper(candidate):
        composite *= 0.60
        breakdown["langchain_wrapper_penalty"] = 0.60

    return composite, breakdown
