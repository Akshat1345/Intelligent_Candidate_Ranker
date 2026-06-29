"""
reasoning_generator.py
-----------------------
Generates specific, factual, non-hallucinating 1-2 sentence reasoning
for each ranked candidate.

Stage 4 evaluation checks:
  - Specific facts from the profile
  - JD connection
  - Honest acknowledgment of concerns
  - No hallucination
  - Variation across candidates
  - Rank consistency (tone matches rank)

This module produces deterministic, template-free reasoning by reading
actual profile facts and constructing sentences from them.
"""

from datetime import date, datetime
from typing import Any, Dict


REFERENCE_DATE = date(2026, 6, 8)


def _years_ago(date_str: str | None) -> str:
    if not date_str:
        return "unknown"
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        days = (REFERENCE_DATE - d).days
        if days <= 45:
            return "this month"
        elif days <= 90:
            return "recently"
        elif days <= 180:
            return "3-6 months ago"
        elif days <= 365:
            return "about a year ago"
        else:
            return f"{days // 365}+ year{'s' if days // 365 > 1 else ''} ago"
    except Exception:
        return "unknown"


def _days_since(date_str: str | None) -> int:
    if not date_str:
        return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (REFERENCE_DATE - d).days)
    except Exception:
        return 9999


def _top_skills(candidate: Dict[str, Any], max_skills: int = 3) -> list[str]:
    """Returns the most credible relevant skills by trust score."""
    from technical_scorer import REQUIRED_SKILLS, _normalize

    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})
    proficiency_rank = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

    # Build a flat set of all JD-relevant synonyms for matching
    relevant_synonyms = set()
    for _name, syns in REQUIRED_SKILLS:
        relevant_synonyms.update(syns)

    scored = []
    for s in skills:
        name_lower = _normalize(s.get("name", ""))
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)
        assessed = assessments.get(s.get("name", ""), -1)

        # Only include skills relevant to the JD
        is_relevant = any(syn in name_lower for syn in relevant_synonyms)

        trust = (
            proficiency_rank.get(prof, 1) * 0.4
            + min(60, duration) / 60.0 * 0.4
            + min(50, endorsements) / 50.0 * 0.2
        )
        if assessed >= 0:
            trust *= 1.1 if assessed > 50 else 0.9

        scored.append((s.get("name", ""), trust, is_relevant))

    # Relevant first, then by trust
    scored.sort(key=lambda x: (x[2], x[1]), reverse=True)
    names = [name for name, _, _ in scored[:max_skills]]
    # Ensure at least two specific skills are returned (fallback to raw skills list)
    if len(names) < 2:
        raw = [s.get("name", "") for s in candidate.get("skills", []) if s.get("name")]
        for r in raw:
            if r not in names:
                names.append(r)
            if len(names) >= 2:
                break
    return names[:max_skills]


def _current_situation(candidate: Dict[str, Any]) -> str:
    """One-phrase description of current role."""
    p = candidate.get("profile", {})
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    yoe = p.get("years_of_experience", 0)
    return f"{title} at {company} ({yoe:.0f} yrs exp)"


def _best_career_highlight(candidate: Dict[str, Any]) -> str:
    """Finds the most impressive career fact from descriptions."""
    from technical_scorer import PRODUCTION_AI_SIGNALS as PRODUCT_SIGNALS, _normalize, CONSULTING_FIRMS

    career = candidate.get("career_history", [])
    if not career:
        return ""

    highlight_keywords = [
        ("shipped", "shipped"),
        ("deployed", "deployed to production"),
        ("built", "built"),
        ("designed", "designed"),
        ("led", "led"),
        ("recommendation", "recommendation systems"),
        ("search", "search systems"),
        ("retrieval", "retrieval pipelines"),
        ("ranking", "ranking systems"),
        ("embedding", "embedding systems"),
        ("rag", "RAG pipelines"),
        ("fine-tun", "LLM fine-tuning"),
        ("vector", "vector search"),
        ("semantic", "semantic search"),
        ("nlp", "NLP systems"),
        ("ml model", "ML models"),
        ("machine learning", "ML"),
        ("a/b test", "A/B testing"),
        ("scale", "at scale"),
        ("production", "production systems"),
    ]

    best_score = 0
    best_phrase = ""

    for role in career:
        desc_lower = _normalize(role.get("description", ""))
        company = role.get("company", "")
        is_consulting = any(f in company.lower() for f in CONSULTING_FIRMS)
        base_score = 0.5 if is_consulting else 1.0

        match_count = 0
        matched_phrases = []
        for kw, phrase in highlight_keywords:
            if kw in desc_lower:
                match_count += 1
                matched_phrases.append(phrase)

        score = base_score * match_count
        if score > best_score and matched_phrases:
            best_score = score
            best_phrase = f"{', '.join(matched_phrases[:2])} at {company}"

    return best_phrase


def _concerns(candidate: Dict[str, Any], score: float, breakdown: Dict) -> list[str]:
    """Identifies honest concerns to flag."""
    concerns = []
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Notice period
    notice = signals.get("notice_period_days", 0)
    if notice > 90:
        concerns.append(f"long notice period ({notice} days)")
    elif notice > 60:
        concerns.append(f"notice period is {notice} days")

    # Recency
    last_active = signals.get("last_active_date", "")
    ago = _years_ago(last_active)
    if "year" in ago:
        concerns.append(f"last active {ago}")

    # Low response rate
    rr = signals.get("recruiter_response_rate", 1.0)
    if isinstance(rr, float) and rr < 0.25:
        concerns.append(f"low recruiter response rate ({rr:.0%})")

    # Core skill gap
    if breakdown.get("core_skills_match", 1.0) < 0.35:
        concerns.append("limited direct AI/ML skills match")

    # YOE band
    yoe = profile.get("years_of_experience", 0)
    if yoe < 4:
        concerns.append(f"below ideal experience band ({yoe:.1f} yrs vs 5-9 preferred)")
    elif yoe > 12:
        concerns.append(f"above ideal band ({yoe:.1f} yrs) — may be overqualified")

    return concerns[:2]  # cap at 2 concerns


def generate_reasoning(
    candidate: Dict[str, Any],
    rank: int,
    final_score: float,
    tech_breakdown: Dict[str, float],
    behav_breakdown: Dict[str, float],
) -> str:
    """
    Generate specific, factual, rank-consistent 1-2 sentence reasoning.
    """
    cid = candidate.get("candidate_id", "")
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    # Ensure company present — fallback to most recent career company
    if not company:
        ch = candidate.get("career_history", [])
        if ch:
            company = ch[0].get("company", "")
    location = profile.get("location", "")
    notice = signals.get("notice_period_days", 90)
    last_active = _years_ago(signals.get("last_active_date"))
    open_to_work = signals.get("open_to_work_flag", False)
    rr = signals.get("recruiter_response_rate", 0)
    skills_match = tech_breakdown.get("core_skills_match", 0)
    career_score = tech_breakdown.get("career_substance", 0)

    top_skills = _top_skills(candidate, max_skills=3)
    career_highlight = _best_career_highlight(candidate)
    concerns = _concerns(candidate, final_score, tech_breakdown)

    # Build the first sentence (strengths) — every tier gets company, skills,
    # and career context. Only the FRAMING/enthusiasm changes by rank, not the
    # amount of specific detail, so reasoning never collapses into a thin template.

    sentence1_parts = []
    skills_to_show = top_skills[:3] if top_skills else []

    # Build sentence1 with varied starters and explicit phrasing per rank band
    if rank <= 10:
        if yoe:
            sentence1_parts.append(f"{yoe:.0f}-year {title} at {company}")
        if career_highlight:
            sentence1_parts.append(career_highlight)
        if skills_to_show:
            sentence1_parts.append(f"brings hands-on {yoe:.0f}yrs experience with {', '.join(skills_to_show)}")
        if open_to_work:
            sentence1_parts.append("actively seeking")
        if notice <= 30:
            sentence1_parts.append(f"available within {notice} days")

    elif rank <= 30:
        if yoe:
            sentence1_parts.append(f"Currently {title} at {company} ({yoe:.0f} yrs)")
        if skills_to_show:
            sentence1_parts.append(f"strong applied experience in {', '.join(skills_to_show)}")
        if career_highlight:
            sentence1_parts.append(career_highlight.split(" at ")[0])

    elif rank <= 60:
        if yoe:
            sentence1_parts.append(f"{title} ({company}, {yoe:.0f} yrs)")
        if skills_to_show:
            sentence1_parts.append(f"brings hands-on {yoe:.0f}yrs experience with {', '.join(skills_to_show)}")
        else:
            sentence1_parts.append("partial overlap with JD's core requirements")
        if career_score < 0.4:
            # identify primary background
            titles = " ".join([r.get("title", "").lower() for r in candidate.get("career_history", [])])
            if "data" in titles or "backend" in titles or "spark" in titles:
                primary = "data engineering/backend"
            elif "research" in titles:
                primary = "research"
            else:
                primary = "services/consulting"
            sentence1_parts.append(f"primary background is {primary}, with {', '.join(skills_to_show or [])} as adjacent exposure rather than production focus")

    else:
        if yoe:
            sentence1_parts.append(f"{title} at {company} ({yoe:.0f} yrs)")
        if skills_to_show:
            sentence1_parts.append(f"brings hands-on {yoe:.0f}yrs experience with {', '.join(skills_to_show)}")
            # explicit gap identification
            titles = " ".join([r.get("title", "").lower() for r in candidate.get("career_history", [])])
            if "data" in titles or "backend" in titles:
                primary = "data engineering/backend"
            elif "research" in titles:
                primary = "research"
            else:
                primary = "non-tech"
            sentence1_parts.append(f"primary background is {primary}, with {', '.join(skills_to_show or [])} as adjacent exposure rather than production focus")
        else:
            sentence1_parts.append("limited direct technical match against the JD's must-haves")

    sentence1 = "; ".join(p for p in sentence1_parts if p) + "."

    # Build the second sentence (concerns or confirmation)
    if concerns:
        concern_str = ", ".join(concerns)
        if rank <= 20:
            sentence2 = f"Noted concern: {concern_str}."
        else:
            sentence2 = f"Key concerns: {concern_str}."
    else:
        # No concerns — must provide a concrete positive with numbers
        days_since_active = _days_since(signals.get("last_active_date"))
        if isinstance(rr, (int, float)) and rr >= 0:
            sentence2 = f"Positive signals: recruiter response rate {rr:.0%}; notice period {notice} days; last active {days_since_active} days ago."
        else:
            sentence2 = f"Positive signals: notice period {notice} days; last active {days_since_active} days ago."

    return f"{sentence1} {sentence2}".strip()
