"""
behavioral_scorer.py
--------------------
Computes a behavioral multiplier [0.5, 1.5] from redrob_signals.
This is applied multiplicatively on top of the technical score.

The JD explicitly says: "a perfect-on-paper candidate who hasn't logged in
for 6 months and has a 5% recruiter response rate is, for hiring purposes,
not actually available. Down-weight them appropriately."

Three sub-components:
  availability  (0.40 weight) — can we actually hire them?
  engagement    (0.35 weight) — are they active and responsive?
  platform_trust (0.25 weight) — are their signals credible?
"""

from datetime import date, datetime
from typing import Any, Dict


REFERENCE_DATE = date(2026, 6, 8)
MIN_MULTIPLIER = 0.50
MAX_MULTIPLIER = 1.50


def _days_since(date_str: str | None) -> int:
    """Days between REFERENCE_DATE and the given date string. Returns 9999 if unparseable."""
    if not date_str:
        return 9999
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return max(0, (REFERENCE_DATE - d).days)
    except (ValueError, TypeError):
        return 9999


def _recency_score(days: int) -> float:
    """Decay function: recent activity = high score."""
    if days <= 30:
        return 1.0
    elif days <= 90:
        return 0.85
    elif days <= 180:
        return 0.65
    elif days <= 365:
        return 0.35
    else:
        return 0.10


def _notice_period_score(days: int) -> float:
    """JD says they want sub-30 day notice; can buy out 30 days."""
    if days <= 0:
        return 1.0
    elif days <= 30:
        return 1.0
    elif days <= 60:
        return 0.80
    elif days <= 90:
        return 0.60
    elif days <= 120:
        return 0.40
    else:
        return 0.20


def _score_availability(signals: Dict[str, Any]) -> float:
    """
    0.0–1.0. High if: open_to_work, recently active, short notice, willing to relocate.
    """
    score = 0.0

    # open_to_work is the single strongest availability signal
    if signals.get("open_to_work_flag", False):
        score += 0.40

    # Recency of last login
    days_since_active = _days_since(signals.get("last_active_date"))
    score += _recency_score(days_since_active) * 0.35

    # Notice period
    notice = signals.get("notice_period_days", 90)
    score += _notice_period_score(notice) * 0.15

    # Willing to relocate (JD says Pune/Noida preferred, open to others)
    if signals.get("willing_to_relocate", False):
        score += 0.10

    return min(1.0, score)


def _score_engagement(signals: Dict[str, Any]) -> float:
    """
    0.0–1.0. High if: responds to recruiters, completes interviews,
    applies actively, profile is complete.
    """
    score = 0.0

    # Recruiter response rate — direct measure of reachability
    response_rate = signals.get("recruiter_response_rate", 0.0)
    if isinstance(response_rate, (int, float)) and response_rate >= 0:
        score += response_rate * 0.35

    # Interview completion rate
    icr = signals.get("interview_completion_rate", 0.0)
    if isinstance(icr, (int, float)) and icr >= 0:
        score += icr * 0.25

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 0.0) / 100.0
    score += completeness * 0.20

    # Applications submitted recently (shows active job-seeking)
    apps = min(10, signals.get("applications_submitted_30d", 0))
    score += (apps / 10.0) * 0.10

    # Saved by recruiters recently (social proof)
    saved = min(10, signals.get("saved_by_recruiters_30d", 0))
    score += (saved / 10.0) * 0.10

    return min(1.0, score)


def _score_platform_trust(signals: Dict[str, Any], skills: list) -> float:
    """
    0.0–1.0. High if: verified contact info, consistent assessment scores,
    github activity (for technical role), linkedin connected.
    """
    score = 0.0

    # Verification signals
    if signals.get("verified_email", False):
        score += 0.20
    if signals.get("verified_phone", False):
        score += 0.15
    if signals.get("linkedin_connected", False):
        score += 0.10

    # GitHub activity (very relevant for Senior AI Engineer role)
    github = signals.get("github_activity_score", -1)
    if isinstance(github, (int, float)) and github >= 0:
        score += (github / 100.0) * 0.30
    else:
        # No GitHub: neutral (not a hard requirement, but preferred)
        score += 0.05

    # GitHub bonus for this JD as explicit differentiator
    # >=70 => +0.12, >=40 => +0.06
    if isinstance(github, (int, float)) and github >= 0:
        if github >= 70:
            score += 0.12
        elif github >= 40:
            score += 0.06

    # Assessment score quality — take the average of completed assessments
    assessments = signals.get("skill_assessment_scores", {})
    if assessments:
        avg_score = sum(assessments.values()) / len(assessments)
        score += (avg_score / 100.0) * 0.20
    else:
        # No assessments taken: small penalty (didn't engage with platform)
        score += 0.05

    # Response time — faster = better
    avg_response_hours = signals.get("avg_response_time_hours", 999)
    if isinstance(avg_response_hours, (int, float)) and avg_response_hours >= 0:
        if avg_response_hours <= 24:
            score += 0.05
        elif avg_response_hours <= 72:
            score += 0.03

    return min(1.0, score)


def compute_behavioral_multiplier(
    candidate: Dict[str, Any],
) -> tuple[float, Dict[str, float]]:
    """
    Returns (multiplier, breakdown_dict).
    multiplier is in [MIN_MULTIPLIER, MAX_MULTIPLIER].
    """
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])

    availability = _score_availability(signals)
    engagement = _score_engagement(signals)
    platform_trust = _score_platform_trust(signals, skills)
    # Location score to slightly bias India preferred cities
    def _score_location(signals: Dict[str, Any]) -> float:
        loc = signals.get("location", "") or ""
        # Use profile country and willing_to_relocate if available
        country = signals.get("country", "")
        # Best: India preferred cities
        profile_loc = loc.lower() if isinstance(loc, str) else ""
        if any(x in profile_loc for x in ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "bangalore", "bengaluru"]):
            return 0.05
        # India but other city -> neutral
        if country and country.lower() == "india":
            return 0.0
        # Outside India
        if signals.get("willing_to_relocate", False):
            return -0.03
        return -0.08

    location_score = _score_location({**signals, **{"location": candidate.get("profile", {}).get("location", ""), "country": candidate.get("profile", {}).get("country", "")}})

    # Weighted composite behavioral score [0, 1]
    behavioral_score = (
        availability * 0.37
        + engagement * 0.33
        + platform_trust * 0.25
        + location_score * 0.05
    )

    # Map [0, 1] → [MIN_MULTIPLIER, MAX_MULTIPLIER]
    multiplier = MIN_MULTIPLIER + behavioral_score * (MAX_MULTIPLIER - MIN_MULTIPLIER)

    breakdown = {
        "availability": round(availability, 4),
        "engagement": round(engagement, 4),
        "platform_trust": round(platform_trust, 4),
        "location_score": round(location_score, 4),
        "behavioral_composite": round(behavioral_score, 4),
        "multiplier": round(multiplier, 4),
    }
    return round(multiplier, 4), breakdown
