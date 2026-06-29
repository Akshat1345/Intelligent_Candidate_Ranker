"""
honeypot_detector.py
--------------------
Detects the ~80 true honeypot profiles in the 100K dataset.

STATISTICAL REALITY CHECK:
80/100,000 = 0.08% base rate. In our 50-candidate sample we should expect to
see ZERO true honeypots (expected count = 0.04). This means we cannot eyeball
calibrate against the sample file — any single-field "weirdness" check fires on
1-25% of candidates because the dataset has substantial per-field noise
injected everywhere (salary min>max in 26% of profiles, active-before-signup
in ~4%, YOE-vs-graduation mismatches in ~4%, etc). These are NOT honeypots —
they're realistic messy data.

KEY INSIGHT: a true honeypot is a profile that is impossible across MULTIPLE
independent dimensions simultaneously. Noise in any one field is common;
noise stacking on the same candidate across 2+ unrelated fields is rare by
construction (if each field is independently noisy at rate p, the chance of
2 independent fields both firing on the same candidate is roughly p^2).

We score each candidate on a 0-N "impossibility count" across many weak
signals, and only flag candidates whose count clears a threshold tuned so
that the total flagged count lands near the documented ~80 (0.08%) when run
against the full 100K pool. This is necessarily an estimate — recalibrate
the IMPOSSIBILITY_THRESHOLD constant after running on the full dataset and
checking the printed honeypot rate.
"""

from datetime import datetime, date
from typing import Any, Dict


REFERENCE_DATE = date(2026, 6, 8)

# Tune this after seeing the full-dataset honeypot count.
# Each weak signal below contributes a partial point; only profiles with
# point totals >= threshold are flagged.
#
# CALIBRATED against the full 100K dataset (see analyze_honeypots.py output):
# distribution showed a clean noise floor at 0.0/0.4/0.8/1.0/1.4 (99,933 candidates
# from single or coincidental double weak-signal hits), then an isolated, sharply
# distinct cluster of 62 candidates at EXACTLY 1.8 points -- every one stacking
# the same three flags (active_before_signup + salary_min_gt_max +
# yoe_exceeds_grad_window), plus 1 extreme outlier at 2.4. This 1.8 cluster is the
# honeypot signature: deliberately constructed, not coincidental noise. Total
# candidates >= 1.5 = 66, matching the documented "~80" honeypots closely enough
# to treat 1.5 as the real boundary between noise and constructed impossibility.
IMPOSSIBILITY_THRESHOLD = 1.5


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _impossibility_points(candidate: Dict[str, Any]) -> tuple[float, list[str]]:
    """
    Returns (points, reasons). Each independent weak signal contributes
    a partial point. True honeypots should stack several of these on the
    same profile; random noise should mostly contribute 0-1 points.
    """
    points = 0.0
    flags = []

    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    education = candidate.get("education", [])
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    # --- Hard physical impossibilities (always real, but individually common-ish) ---

    # last_active before signup
    signup = _parse_date(signals.get("signup_date"))
    last_active = _parse_date(signals.get("last_active_date"))
    if signup and last_active and (signup - last_active).days > 7:
        points += 1.0
        flags.append("active_before_signup")

    # role end before start
    for role in career:
        start = _parse_date(role.get("start_date"))
        end = _parse_date(role.get("end_date")) if role.get("end_date") else None
        if start and end and end < start:
            points += 1.0
            flags.append(f"role_end_before_start:{role.get('company','?')}")

    # education end before start
    for edu in education:
        sy, ey = edu.get("start_year", 0), edu.get("end_year", 9999)
        if sy and ey and ey < sy:
            points += 1.0
            flags.append(f"edu_end_before_start:{edu.get('institution','?')}")

    # out-of-range fractions (schema violations)
    for field, val in [
        ("recruiter_response_rate", signals.get("recruiter_response_rate")),
        ("interview_completion_rate", signals.get("interview_completion_rate")),
    ]:
        if isinstance(val, (int, float)) and (val < 0 or val > 1):
            points += 1.0
            flags.append(f"{field}_out_of_range:{val}")

    oar = signals.get("offer_acceptance_rate")
    if isinstance(oar, (int, float)) and oar != -1 and (oar < 0 or oar > 1):
        points += 1.0
        flags.append(f"offer_acceptance_rate_out_of_range:{oar}")

    pcs = signals.get("profile_completeness_score")
    if isinstance(pcs, (int, float)) and (pcs < 0 or pcs > 100):
        points += 1.0
        flags.append(f"profile_completeness_out_of_range:{pcs}")

    gas = signals.get("github_activity_score")
    if isinstance(gas, (int, float)) and gas != -1 and (gas < 0 or gas > 100):
        points += 1.0
        flags.append(f"github_score_out_of_range:{gas}")

    # --- Softer / noisier signals (each contributes a FRACTION of a point) ---

    # salary min > max — common noise (~26%), weight low
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if isinstance(sal, dict):
        smin, smax = sal.get("min", 0), sal.get("max", 999)
        if isinstance(smin, (int, float)) and isinstance(smax, (int, float)) and smin > smax:
            points += 0.4
            flags.append("salary_min_gt_max")

    # YOE vs years-since-earliest-graduation mismatch — common noise, weight low
    yoe = profile.get("years_of_experience", 0)
    if education:
        earliest_end = min((e.get("end_year", 2026) for e in education), default=2026)
        years_since_grad = REFERENCE_DATE.year - earliest_end
        if years_since_grad >= 0 and yoe > years_since_grad + 5:
            points += 0.4
            flags.append(f"yoe_exceeds_grad_window:{yoe}vs{years_since_grad}")

    # total career months vs stated YOE — moderate weight, can be legitimately noisy
    total_months = sum(r.get("duration_months", 0) for r in career)
    if total_months / 12.0 > yoe + 5.0:
        points += 0.5
        flags.append(f"career_months_exceeds_yoe:{total_months/12:.1f}vs{yoe}")

    # expert/advanced skill with 0 duration AND 0 endorsements (double-zero is rarer than single)
    double_zero = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 1) == 0
        and s.get("endorsements", 1) == 0
    ]
    if double_zero:
        points += 0.6 * len(double_zero)
        flags.append(f"expert_zero_duration_zero_endorsements:{double_zero}")

    # assessment score wildly contradicts proficiency (expert but assessed near-zero)
    assessments = signals.get("skill_assessment_scores", {})
    severe_mismatches = 0
    for s in skills:
        sname = s.get("name", "")
        if sname in assessments:
            score = assessments[sname]
            prof = s.get("proficiency", "")
            if prof == "expert" and score < 20:
                severe_mismatches += 1
            elif prof == "advanced" and score < 12:
                severe_mismatches += 1
    if severe_mismatches >= 1:
        points += 0.5 * severe_mismatches
        flags.append(f"severe_assessment_mismatch_count:{severe_mismatches}")

    return points, flags


def is_honeypot(candidate: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Returns (is_honeypot, reasons).
    Flags only candidates whose stacked impossibility score clears
    IMPOSSIBILITY_THRESHOLD — i.e. multiple independent red flags on the
    SAME profile, not a single noisy field.
    """
    points, flags = _impossibility_points(candidate)
    return (points >= IMPOSSIBILITY_THRESHOLD), flags
