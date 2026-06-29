#!/usr/bin/env python3
"""
rank.py
-------
Redrob Hackathon — Intelligent Candidate Discovery & Ranking
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Produces a top-100 ranked CSV in the format required by submission_spec.md.
Runs in < 5 minutes on CPU with 16GB RAM, no GPU, no network required.

Architecture:
  Layer 0: Load all 100K candidates
  Layer 1: Honeypot detection → score = 0 for impossible profiles
  Layer 2: Hard disqualifiers from JD
  Layer 3: Technical scoring (skills, career, experience, trust, education)
  Layer 4: Behavioral multiplier (availability, engagement, platform trust)
  Layer 5: Sort, top-100, generate per-candidate reasoning
  Layer 6: Validate and write CSV
"""

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from honeypot_detector import is_honeypot
from technical_scorer import compute_technical_score
from behavioral_scorer import compute_behavioral_multiplier
from reasoning_generator import generate_reasoning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ranker")


def load_candidates(path: str) -> list[dict]:
    """Load candidates from .jsonl or .jsonl.gz file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    candidates = []
    opener = gzip.open if p.suffix == ".gz" else open
    mode = "rt" if p.suffix == ".gz" else "r"

    with opener(p, mode, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning(f"Line {i+1}: JSON parse error — {e}")

    log.info(f"Loaded {len(candidates):,} candidates from {p.name}")
    return candidates


def score_candidate(candidate: dict) -> dict:
    """
    Score a single candidate. Returns a dict with all scoring details.
    """
    cid = candidate.get("candidate_id", "UNKNOWN")

    # Layer 1: Honeypot check
    honeypot, honeypot_flags = is_honeypot(candidate)
    if honeypot:
        return {
            "candidate_id": cid,
            "final_score": 0.001,
            "tech_score": 0.0,
            "multiplier": 0.5,
            "tech_breakdown": {},
            "behav_breakdown": {},
            "is_honeypot": True,
            "honeypot_flags": honeypot_flags,
            "reasoning": "",
        }

    # Layer 2 + 3: Technical score
    tech_score, tech_breakdown = compute_technical_score(candidate)

    # Layer 4: Behavioral multiplier
    multiplier, behav_breakdown = compute_behavioral_multiplier(candidate)

    # Final score
    final_score = tech_score * multiplier

    return {
        "candidate_id": cid,
        "final_score": round(final_score, 6),
        "tech_score": round(tech_score, 6),
        "multiplier": round(multiplier, 6),
        "tech_breakdown": tech_breakdown,
        "behav_breakdown": behav_breakdown,
        "is_honeypot": False,
        "honeypot_flags": [],
        "reasoning": "",
    }


def run_pipeline(candidates_path: str, output_path: str) -> None:
    t0 = time.time()

    # --- Load ---
    log.info("Loading candidates...")
    candidates = load_candidates(candidates_path)
    log.info(f"Loaded in {time.time()-t0:.1f}s")

    # --- Score all ---
    log.info("Scoring all candidates (technical + behavioral)...")
    t1 = time.time()
    scored = []
    honeypot_count = 0
    for i, c in enumerate(candidates):
        result = score_candidate(c)
        result["_candidate_obj"] = c  # keep ref for reasoning generation
        scored.append(result)
        if result["is_honeypot"]:
            honeypot_count += 1
        if (i + 1) % 10000 == 0:
            log.info(f"  Scored {i+1:,} / {len(candidates):,} ...")

    log.info(
        f"Scoring complete in {time.time()-t1:.1f}s | "
        f"Honeypots detected: {honeypot_count:,}"
    )

    # --- Sort ---
    scored.sort(key=lambda x: x["final_score"], reverse=True)

    # --- Top 100 ---
    top100 = scored[:100]

    # Verify no honeypots in top 100
    hp_in_top100 = [r for r in top100 if r["is_honeypot"]]
    if hp_in_top100:
        log.warning(f"WARNING: {len(hp_in_top100)} honeypot(s) in top 100 — check scoring")

    # --- Generate reasoning for top 100 ---
    log.info("Generating per-candidate reasoning...")
    for rank_0based, result in enumerate(top100):
        rank = rank_0based + 1
        result["reasoning"] = generate_reasoning(
            candidate=result["_candidate_obj"],
            rank=rank,
            final_score=result["final_score"],
            tech_breakdown=result["tech_breakdown"],
            behav_breakdown=result["behav_breakdown"],
        )

    # --- Write CSV ---
    log.info(f"Writing output to {output_path}")
    rows = []
    for rank_0based, result in enumerate(top100):
        rank = rank_0based + 1
        rows.append({
            "candidate_id": result["candidate_id"],
            "rank": rank,
            "score": result["final_score"],
            "reasoning": result["reasoning"],
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    log.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"Top-3 candidates:")
    for r in rows[:3]:
        log.info(f"  Rank {r['rank']}: {r['candidate_id']}  score={r['score']:.4f}")
        log.info(f"    {r['reasoning']}")

    # Score distribution
    scores = [r["score"] for r in rows]
    log.info(
        f"Score range: top={scores[0]:.4f}  rank{min(50,len(scores))}={scores[min(49,len(scores)-1)]:.4f}  rank{len(scores)}={scores[-1]:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Hackathon — Candidate Ranker"
    )
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output path for ranked CSV",
    )
    args = parser.parse_args()
    run_pipeline(args.candidates, args.out)


if __name__ == "__main__":
    main()
