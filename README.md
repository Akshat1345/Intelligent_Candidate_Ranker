# Redrob Hackathon — Intelligent Candidate Ranker

Deterministic candidate-ranking system for the Redrob Hackathon challenge, built to shortlist the 100 best matches for the Senior AI Engineer founding-team role.

This repository is designed for the exact submission flow described by the challenge:

1. Read `candidates.jsonl`.
2. Rank all 100,000 candidates with no network calls and no GPU.
3. Produce a top-100 CSV in the required format: `candidate_id, rank, score, reasoning`.
4. Validate the CSV before submission.

The implementation is fully CPU-bound, reproducible, and tuned for the JD's explicit signals rather than generic keyword similarity.

## Why this approach wins this challenge

The challenge is not asking for abstract semantic similarity. It is asking for recruiter-grade judgment under strong dataset noise:

- role descriptions are shuffled and cannot be trusted as the primary signal,
- skills can be keyword-stuffed,
- some candidates are honeypots with stacked impossibilities,
- and the JD itself gives unusually explicit filters and preferences.

This project treats those constraints as the core problem. It uses a deterministic pipeline that combines title-based career analysis, JD-aware skill matching, behavioral availability signals, and recruiter-style reasoning text.

## What’s included

- `rank.py` — main entry point that loads candidates, scores them, sorts them, and writes the submission CSV.
- `src/honeypot_detector.py` — stacked impossibility scoring for honeypots and other invalid profiles.
- `src/technical_scorer.py` — technical relevance scoring across skills, AI career depth, experience fit, trust, trajectory, and education.
- `src/behavioral_scorer.py` — availability, engagement, platform trust, GitHub activity, and location-aware multiplier.
- `src/reasoning_generator.py` — two-sentence, recruiter-style reasoning for every shortlisted candidate.
- `inspect_submission.py` — sanity-check tool that prints the top, middle, and bottom of the submission plus duplicate/uniqueness checks.
- `config/scoring.yaml` — documented calibration values and thresholds used during development.
- `requirements.txt` — minimal runtime dependencies.

## Core architecture

The ranker is a four-layer deterministic pipeline.

### Layer 1: Honeypot detection

Honeypots are detected with a stacked impossibility scoring model. Individual signals are noisy, so the detector only flags candidates when several weak contradictions stack together. The calibrated threshold is intentionally kept at `1.5` impossibility points.

This layer exists to eliminate impossible profiles without harming legitimate candidates that happen to contain one noisy field.

### Layer 2: Hard disqualifiers

The JD explicitly rules out a few profile classes. The ranker hard-disqualifies only the cases the JD is explicit about, including:

- pure non-tech careers,
- CV / speech / robotics specialists with no NLP or IR exposure,
- pure consulting-only careers in the firm list the JD names,
- and other clear no-fit patterns that the challenge text calls out directly.

### Layer 3: Technical score

The technical score is the main relevance signal. It combines:

- JD-required core skills,
- AI career depth from role titles, not descriptions,
- career substance and product-company exposure,
- experience fit for the 5-9 year band,
- skill trust to resist keyword stuffing,
- career trajectory,
- and education tier as a light tie-breaker.

The current scoring logic is intentionally title-aware because the dataset shuffles role descriptions. That means a misleading description is down-weighted, while the role title remains trustworthy.

### Layer 4: Behavioral multiplier

Behavioral signals are applied multiplicatively to avoid rewarding an unavailable candidate too highly. The multiplier combines:

- availability: open-to-work, recency, notice period,
- engagement: recruiter response rate, interview completion, profile completeness,
- platform trust: verification, GitHub activity, assessments,
- and a small location preference adjustment aligned with the JD.

This keeps the final shortlist grounded in who a recruiter can actually hire.

## Technical scoring details

`src/technical_scorer.py` uses a weighted composite score with the following structure:

| Component | Purpose |
|---|---|
| `core_skills_match` | Match against JD must-haves: embeddings, vector DB, ranking / retrieval, Python, evaluation frameworks |
| `ai_career_depth` | Proportion of career spent in core AI / ML titles based on title sequence |
| `career_substance` | Product-company vs services exposure and production-relevant work |
| `experience_fit` | Fit for the JD's preferred experience band |
| `skill_trust` | Endorsements, duration, proficiency, and assessment consistency |
| `trajectory` | Promotions and progression within AI roles |
| `education_tier` | Light support signal only |

The scorer also applies a LangChain-wrapper penalty when a profile looks like thin wrapper work without real production depth.

### Why title-based AI depth matters

The JD says it wants someone with years of applied ML / AI work at product companies. In this dataset, titles are more reliable than descriptions. The code therefore measures AI career depth from role titles and the amount of time spent in core AI / ML titles, instead of trusting potentially shuffled descriptions.

### Why skill trust is stricter

The dataset contains keyword-stuffed profiles with AI skills but no backing duration or endorsements. The trust score therefore heavily discounts advanced / expert claims that are not supported by duration, endorsements, and assessment scores.

## Behavioral scoring details

`src/behavioral_scorer.py` produces a multiplier in the range `[0.5, 1.5]`.

It includes:

- availability from open-to-work, last active date, notice period, and relocation willingness,
- engagement from response rate, interview completion, profile completeness, and recent activity,
- platform trust from verification signals, assessment consistency, and GitHub activity,
- and a small location bias for India, especially the JD-preferred cities.

GitHub activity gets an explicit bonus because the JD calls out open-source contributions as a differentiator.

## Reasoning quality

Stage 4 of the hackathon evaluates the reasoning column manually. The generator in `src/reasoning_generator.py` is written to satisfy that review:

- every reasoning includes the candidate's company name,
- every reasoning includes at least two specific skills,
- the first sentence is specific and rank-appropriate,
- the second sentence always contains either a concrete concern or a concrete positive signal with numbers,
- and the wording avoids repeated mechanical phrases.

The output is intentionally recruiter-like, not generic model prose.

## Repository layout

```text
ranker/
├── README.md
├── rank.py
├── requirements.txt
├── inspect_submission.py
├── submission.csv                # generated locally
├── test_candidates.jsonl         # small smoke-test set
├── test_submission.csv           # sample expected output
├── config/
│   └── scoring.yaml
└── src/
    ├── behavioral_scorer.py
    ├── honeypot_detector.py
    ├── reasoning_generator.py
    └── technical_scorer.py
```

The challenge bundle in this workspace is stored separately under `../given/` and contains the official dataset, validator, and sample files.

## Requirements

- Python 3.11+ recommended
- macOS / Linux / Windows supported
- CPU only
- No network access required during ranking

Install dependencies:

```bash
pip install -r requirements.txt
```

The runtime dependency list is intentionally small. The ranker uses only the Python standard library plus `PyYAML` and `pytest` for development convenience.

## Quick start

If you have the challenge bundle next to this repo, the fastest full run is:

```bash
python rank.py --candidates ../given/candidates.jsonl --out ./submission.csv
python ../given/validate_submission.py ./submission.csv
python inspect_submission.py ./submission.csv
```

If you copied the dataset into the repo root instead, use:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
python inspect_submission.py ./submission.csv
```

## Smoke test

Before the full 100K run, use the provided 50-candidate test file:

```bash
python rank.py --candidates ./test_candidates.jsonl --out ./test_submission.csv
python inspect_submission.py ./test_submission.csv
```

That gives a fast sanity check for score ordering and reasoning quality.

## Full submission workflow

1. Run the ranker on the full challenge dataset.
2. Validate the CSV with the official validator.
3. Inspect the top, middle, and bottom of the shortlist.
4. Confirm that scores are non-increasing and reasonings are not duplicated.
5. Upload the final CSV, deck PDF, and repository link as required by the hackathon.

## Expected output

The submission CSV contains exactly four columns:

| Column | Description |
|---|---|
| `candidate_id` | Original candidate identifier from the challenge data |
| `rank` | 1-based rank in descending score order |
| `score` | Final score after technical score × behavioral multiplier |
| `reasoning` | Two-sentence recruiter-style rationale |

The validator expects the rows to be sorted by descending score with no format violations.

## Observed performance

On the provided 100K candidate dataset, the current build completed in about 38 seconds on CPU and detected 66 honeypots.

Observed output quality checks on the generated submission:

- no exact duplicate reasoning strings,
- 95 unique opening phrases across the 100 shortlisted candidates,
- non-increasing score order,
- valid CSV format according to the official validator.

These are empirical results from the current codebase, not guaranteed runtime on every machine.

## Why this beats pure semantic search

Embeddings and cross-encoders are good tools, but this challenge is dominated by noisy structured signals:

- shuffled descriptions,
- explicit disqualifiers,
- skill stuffing,
- career-stage filtering,
- and availability / responsiveness.

For this problem, a calibrated deterministic scorer is easier to inspect, faster to run, and better aligned with the scoring rubric than a generic similarity model.

## Troubleshooting

- If `rank.py` cannot find the dataset, check the path you pass to `--candidates`.
- If the validator is missing, make sure you are pointing to the official challenge bundle under `../given/` or have copied it into the repo root.
- If the CSV looks odd, run `inspect_submission.py` to inspect the top 100 and check for duplicates or ordering issues.
- If you modify scoring logic, always re-run the smoke test before the full 100K pass.

## Submission checklist

- `submission.csv` generated from the full candidate set
- `validate_submission.py` passes
- `inspect_submission.py` shows non-duplicated, diverse reasoning
- deck PDF completed
- repository pushed to GitHub
- metadata file completed if required by the submission portal

## Notes for reviewers

This codebase is intentionally transparent. Every signal used in ranking maps back to an explicit JD requirement or an explicit dataset trap. The goal is not to maximize superficial similarity, but to approximate how a strong recruiter would shortlist candidates for a production AI hiring role.

## License

Built for the Redrob Hackathon challenge workspace. Use according to the hackathon rules and repository policy.
