"""
inspect_submission.py
----------------------
Quick sanity check on submission.csv before final submit.
Usage: python3 inspect_submission.py submission.csv
"""

import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

with open(path) as f:
    rows = list(csv.DictReader(f))

print(f"Total rows: {len(rows)}\n")

def show(label, slice_rows):
    print(f"=== {label} ===")
    for r in slice_rows:
        print(f"#{r['rank']} {r['candidate_id']} score={r['score']}")
        print(f"  {r['reasoning']}\n")

show("Ranks 1-5", rows[:5])
show("Ranks 40-55 (mid-tier check)", rows[39:55])
show("Ranks 90-100 (bottom of shortlist)", rows[89:100])

# Check for repetitive reasoning patterns
reasonings = [r["reasoning"] for r in rows]
unique_openings = set(" ".join(r.split()[:5]) for r in reasonings)
print(f"Unique opening phrases (first 5 words): {len(unique_openings)} / {len(reasonings)}")

# Check for exact duplicate reasoning (red flag if any)
seen = {}
dupes = []
for r in rows:
    if r["reasoning"] in seen:
        dupes.append((seen[r["reasoning"]], r["rank"]))
    else:
        seen[r["reasoning"]] = r["rank"]
if dupes:
    print(f"\nWARNING: {len(dupes)} exact duplicate reasoning pairs found:")
    for a, b in dupes[:10]:
        print(f"  rank {a} and rank {b} have identical reasoning")
else:
    print("\nNo exact duplicate reasoning found — good.")

# Score monotonicity check
scores = [float(r["score"]) for r in rows]
non_increasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
print(f"Scores non-increasing by rank: {non_increasing}")