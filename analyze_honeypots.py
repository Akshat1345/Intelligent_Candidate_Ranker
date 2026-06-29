import sys, json
sys.path.insert(0, 'src')
from honeypot_detector import _impossibility_points
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else '../given/candidates.jsonl'

dist = Counter()
top_candidates = []

with open(path) as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        pts, flags = _impossibility_points(c)
        # bucket points to nearest 0.1
        bucket = round(pts, 1)
        dist[bucket] += 1
        if pts >= 1.5:
            top_candidates.append((pts, c['candidate_id'], flags))

print("Points distribution (bucket: count):")
for pts in sorted(dist.keys()):
    print(f"  {pts:.1f}: {dist[pts]:,}")

print()
print(f"Total candidates: {sum(dist.values()):,}")
print()
print("Candidates with points >= 1.5 (sorted by points desc):")
top_candidates.sort(reverse=True)
for pts, cid, flags in top_candidates[:50]:
    print(f"  {pts:.1f} | {cid} | {flags}")
print(f"\nTotal with points >= 1.5: {len(top_candidates)}")