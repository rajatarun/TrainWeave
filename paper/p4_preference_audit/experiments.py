"""Regenerate every number in paper 4 (preference-label audit). No model, no GPU.

The judge is simulated: with probability `bias` it answers "1" whatever the
order (position bias); otherwise it prefers the truly better response with
probability 1 (it is a good judge). The heuristic label agrees with the truly
better response with probability `p`. So the population agreement between the
heuristic and an unbiased judge is p, and the audit should say labels_hold
exactly when p is comfortably above 0.7.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "orchestrator"))
import label_audit as L  # noqa: E402


def make_pool(n, p, rng):
    pool = []
    for i in range(n):
        good, bad = f"GOOD {i}", f"bad {i}"
        agree = rng.random() < p
        pool.append({"prompt": f"t{i}", "chosen": good if agree else bad, "rejected": bad if agree else good,
                     "delta": rng.uniform(0.4, 1.2)})
    return pool


def judge(bias, rng):
    def invoke(prompt):
        if rng.random() < bias:
            return '{"better": "1"}'
        r1 = prompt.split("RESPONSE 1:\n", 1)[1].split("\n\nRESPONSE 2:", 1)[0]
        return '{"better": "1"}' if r1.startswith("GOOD") else '{"better": "2"}'
    return invoke


out = {"power": {}, "bias": {}}
TRIALS = 300
for n in (50, 100, 200):
    for p in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        hold = 0
        for t in range(TRIALS):
            rng = random.Random(100_000 * n + int(p * 1000) * 1000 + t)
            res = L.audit(make_pool(n, p, rng), judge(0.0, rng))
            hold += res.decision == "labels_hold"
        out["power"][f"n={n},p={p}"] = round(hold / TRIALS, 3)
for bias in (0.0, 0.2, 0.4, 0.6):
    stats = {"hold": 0, "consistency": 0.0, "decisive": 0.0, "agreement": 0.0}
    for t in range(TRIALS):
        rng = random.Random(7_000_000 + int(bias * 100) * 1000 + t)
        res = L.audit(make_pool(200, 0.85, rng), judge(bias, rng))
        stats["hold"] += res.decision == "labels_hold"
        stats["consistency"] += res.position_consistency or 0.0
        stats["decisive"] += res.decisive
        stats["agreement"] += res.agreement or 0.0
    out["bias"][str(bias)] = {"p_hold": round(stats["hold"] / TRIALS, 3),
                              "mean_consistency": round(stats["consistency"] / TRIALS, 3),
                              "mean_decisive": round(stats["decisive"] / TRIALS, 1),
                              "mean_agreement": round(stats["agreement"] / TRIALS, 3)}
print(json.dumps(out, indent=1))
