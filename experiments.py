"""
Reproducible experiments
========================
Everything the app and README report can be regenerated from here, with fixed seeds, exact scoring
(`solver.py`) and the software versions recorded next to the results.

    python experiments.py race     [--seeds 0 1 2] [--budget 1500000] [--dqn-budget 200000] [--rule S17]
    python experiments.py report   # exact grades of the shipped agents -> results/report.md
    python experiments.py export   # exact value tables and agent Q-tables as CSV -> results/

`python blackjack_rl.py train` / `refine` rebuild the shipped agents themselves.
"""

import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool

import algorithms as alg
import blackjack_rl as rl
import solver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RULES = {"S17": False, "H17": True}


def environment():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {"python": platform.python_version(), "platform": platform.platform(), "git_commit": commit,
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


# ---------------------------------------------------------
# Algorithm race over several seeds
# ---------------------------------------------------------

def _one_seed(job):
    seed, budget, dqn_budget, hs17 = job
    t0 = time.time()
    out = alg.race(budget, dqn_budget, hs17, seed=seed)
    return seed, out, time.time() - t0


def _aggregate(runs):
    """Mean and standard deviation across seeds for every checkpoint and metric."""
    first = runs[0]
    merged = {}
    for name, rows in first.items():
        merged[name] = []
        for k, row in enumerate(rows):
            agg = {"hands": row["hands"]}
            for key in row:
                if key == "hands":
                    continue
                vals = [r[name][k][key] for r in runs]
                mean = sum(vals) / len(vals)
                sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
                agg[key], agg[key + "_sd"] = mean, sd
            merged[name].append(agg)
    return merged


def cmd_race(args):
    hs17 = RULES[args.rule]
    jobs = [(s, args.budget, args.dqn_budget, hs17) for s in args.seeds]
    t0 = time.time()
    with Pool(min(len(jobs), args.workers)) as pool:
        done = pool.map(_one_seed, jobs)
    runs = [out for _, out, _ in sorted(done)]
    results = _aggregate(runs)
    meta = {"seeds": args.seeds, "budget": args.budget, "dqn_budget": args.dqn_budget, "rule": args.rule,
            "rules": solver.AGENT_RULES[args.rule].label(),
            "perfect_play_per_100": solver.house_edge(solver.AGENT_RULES[args.rule]) * 100,
            "scoring": "exact (solver.py): no simulation noise", "seconds": round(time.time() - t0),
            **environment()}
    alg.save_race(results, meta=meta)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "race_by_seed.json"), "w") as f:
        json.dump({"meta": meta, "runs": {str(s): r for s, r in zip(args.seeds, runs)}}, f, indent=1)
    print(f"{len(args.seeds)} seeds in {meta['seconds']}s  (perfect play {meta['perfect_play_per_100']:+.3f} per $100)")
    for name, rows in results.items():
        r = rows[-1]
        print(f"  {name:>24}: {r['hands']:>9,} hands  optimal {r['optimal']:.1%} ±{r['optimal_sd']:.1%}  "
              f"gap {r['regret_100'] * 100:.2f}¢ ±{r['regret_100_sd'] * 100:.2f} per $100")


# ---------------------------------------------------------
# Report on the shipped agents
# ---------------------------------------------------------

def grade_bundle(rule):
    b = rl.load_bundle(os.path.join(BASE_DIR, f"q_expert_{rule}.pkl"))
    Q = rl.robust_q(b["Q"], b["N"])
    r = solver.regret(solver.greedy_policy(Q), solver.AGENT_RULES[rule])
    return b, r


def cmd_report(_args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    lines = ["# Royal Blackjack — exact grade of the shipped agents", "",
             f"Generated {environment()['date']} (commit {environment()['git_commit'] or 'n/a'}).", "",
             "All numbers are exact expected values from `solver.py` (infinite deck), not simulations.", ""]
    rows = []
    for rule in RULES:
        b, r = grade_bundle(rule)
        wrong = [d for d in r["decisions"] if d["cost"] > 1e-12]
        ref = b.get("refined") or {}
        lines += [f"## Dealer {rule} ({solver.AGENT_RULES[rule].label()})", "",
                  f"* Training: {b['hands']:,} hands of Monte Carlo self-play"
                  + (f" + {ref['deals']:,} precision-practice deals ({ref['settled']}/{ref['total']} decisions "
                     f"settled at z={ref['z']}, tie tolerance {ref['tol']})" if ref else ""),
                  f"* Perfect play: **{r['optimal_ev'] * 100:+.4f}** per $100",
                  f"* Agent: **{r['policy_ev'] * 100:+.4f}** per $100",
                  f"* Gap to perfect play: **{r['regret'] * 10000:.2f}¢** per $100",
                  f"* Perfect move on **{r['optimal_share']:.1%}** of the 330 first decisions", ""]
        if wrong:
            lines += ["| Hand | Dealer | Agent | Perfect | Cost per $1 | Frequency |", "|---|---|---|---|---|---|"]
            for d in sorted(wrong, key=lambda d: -d["cost"] * d["weight"]):
                lines.append(f"| {d['kind']} {d['total']} | {d['up']} | {d['agent']} | {d['optimal']} | "
                             f"{d['cost'] * 100:.3f}¢ | {d['weight'] * 100:.3f}% |")
            lines.append("")
        for d in r["decisions"]:
            rows.append({"rule": rule, "kind": d["kind"], "total": d["total"], "dealer": d["up"],
                         "agent": d["agent"], "optimal": d["optimal"], "cost": d["cost"], "frequency": d["weight"],
                         **{f"ev_{k}": v for k, v in d["values"].items()}})
    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    _write_csv(os.path.join(RESULTS_DIR, "expert_decisions.csv"), rows)
    print("\n".join(lines))


# ---------------------------------------------------------
# Exports
# ---------------------------------------------------------

def _write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def exact_table_rows(rules):
    rows = []
    for kind, t, state in solver.two_card_states():
        vals = solver.action_values(state, rules)
        best = max(vals, key=vals.get)
        rows.append({"kind": kind, "total": t, "dealer": state[1], "state": str(state),
                     "best": solver.NAMES[best], **{f"ev_{solver.NAMES[a]}": v for a, v in vals.items()}})
    return rows


def q_table_rows(rule):
    b = rl.load_bundle(os.path.join(BASE_DIR, f"q_expert_{rule}.pkl"))
    rows = []
    for state, q in sorted(b["Q"].items()):
        n = b["N"].get(state, [0] * 4)
        rows.append({"total": state[0], "dealer": state[1], "soft": state[2], "can_double": state[3],
                     "pair": state[4], **{f"q_{a}": q[i] for i, a in enumerate(rl.ACTIONS)},
                     **{f"n_{a}": n[i] for i, a in enumerate(rl.ACTIONS)}})
    return rows


def cmd_export(_args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for rule in RULES:
        _write_csv(os.path.join(RESULTS_DIR, f"exact_values_{rule}.csv"), exact_table_rows(solver.AGENT_RULES[rule]))
        _write_csv(os.path.join(RESULTS_DIR, f"agent_q_table_{rule}.csv"), q_table_rows(rule))
    print(f"wrote CSV files to {RESULTS_DIR}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("race", help="multi-seed algorithm comparison, scored exactly")
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    r.add_argument("--budget", type=int, default=1_500_000)
    r.add_argument("--dqn-budget", type=int, default=200_000)
    r.add_argument("--rule", choices=list(RULES), default="S17")
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    sub.add_parser("report", help="exact grade of the shipped agents")
    sub.add_parser("export", help="CSV exports of exact values and agent Q-tables")
    args = ap.parse_args(argv)
    {"race": cmd_race, "report": cmd_report, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
