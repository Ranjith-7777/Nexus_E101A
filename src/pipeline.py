from __future__ import annotations
import os, json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

from src.utils import parse_dt, clamp, sigmoid

# ---- Tunables (MVP) ----
W_DELAY = 0.5
W_BLOCKED = 0.3
W_REWORK = 0.2

P_MAX = 0.25
LAMBDA_EVIDENCE = 0.15
MIN_PEER_CONFIRM = 2

SPRINT_DAYS = 14
MONTH_DAYS = 30
ROLLING_DAYS = 56

def load_csvs(data_dir: str) -> Dict[str, pd.DataFrame]:
    def read(name: str) -> pd.DataFrame:
        path = os.path.join(data_dir, f"{name}.csv")
        return pd.read_csv(path)
    return {
        "users": read("users"),
        "tasks": read("tasks"),
        "commits": read("commits"),
        "messages": read("messages"),
        "handoffs": read("handoffs"),
        "evidence": read("evidence"),
    }

def tenure_bucket(days: int) -> str:
    if days <= 14: return "0-14"
    if days <= 60: return "15-60"
    if days <= 180: return "61-180"
    return "180+"

def build_task_graph(tasks: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    for _, r in tasks.iterrows():
        g.add_node(r["task_id"])
    def split_ids(s):
        if pd.isna(s) or str(s).strip()=="":
            return []
        return [x.strip() for x in str(s).split(",") if x.strip()]
    for _, r in tasks.iterrows():
        t = r["task_id"]
        for dep in split_ids(r.get("depends_on","")) + split_ids(r.get("blocked_by","")):
            if dep in g.nodes:
                g.add_edge(dep, t)
    return g

def concurrency_score(tasks: pd.DataFrame) -> float:
    intervals = []
    for _, r in tasks.iterrows():
        s = parse_dt(r.get("started_at"))
        e = parse_dt(r.get("completed_at"))
        if s and e and e > s:
            intervals.append((s, e))
    if len(intervals) < 2:
        return 0.0
    intervals.sort()
    # total active time (union length) and overlapped time (sum overlaps)
    total = 0.0
    overlap = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            overlap += (min(cur_e, e) - s).total_seconds()
            cur_e = max(cur_e, e)
        else:
            total += (cur_e - cur_s).total_seconds()
            cur_s, cur_e = s, e
    total += (cur_e - cur_s).total_seconds()
    return float(overlap / max(1.0, total))

def dep_density(g: nx.DiGraph) -> float:
    v = g.number_of_nodes()
    e = g.number_of_edges()
    return float(e / max(1, v))

def is_sequential_mode(g: nx.DiGraph, conc: float) -> bool:
    return (dep_density(g) > 0.8) and (conc < 0.2)

def topo_samples(g: nx.DiGraph, k: int = 10) -> List[List[str]]:
    if not nx.is_directed_acyclic_graph(g):
        # fallback: ignore sampling if cyclic
        return [list(g.nodes)]
    samples = []
    nodes = list(g.nodes)
    for _ in range(k):
        # random topological sort via randomized Kahn
        indeg = {n: g.in_degree(n) for n in nodes}
        zero = [n for n in nodes if indeg[n] == 0]
        order = []
        while zero:
            n = np.random.choice(zero)
            zero.remove(n)
            order.append(n)
            for nxt in g.successors(n):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    zero.append(nxt)
        if len(order) == len(nodes):
            samples.append(order)
    return samples or [list(g.nodes)]

def base_metrics(tasks: pd.DataFrame) -> Tuple[float, int, float]:
    # Delay: sum cycle times in hours
    delay = 0.0
    for _, r in tasks.iterrows():
        s = parse_dt(r.get("started_at")) or parse_dt(r.get("created_at"))
        e = parse_dt(r.get("completed_at"))
        if s and e and e > s:
            delay += (e - s).total_seconds() / 3600.0
    blocked = int((tasks["status"].astype(str).str.lower() == "blocked").sum())
    rework_proxy = 0.0  # MVP: keep 0 unless you add reopen data
    return delay, blocked, rework_proxy

def counterfactual_for_user(tasks: pd.DataFrame, g: nx.DiGraph, user_id: str,
                            sequential: bool, K: int = 10) -> Tuple[float, int, float]:
    base_delay, base_blocked, base_rework = base_metrics(tasks)

    # Replacement delays (MVP constants)
    DELAY_TASK_HOURS = 24.0
    DELAY_BLOCK_HOURS = 12.0

    def simulate_one(order: List[str]) -> Tuple[float, int, float]:
        # copy tasks
        tdf = tasks.copy()
        # if user completed a task, push its completion later (simple proxy)
        for idx, r in tdf.iterrows():
            if str(r.get("assignee_id")) == user_id:
                comp = parse_dt(r.get("completed_at"))
                if comp:
                    tdf.at[idx, "completed_at"] = (comp + pd.Timedelta(hours=DELAY_TASK_HOURS)).isoformat()
        # if task depends on a user-owned task, add small extra delay to downstream (proxy for blocking)
        # simplistic: for each edge u_task -> v_task where u_task owned by user, delay v_task completion
        owned = set(tdf.loc[tdf["assignee_id"].astype(str)==user_id, "task_id"].astype(str).tolist())
        for u, v in g.edges():
            if u in owned:
                vrows = tdf.index[tdf["task_id"].astype(str)==str(v)].tolist()
                for idx in vrows:
                    comp = parse_dt(tdf.at[idx, "completed_at"])
                    if comp:
                        tdf.at[idx, "completed_at"] = (comp + pd.Timedelta(hours=DELAY_BLOCK_HOURS)).isoformat()

        cf_delay, cf_blocked, cf_rework = base_metrics(tdf)
        return (cf_delay - base_delay), (cf_blocked - base_blocked), (cf_rework - base_rework)

    if sequential:
        dD, dB, dR = simulate_one(list(g.nodes))
        return dD, dB, dR

    # fairness mode: average across sampled valid orders
    samples = topo_samples(g, k=K)
    deltas = [simulate_one(o) for o in samples]
    dD = float(np.mean([x[0] for x in deltas]))
    dB = int(round(np.mean([x[1] for x in deltas])))
    dR = float(np.mean([x[2] for x in deltas]))
    return dD, dB, dR

def integrity_penalty(users: pd.DataFrame, commits: pd.DataFrame, messages: pd.DataFrame, handoffs: pd.DataFrame) -> Dict[str, dict]:
    # compute per user
    out = {}
    # commits fragmentation
    commits["lines_changed"] = commits["lines_added"].fillna(0) + commits["lines_deleted"].fillna(0)
    frag = commits.groupby("user_id").apply(lambda df: len(df) / max(1.0, float(df["lines_changed"].sum()))).to_dict()

    # message flooding + info density (MVP heuristic: meaningful if len(text) >= 12)
    messages["meaningful"] = messages["text"].astype(str).apply(lambda t: 1 if len(t.strip()) >= 12 else 0)
    msg_rate = messages.groupby("user_id").size().to_dict()
    info_density = messages.groupby("user_id")["meaningful"].mean().fillna(0).to_dict()

    # routing inflation
    handoff_rate = {}
    if len(handoffs) > 0:
        handoff_rate = handoffs.groupby("to_user_id").size().to_dict()

    # team baselines (simple mean/std over users present)
    uids = users["user_id"].astype(str).tolist()
    def mean_std(dct, default=0.0):
        vals = [float(dct.get(u, default)) for u in uids]
        return float(np.mean(vals)), float(np.std(vals) + 1e-9)

    frag_mu, frag_sd = mean_std(frag, 0.0)
    msg_mu, msg_sd = mean_std(msg_rate, 0.0)

    for u in uids:
        flags = []
        score = 0.0

        f = float(frag.get(u, 0.0))
        m = float(msg_rate.get(u, 0.0))
        iden = float(info_density.get(u, 0.0))
        h = float(handoff_rate.get(u, 0.0))

        flag_frag = (f > frag_mu + 2 * frag_sd) and (f > 0.05)
        flag_flood = (m > msg_mu + 2 * msg_sd) and (iden < 0.35) and (m > 30)
        flag_route = (h > 10)  # MVP threshold

        if flag_frag: flags.append("commit_fragmentation")
        if flag_flood: flags.append("message_flooding_low_info")
        if flag_route: flags.append("routing_inflation")

        # soft penalty: more flags -> higher
        score = sigmoid(1.2*(1 if flag_frag else 0) + 1.2*(1 if flag_flood else 0) + 0.8*(1 if flag_route else 0)) - 0.5
        penalty = clamp(0.0, score, P_MAX)

        out[u] = {"penalty": penalty, "flags": flags}
    return out

def evidence_scores(evidence: pd.DataFrame) -> Dict[str, dict]:
    out = {}
    if evidence is None or len(evidence) == 0:
        return out
    for _, r in evidence.iterrows():
        u = str(r["user_id"])
        peers = str(r.get("peer_confirmations", "")).strip()
        peer_list = [p.strip() for p in peers.split(",") if p.strip()]
        if len(peer_list) >= MIN_PEER_CONFIRM:
            strength = min(1.0, len(peer_list)/4.0)
            out[u] = out.get(u, {"peer_validated_claims": 0, "invisible_score": 0.0})
            out[u]["peer_validated_claims"] += 1
            out[u]["invisible_score"] += LAMBDA_EVIDENCE * strength
    return out

def role_scores(features: Dict[str, dict], cf: Dict[str, dict]) -> Dict[str, dict]:
    out = {}
    for u, f in features.items():
        # pull simple proxies
        tasks_done = f["tasks_done"]
        msgs = f["messages"]
        commits = f["commits"]
        handoffs_in = f["handoffs_in"]
        dDelay = cf[u]["dDelay"]

        # role heuristics (MVP, explainable)
        silent_arch = 0.6 * max(0.0, dDelay) + 0.4 * max(0.0, (handoffs_in + 1) / (msgs + 1))  # quiet but impactful
        quality_guard = 0.7 * max(0.0, (commits + 1) / (tasks_done + 1)) + 0.3 * max(0.0, dDelay*0.1)
        visible_driver = 0.8 * tasks_done + 0.2 * msgs
        router = 0.7 * handoffs_in + 0.3 * msgs

        scores = {
            "Silent Architect": float(silent_arch),
            "Quality Guardian": float(quality_guard),
            "Visible Driver": float(visible_driver),
            "Router / Coordinator": float(router),
        }
        # top2
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top1, top2 = ordered[0], ordered[1]
        # confidence by gap
        gap = top1[1] - top2[1]
        conf = float(clamp(0.35, gap / (abs(top1[1]) + 1e-6), 0.9))
        out[u] = {"top_roles": [{"name": top1[0], "confidence": conf}, {"name": top2[0], "confidence": 1-conf}]}
    return out

def compute_user_features(users, tasks, commits, messages, handoffs) -> Dict[str, dict]:
    uids = users["user_id"].astype(str).tolist()
    out = {}
    for u in uids:
        out[u] = {
            "tasks_done": int((tasks["assignee_id"].astype(str)==u).sum()),
            "commits": int((commits["user_id"].astype(str)==u).sum()) if len(commits)>0 else 0,
            "messages": int((messages["user_id"].astype(str)==u).sum()) if len(messages)>0 else 0,
            "handoffs_in": int((handoffs["to_user_id"].astype(str)==u).sum()) if len(handoffs)>0 else 0,
        }
    return out

def band_from_score(score: float, ramp_up: bool) -> str:
    if ramp_up:
        return "Ramp-Up Mode"
    if score <= 30: return "Emerging Contributor"
    if score <= 45: return "Supporting Contributor"
    if score <= 60: return "Core Contributor"
    if score <= 80: return "Key Contributor"
    return "Critical Contributor"

def main(data_dir: str, output_path: str = "output.json"):
    dfs = load_csvs(data_dir)
    users = dfs["users"]
    tasks = dfs["tasks"]
    commits = dfs["commits"]
    messages = dfs["messages"]
    handoffs = dfs["handoffs"]
    evidence = dfs["evidence"]

    g = build_task_graph(tasks)
    conc = concurrency_score(tasks)
    sequential = is_sequential_mode(g, conc)

    u_features = compute_user_features(users, tasks, commits, messages, handoffs)
    penalties = integrity_penalty(users, commits, messages, handoffs)
    evid = evidence_scores(evidence)

    # counterfactual per user
    cf = {}
    for u in users["user_id"].astype(str).tolist():
        dD, dB, dR = counterfactual_for_user(tasks, g, u, sequential=sequential, K=10)
        cf[u] = {"dDelay": float(max(0.0, dD)), "dBlocked": int(max(0, dB)), "dRework": float(max(0.0, dR))}

    # raw impact
    raw = {}
    for u in cf:
        raw[u] = (W_DELAY*cf[u]["dDelay"] + W_BLOCKED*cf[u]["dBlocked"] + W_REWORK*cf[u]["dRework"])

    # apply penalties + evidence
    impact = {}
    for u in raw:
        pen = float(penalties.get(u, {}).get("penalty", 0.0))
        inv = float(evid.get(u, {}).get("invisible_score", 0.0))
        impact[u] = raw[u] * (1.0 - pen) + inv

    # tenure normalization (simple)
    now = datetime.now().date()
    users["tenure_days"] = users["join_date"].apply(lambda d: (now - pd.to_datetime(d).date()).days)
    users["tenure_bucket"] = users["tenure_days"].apply(tenure_bucket)

    expected_by_bucket = users.assign(imp=users["user_id"].astype(str).map(impact)).groupby("tenure_bucket")["imp"].median().to_dict()
    team_expected = float(np.median(list(impact.values()))) if impact else 1.0

    impact_norm = {}
    ramp = {}
    for _, r in users.iterrows():
        u = str(r["user_id"])
        b = str(r["tenure_bucket"])
        exp = float(expected_by_bucket.get(b, team_expected))
        impact_norm[u] = float(impact[u] / max(1e-6, exp))
        ramp[u] = (int(r["tenure_days"]) < 14)

    # map to 0-100 within team
    vals = np.array(list(impact_norm.values()), dtype=float)
    mn, mx = float(vals.min()), float(vals.max())
    scores = {}
    for u, v in impact_norm.items():
        if mx - mn < 1e-9:
            scores[u] = 50.0
        else:
            scores[u] = float(100.0 * (v - mn) / (mx - mn))

    roles = role_scores(u_features, cf)

    # explainability + recourse (MVP rules)
    results = []
    for _, r in users.iterrows():
        u = str(r["user_id"])
        sc = float(scores[u])
        band = band_from_score(sc, ramp_up=ramp[u])
        pen = penalties.get(u, {"penalty": 0.0, "flags": []})
        ev = evid.get(u, {"peer_validated_claims": 0, "invisible_score": 0.0})

        # top factors (simple contributions)
        factors_pos = []
        if cf[u]["dDelay"] > 0: factors_pos.append("Prevents downstream delays (counterfactual)")
        if cf[u]["dBlocked"] > 0: factors_pos.append("Unblocks dependencies")
        if ev.get("invisible_score", 0.0) > 0: factors_pos.append("Peer-validated invisible work")

        factors_neg = []
        if pen["penalty"] > 0: factors_neg.append(f"Integrity dampening applied ({', '.join(pen['flags'])})")

        recourse = []
        if "message_flooding_low_info" in pen["flags"]:
            recourse.append("Improve message information density (fewer low-content messages).")
        if "commit_fragmentation" in pen["flags"]:
            recourse.append("Avoid splitting trivial commits; batch meaningful changes.")
        if cf[u]["dDelay"] == 0 and not ramp[u]:
            recourse.append("Pick dependency-resolving tasks to increase system impact.")

        results.append({
            "user_id": u,
            "name": str(r["name"]),
            "score": round(sc, 2),
            "band": band,
            "roles": roles[u]["top_roles"],
            "trend": "Stable",
            "explanations": {"top_positive": factors_pos[:3], "top_negative": factors_neg[:3]},
            "integrity": {"penalty": round(float(pen["penalty"]), 3), "flags": pen["flags"]},
            "evidence": {"peer_validated_claims": int(ev.get("peer_validated_claims", 0)), "invisible_score": round(float(ev.get("invisible_score", 0.0)), 3)},
            "confidence": "medium",
            "guardrails": ["Do not use as sole metric. Combine with qualitative review."]
        })

    out = {
        "meta": {
            "mode": "sequential" if sequential else "fairness_sampling",
            "concurrency_score": round(float(conc), 3),
            "dependency_density": round(dep_density(g), 3),
        },
        "team_summary": {
            "members": len(results),
        },
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {output_path} from data_dir={data_dir}")

if __name__ == "__main__":
    # Example run:
    # python src/pipeline.py data/demo/scenario_silent_architect
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "demo", "scenario_silent_architect")
    main(data_dir, output_path="output.json")
