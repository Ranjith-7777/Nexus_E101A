import os
import pandas as pd
from datetime import datetime, timedelta
import random

BASE = os.path.join(os.path.dirname(__file__), "demo")

def dt(s: str) -> str:
    return s  # keep as ISO-like strings for simplicity

def ensure_dir():
    os.makedirs(BASE, exist_ok=True)

def write_all(prefix: str, users, tasks, commits, messages, handoffs, evidence):
    outdir = os.path.join(BASE, prefix)
    os.makedirs(outdir, exist_ok=True)

    pd.DataFrame(users).to_csv(os.path.join(outdir, "users.csv"), index=False)
    pd.DataFrame(tasks).to_csv(os.path.join(outdir, "tasks.csv"), index=False)
    pd.DataFrame(commits).to_csv(os.path.join(outdir, "commits.csv"), index=False)
    pd.DataFrame(messages).to_csv(os.path.join(outdir, "messages.csv"), index=False)
    pd.DataFrame(handoffs).to_csv(os.path.join(outdir, "handoffs.csv"), index=False)
    pd.DataFrame(evidence).to_csv(os.path.join(outdir, "evidence.csv"), index=False)

def scenario_silent_architect():
    # Team: A (silent architect), B (driver), C (guardian)
    now = datetime.now()
    team_id = "team1"
    users = [
        {"user_id": "uA", "name": "A (Silent)", "join_date": (now - timedelta(days=200)).date().isoformat(), "team_id": team_id},
        {"user_id": "uB", "name": "B (Driver)", "join_date": (now - timedelta(days=120)).date().isoformat(), "team_id": team_id},
        {"user_id": "uC", "name": "C (Guardian)", "join_date": (now - timedelta(days=300)).date().isoformat(), "team_id": team_id},
    ]

    # Tasks with dependencies: T1 -> T2 -> T3; A resolves blockers on T1/T2 but low visibility
    t0 = now - timedelta(days=10)
    tasks = [
        {"task_id": "T1", "team_id": team_id, "created_at": dt((t0).isoformat()), "started_at": dt((t0+timedelta(hours=2)).isoformat()),
         "completed_at": dt((t0+timedelta(days=2)).isoformat()), "status": "done", "assignee_id": "uA",
         "priority": 4, "blocked_by": "", "depends_on": ""},
        {"task_id": "T2", "team_id": team_id, "created_at": dt((t0+timedelta(hours=3)).isoformat()), "started_at": dt((t0+timedelta(days=2, hours=2)).isoformat()),
         "completed_at": dt((t0+timedelta(days=5)).isoformat()), "status": "done", "assignee_id": "uB",
         "priority": 3, "blocked_by": "T1", "depends_on": "T1"},
        {"task_id": "T3", "team_id": team_id, "created_at": dt((t0+timedelta(days=1)).isoformat()), "started_at": dt((t0+timedelta(days=5, hours=2)).isoformat()),
         "completed_at": dt((t0+timedelta(days=7)).isoformat()), "status": "done", "assignee_id": "uC",
         "priority": 2, "blocked_by": "T2", "depends_on": "T2"},
    ]

    commits = [
        {"commit_id": "c1", "user_id": "uA", "timestamp": dt((t0+timedelta(days=1)).isoformat()), "lines_added": 120, "lines_deleted": 10, "files_changed": 5, "task_id": "T1"},
        {"commit_id": "c2", "user_id": "uB", "timestamp": dt((t0+timedelta(days=3)).isoformat()), "lines_added": 200, "lines_deleted": 50, "files_changed": 8, "task_id": "T2"},
        {"commit_id": "c3", "user_id": "uC", "timestamp": dt((t0+timedelta(days=6)).isoformat()), "lines_added": 30, "lines_deleted": 5, "files_changed": 2, "task_id": "T3"},
    ]

    # A is quiet: few messages
    messages = []
    mid = 0
    for u, count in [("uA", 3), ("uB", 20), ("uC", 8)]:
        for i in range(count):
            mid += 1
            text = "ok" if (u == "uB" and i % 6 == 0) else "update on task" if u != "uA" else "unblocked dependency"
            messages.append({"msg_id": f"m{mid}", "user_id": u, "timestamp": dt((t0+timedelta(hours=4+i)).isoformat()), "channel_id": "ch1", "text": text})

    handoffs = [
        {"event_id": "h1", "task_id": "T2", "from_user_id": "uA", "to_user_id": "uB", "timestamp": dt((t0+timedelta(days=2)).isoformat()), "reason": "handoff"},
    ]

    evidence = [
        {"claim_id": "e1", "user_id": "uA", "timestamp": dt((t0+timedelta(days=4)).isoformat()), "type": "mentoring",
         "description": "Helped unblock design discussion", "peer_confirmations": "uB,uC", "manager_confirmed": False}
    ]

    write_all("scenario_silent_architect", users, tasks, commits, messages, handoffs, evidence)

def scenario_message_spammer():
    now = datetime.now()
    team_id = "team2"
    users = [
        {"user_id": "u1", "name": "Worker", "join_date": (now - timedelta(days=150)).date().isoformat(), "team_id": team_id},
        {"user_id": "u2", "name": "Spammer", "join_date": (now - timedelta(days=150)).date().isoformat(), "team_id": team_id},
        {"user_id": "u3", "name": "Reviewer", "join_date": (now - timedelta(days=150)).date().isoformat(), "team_id": team_id},
    ]
    t0 = now - timedelta(days=10)

    tasks = [
        {"task_id": "A1", "team_id": team_id, "created_at": dt(t0.isoformat()), "started_at": dt((t0+timedelta(hours=2)).isoformat()),
         "completed_at": dt((t0+timedelta(days=3)).isoformat()), "status": "done", "assignee_id": "u1",
         "priority": 3, "blocked_by": "", "depends_on": ""},
        {"task_id": "A2", "team_id": team_id, "created_at": dt((t0+timedelta(hours=1)).isoformat()), "started_at": dt((t0+timedelta(days=1)).isoformat()),
         "completed_at": dt((t0+timedelta(days=4)).isoformat()), "status": "done", "assignee_id": "u2",
         "priority": 2, "blocked_by": "", "depends_on": ""},
    ]

    commits = [
        {"commit_id": "k1", "user_id": "u1", "timestamp": dt((t0+timedelta(days=2)).isoformat()), "lines_added": 180, "lines_deleted": 20, "files_changed": 6, "task_id": "A1"},
        # spammer splits commits: many tiny commits with tiny diff
    ]
    for i in range(25):
        commits.append({"commit_id": f"kS{i}", "user_id": "u2", "timestamp": dt((t0+timedelta(days=2, hours=i%6)).isoformat()),
                        "lines_added": 2, "lines_deleted": 1, "files_changed": 1, "task_id": "A2"})

    messages = []
    mid = 0
    # spammer floods short messages
    for i in range(120):
        mid += 1
        messages.append({"msg_id": f"ms{mid}", "user_id": "u2", "timestamp": dt((t0+timedelta(hours=i)).isoformat()), "channel_id": "ch2", "text": "ok"})
    # others normal
    for u, count in [("u1", 12), ("u3", 10)]:
        for i in range(count):
            mid += 1
            messages.append({"msg_id": f"ms{mid}", "user_id": u, "timestamp": dt((t0+timedelta(hours=2+i)).isoformat()),
                             "channel_id": "ch2", "text": "review/update details"})

    handoffs = []
    evidence = []
    write_all("scenario_message_spammer", users, tasks, commits, messages, handoffs, evidence)

def scenario_new_joiner():
    now = datetime.now()
    team_id = "team3"
    users = [
        {"user_id": "uX", "name": "Senior", "join_date": (now - timedelta(days=250)).date().isoformat(), "team_id": team_id},
        {"user_id": "uY", "name": "New Joiner", "join_date": (now - timedelta(days=7)).date().isoformat(), "team_id": team_id},
        {"user_id": "uZ", "name": "Peer", "join_date": (now - timedelta(days=180)).date().isoformat(), "team_id": team_id},
    ]
    t0 = now - timedelta(days=10)
    tasks = [
        {"task_id": "N1", "team_id": team_id, "created_at": dt(t0.isoformat()), "started_at": dt((t0+timedelta(hours=2)).isoformat()),
         "completed_at": dt((t0+timedelta(days=3)).isoformat()), "status": "done", "assignee_id": "uX",
         "priority": 4, "blocked_by": "", "depends_on": ""},
        {"task_id": "N2", "team_id": team_id, "created_at": dt((t0+timedelta(days=2)).isoformat()), "started_at": dt((t0+timedelta(days=4)).isoformat()),
         "completed_at": dt((t0+timedelta(days=6)).isoformat()), "status": "done", "assignee_id": "uY",
         "priority": 2, "blocked_by": "", "depends_on": ""},
    ]
    commits = [
        {"commit_id": "n1", "user_id": "uX", "timestamp": dt((t0+timedelta(days=1)).isoformat()), "lines_added": 220, "lines_deleted": 60, "files_changed": 9, "task_id": "N1"},
        {"commit_id": "n2", "user_id": "uY", "timestamp": dt((t0+timedelta(days=5)).isoformat()), "lines_added": 40, "lines_deleted": 5, "files_changed": 2, "task_id": "N2"},
    ]
    messages = []
    mid = 0
    for u, count in [("uX", 15), ("uY", 6), ("uZ", 12)]:
        for i in range(count):
            mid += 1
            messages.append({"msg_id": f"mn{mid}", "user_id": u, "timestamp": dt((t0+timedelta(hours=3+i)).isoformat()),
                             "channel_id": "ch3", "text": "update details"})
    handoffs = []
    evidence = [
        {"claim_id": "en1", "user_id": "uY", "timestamp": dt((t0+timedelta(days=6)).isoformat()), "type": "mentoring",
         "description": "Pair programming help", "peer_confirmations": "uX,uZ", "manager_confirmed": False}
    ]
    write_all("scenario_new_joiner", users, tasks, commits, messages, handoffs, evidence)

if __name__ == "__main__":
    ensure_dir()
    scenario_silent_architect()
    scenario_message_spammer()
    scenario_new_joiner()
    print(f"Generated demo datasets in: {BASE}")
