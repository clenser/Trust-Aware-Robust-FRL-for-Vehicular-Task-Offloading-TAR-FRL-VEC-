"""Keyword screen of a folder of PDFs against the project's literature axes.

For each PDF the first 40 pages are converted with ``pdftotext`` and matches of
seven term groups are counted.  A group is "present" at >= 5 matches, which
filters incidental mentions in reference lists.  This is a triage aid for the
literature review (``docs/LITERATURE_REVIEW.md``), not a substitute for
reading: counts say what a paper talks about, not whether it does it well.

    python data/source/screen_papers.py D:/Papers docs/literature_screen.csv
"""

import csv
import os
import re
import subprocess
import sys

GROUPS = {
    "vehicular": r"vehic|VANET|\bIoV\b|V2X|V2V",
    "offloading": r"offload",
    "rl": r"reinforcement|\bDQN\b|Q-learning",
    "fl": r"federated",
    "trust": r"trust|reputation",
    "poisoning": r"poison|byzantine|malicious|backdoor|sign.?flip",
    "edge": r"edge comput|\bMEC\b|\bRSU\b",
}
PRESENT = 5


def text_of(pdf):
    try:
        out = subprocess.run(["pdftotext", "-l", "40", pdf, "-"], capture_output=True, timeout=90)
        return out.stdout.decode("utf-8", "replace")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "D:/Papers"
    dst = sys.argv[2] if len(sys.argv) > 2 else "docs/literature_screen.csv"
    rows = []
    for name in sorted(os.listdir(src)):
        if not name.lower().endswith(".pdf"):
            continue
        t = text_of(os.path.join(src, name))
        counts = {g: len(re.findall(p, t, re.I)) for g, p in GROUPS.items()}
        flags = [g for g, c in counts.items() if c >= PRESENT]
        rows.append({"file": name, "chars": len(t), **counts,
                     "core_hit": int(all(g in flags for g in ("fl", "offloading"))),
                     "fl_rl_offload_attack": int(all(g in flags for g in ("fl", "rl", "offloading", "poisoning")))})
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("papers:", len(rows), " unreadable (no text):", sum(r["chars"] < 500 for r in rows))
    for label, cond in (("FL + vehicular", ("fl", "vehicular")), ("FL + offloading", ("fl", "offloading")),
                        ("FL + RL", ("fl", "rl")), ("FL + RL + offloading", ("fl", "rl", "offloading")),
                        ("FL + poisoning/Byzantine", ("fl", "poisoning")),
                        ("FL + trust", ("fl", "trust")),
                        ("FL + vehicular + poisoning", ("fl", "vehicular", "poisoning")),
                        ("FL + offloading + poisoning", ("fl", "offloading", "poisoning"))):
        hit = [r["file"] for r in rows if all(r[g] >= PRESENT for g in cond)]
        print("%-30s %2d" % (label, len(hit)))
    print("fl+rl+offloading+poisoning:", [r["file"] for r in rows if r["fl_rl_offload_attack"]])


if __name__ == "__main__":
    main()
