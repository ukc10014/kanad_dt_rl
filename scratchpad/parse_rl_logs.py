#!/usr/bin/env python3
"""Parse RL self-play / RLOO training logs into tidy per-step CSVs.

Recovers the per-step dynamics (reward, K, invalid, gen_len, p_model on train rows;
mean_K, K@p, slope on eval rows) that were swept up by the blanket *.log gitignore rule.

Outputs (under results/logs/):
  rl_dynamics_steps.csv  -- one row per (run, step), train+eval fields merged
  rl_dynamics_kbyp.csv   -- long: one row per (run, step, p_position) K@p value
  rl_dynamics_runs.csv   -- one row per run: parsed header metadata
"""
import csv
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
OUT = REPO / "results" / "logs"
OUT.mkdir(parents=True, exist_ok=True)

# Per-run logs with real dynamics. Aggregate concatenations (all.log, overnight*.log,
# orchestrator.log, klctl_batch.log) are excluded -- their constituent runs each have
# their own log already, and mixing headers would corrupt run identity.
EXCLUDE = {"all.log", "overnight.log", "overnight2.log", "orchestrator.log",
           "klctl_batch.log", "watch.log", "scratch_dl.log"}

TRAIN_RE = re.compile(
    r"step=(\d+)\s+train\s+reward=([\-\d.]+)\s+K=([\-\d.]+)\s+invalid=([\-\d.]+)"
    r"\s+gen_len=([\-\d.]+)(?:\s+p_model=([\-\d.]+))?")
EVAL_RE = re.compile(
    r"step=(\d+)\s+eval\s+mean_K=([\-\d.]+)\s+(.*?)slope\(hi-lo\)=([+\-][\d.]+)")
KP_RE = re.compile(r"K@p=([\d.]+)->([\-\d.]+)")
ARM_RE = re.compile(r"\[(evidential_modelpred|evidential)\]\s+model=(\S+)")
HEADER_RE = re.compile(r"^\[selfplay[^\]]*\]|^\[evidential")


def fnum(x):
    if x is None or x == "":
        return ""
    return x


def parse_header(lines):
    meta = {"model": "", "arm": "", "dataset": "", "train_p": "", "pstar": "",
            "kl": "", "kl_ref": "", "seed_adapter": "", "lag": "", "cot": "",
            "snapshot_every": "", "tag": "", "mode_line": ""}
    for ln in lines:
        m = ARM_RE.search(ln)
        if m:
            meta["arm"] = m.group(1)
            meta["model"] = m.group(2)
        for key, pat in [
            ("dataset", r"dataset=(\S+)"),
            ("kl", r"\bkl=([\d.]+)"),
            ("kl_ref", r"kl_ref=(\S+)"),
            ("seed_adapter", r"seed_adapter=(\S+)"),
            ("lag", r"\blag=(\d+)"),
            ("cot", r"cot=(True|False)"),
            ("snapshot_every", r"snapshot_every=(\d+)"),
            ("tag", r"tag=(\S+)"),
            ("pstar", r"p\*=([\d.]+)"),
        ]:
            mm = re.search(pat, ln)
            if mm and not meta[key]:
                meta[key] = mm.group(1)
        mm = re.search(r"train_p=(\[[^\]]*\])", ln)
        if mm and not meta["train_p"]:
            meta["train_p"] = mm.group(1).replace(",", ";")
        mm = re.search(r"grid=(\([^)]*\))", ln)
        if mm and not meta["train_p"]:
            meta["train_p"] = mm.group(1).replace(",", ";")
        if ln.startswith("[selfplay") and not meta["mode_line"]:
            meta["mode_line"] = ln.strip()[:120]
    return meta


def parse_log(path):
    text = path.read_text(errors="replace").splitlines()
    meta = parse_header(text)
    adapter = ""
    steps = {}   # step -> merged dict
    kbyp = []    # (step, position, p, k)
    seen_eval = set()
    for ln in text:
        ms = re.search(r"saved adapter -> .*/adapters/(\S+)", ln)
        if ms:
            adapter = ms.group(1)
        mt = TRAIN_RE.search(ln)
        if mt:
            step = int(mt.group(1))
            row = steps.setdefault(step, {"step": step})
            row.update({
                "reward": fnum(mt.group(2)), "K_train": fnum(mt.group(3)),
                "invalid": fnum(mt.group(4)), "gen_len": fnum(mt.group(5)),
                "p_model": fnum(mt.group(6)),
            })
            continue
        me = EVAL_RE.search(ln)
        if me:
            step = int(me.group(1))
            kps = KP_RE.findall(me.group(3))
            # the final eval line is printed twice; dedup identical (step, mean_K, body)
            sig = (step, me.group(2), me.group(3).strip(), me.group(4))
            if sig in seen_eval:
                continue
            seen_eval.add(sig)
            row = steps.setdefault(step, {"step": step})
            row["mean_K"] = fnum(me.group(2))
            row["slope"] = fnum(me.group(4))
            if kps:
                row["k_at_p_lo"] = kps[0][1]
                row["k_at_p_hi"] = kps[-1][1]
                row["p_lo"] = kps[0][0]
                row["p_hi"] = kps[-1][0]
            for i, (p, k) in enumerate(kps):
                kbyp.append({"step": step, "pos": i, "p": p, "k_rate": k})
    return meta, adapter, steps, kbyp


def main():
    logs = []
    for p in sorted((REPO / "results").rglob("*.log")):
        if p.name in EXCLUDE:
            continue
        txt = p.read_text(errors="replace")
        if re.search(r"step=\d+\s+(train|eval)", txt):
            logs.append(p)

    step_rows, kbyp_rows, run_rows = [], [], []
    for p in logs:
        run = p.relative_to(REPO).as_posix()
        meta, adapter, steps, kbyp = parse_log(p)
        run_id = adapter or p.stem
        n_steps = len(steps)
        max_step = max(steps) if steps else ""
        run_rows.append({
            "run_id": run_id, "log": run, "adapter": adapter, **meta,
            "n_eval_points": n_steps, "max_step": max_step,
        })
        for step in sorted(steps):
            r = steps[step]
            step_rows.append({
                "run_id": run_id, "log": run, "arm": meta["arm"],
                "model": meta["model"], **r,
            })
        for k in kbyp:
            kbyp_rows.append({"run_id": run_id, "log": run, **k})

    step_cols = ["run_id", "log", "arm", "model", "step", "reward", "K_train",
                 "invalid", "gen_len", "p_model", "mean_K", "slope",
                 "k_at_p_lo", "k_at_p_hi", "p_lo", "p_hi"]
    with (OUT / "rl_dynamics_steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=step_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(step_rows)

    with (OUT / "rl_dynamics_kbyp.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "log", "step", "pos", "p", "k_rate"])
        w.writeheader()
        w.writerows(kbyp_rows)

    run_cols = ["run_id", "log", "adapter", "arm", "model", "dataset", "train_p",
                "pstar", "kl", "kl_ref", "seed_adapter", "lag", "cot",
                "snapshot_every", "tag", "n_eval_points", "max_step", "mode_line"]
    with (OUT / "rl_dynamics_runs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=run_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(run_rows)

    print(f"logs parsed: {len(logs)}")
    print(f"step rows:   {len(step_rows)}")
    print(f"kbyp rows:   {len(kbyp_rows)}")
    print(f"runs:        {len(run_rows)}")
    print("wrote:", OUT / "rl_dynamics_steps.csv")
    print("wrote:", OUT / "rl_dynamics_kbyp.csv")
    print("wrote:", OUT / "rl_dynamics_runs.csv")


if __name__ == "__main__":
    main()
