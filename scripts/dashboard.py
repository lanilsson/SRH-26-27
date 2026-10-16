"""Build the read-only results dashboard: results/dashboard.html.

Reads results/runs/*.json (written by src/train.py) and configs/*.yaml and
writes one self-contained HTML file (no server, no network) showing:
  - which of the planned runs (every config x 5 seeds) exist;
  - provenance checks on each run (config drift, git commit, dirty tree);
  - the three dependent variables per condition (seed values, mean +/- SEM);
  - per-epoch training curves, and resource use.

The dashboard only displays what the run records contain. It never starts
runs and never writes anything except the HTML file. Figures for the paper
or poster should still come from a committed script in results/figures/.

Usage:
    python scripts/dashboard.py            # -> results/dashboard.html
    python scripts/dashboard.py --open     # ...and open it in the browser
    python scripts/dashboard.py --runs-dir sandbox/07_mini_experiments/outputs \\
        --out sandbox/07_mini_experiments/outputs/dashboard.html
"""

import argparse
import datetime
import json
import statistics
import subprocess
import webbrowser
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results/runs"
CONFIGS_DIR = ROOT / "configs"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
OUT = ROOT / "results/dashboard.html"

SEEDS = [0, 1, 2, 3, 4]  # same seeds as scripts/run_all.sh
METRICS = ["pr_auc", "f1", "roc_auc"]
CONDITIONS = ["baseline", "ego", "second_order", "combined"]
# The two flags that define each condition (README: Experimental conditions).
CONDITION_FLAGS = {
    "baseline": (False, False),
    "ego": (True, False),
    "second_order": (False, True),
    "combined": (True, True),
}
# The only config keys allowed to differ between conditions of one dataset.
INDEPENDENT_KEYS = {"condition", "model.ego_separation", "model.second_order"}


def flatten(d, prefix=""):
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def config_diff(a, b):
    """Dotted keys whose values differ between two configs, as 'key: a -> b'."""
    fa, fb = flatten(a), flatten(b)
    return [f"{k}: {fa.get(k, '(absent)')} -> {fb.get(k, '(absent)')}"
            for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)]


def head_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip()
    except Exception:
        return None


def load_configs():
    configs = {}
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text())
        configs[(cfg["dataset"], cfg["condition"])] = {"file": path.name, "config": cfg}
    return configs


def check_configs(configs):
    """Study-level checks on the config files as they are now."""
    flags = []
    for (dataset, condition), entry in configs.items():
        model = entry["config"].get("model", {})
        expected = CONDITION_FLAGS.get(condition)
        actual = (model.get("ego_separation"), model.get("second_order"))
        if expected is not None and actual != expected:
            flags.append(f"{entry['file']}: condition '{condition}' should have "
                         f"(ego_separation, second_order) = {expected}, found {actual}")
    for dataset in sorted({d for d, _ in configs}):
        entries = [e for (d, _), e in configs.items() if d == dataset]
        reference = entries[0]
        for other in entries[1:]:
            extra = [line for line in config_diff(reference["config"], other["config"])
                     if line.split(":")[0] not in INDEPENDENT_KEYS]
            if extra:
                flags.append(f"{reference['file']} vs {other['file']} differ beyond the "
                             f"two condition flags: {'; '.join(extra)}")
    return flags


def load_runs(runs_dir):
    runs, unreadable = [], []
    for path in sorted(runs_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            runs.append({
                "file": path.name,
                "run": record["run"],
                "dataset": record["dataset"],
                "condition": record["condition"],
                "seed": record["seed"],
                "test": record["metrics"],
                "val": record["val_metrics"],
                "best_epoch": record.get("best_epoch"),
                "epochs_run": record.get("epochs_run"),
                "num_params": record.get("num_params"),
                "pos_weight": record.get("pos_weight"),
                "resources": record.get("resources", {}),
                "config": record.get("config", {}),
                "history": record.get("history"),
                "git_commit": record.get("git_commit"),
                "git_dirty": record.get("git_dirty"),
                "timestamp": record.get("timestamp"),
                "flags": [],
                "notes": [],
            })
        except (json.JSONDecodeError, KeyError, TypeError) as err:
            unreadable.append(f"{path.name}: {type(err).__name__}: {err}")
    return runs, unreadable


def check_runs(runs, configs):
    """Attach provenance flags (things that could make runs not comparable)
    and notes (worth knowing, not a problem) to each run."""
    majority = {}
    for dataset in {r["dataset"] for r in runs}:
        commits = Counter(r["git_commit"] for r in runs if r["dataset"] == dataset)
        majority[dataset] = commits.most_common(1)[0][0]

    for run in runs:
        flags, notes = run["flags"], run["notes"]
        expected_name = f"{run['dataset']}_{run['condition']}_seed{run['seed']}"
        if run["run"] != expected_name or run["file"] != f"{expected_name}.json":
            flags.append(f"File or run name does not match its contents (expected {expected_name}).")

        current = configs.get((run["dataset"], run["condition"]))
        if current is None:
            flags.append("No config file in configs/ for this dataset and condition.")
        else:
            diff = config_diff(current["config"], run["config"])
            if diff:
                flags.append(f"Config used differs from {current['file']} as it is now "
                             f"(file -> run): {'; '.join(diff)}")

        commit = run["git_commit"]
        if not commit or commit == "unknown":
            flags.append("Git commit was not recorded.")
        elif commit != majority[run["dataset"]]:
            flags.append(f"Git commit {commit[:7]} differs from most {run['dataset']} "
                         f"runs ({str(majority[run['dataset']])[:7]}).")
        if run["git_dirty"]:
            flags.append("src/ or configs/ had uncommitted changes during the run, so the "
                         "recorded commit does not fully describe the code.")
        elif run["git_dirty"] is None:
            notes.append("Whether the working tree was clean was not recorded.")

        max_epochs = run["config"].get("training", {}).get("epochs")
        if max_epochs is not None and run["epochs_run"] == max_epochs:
            notes.append(f"Ran all {max_epochs} epochs without early stopping.")
        if run["history"] is None:
            notes.append("No per-epoch history in this record.")


def summarize(runs):
    """mean and SEM (sample SD / sqrt(n), as pandas .sem()) per split,
    dataset, condition and metric — the same numbers scripts/aggregate.py
    writes for the test split."""
    summary = {"val": {}, "test": {}}
    for split in summary:
        groups = {}
        for run in runs:
            groups.setdefault((run["dataset"], run["condition"]), []).append(run)
        for (dataset, condition), group in groups.items():
            cell = {"n": len(group)}
            for metric in METRICS:
                values = [r[split][metric] for r in group]
                cell[metric] = {
                    "mean": statistics.fmean(values),
                    "sem": statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else None,
                }
            summary[split].setdefault(dataset, {})[condition] = cell
    return summary


def ordered(found, canonical):
    return [c for c in canonical if c in found] + sorted(set(found) - set(canonical))


def build_payload(runs_dir):
    configs = load_configs()
    runs, unreadable = load_runs(runs_dir)
    check_runs(runs, configs)

    planned = {(d, c, s) for (d, c) in configs for s in SEEDS}
    done = {(r["dataset"], r["condition"], r["seed"]) for r in runs}
    datasets = sorted({d for d, _, _ in planned | done})
    conditions = ordered({c for _, c, _ in planned | done}, CONDITIONS)
    seeds = sorted({s for _, _, s in planned | done})

    study_flags = check_configs(configs)
    study_flags += [f"Unreadable run file — {line}" for line in unreadable]
    for dataset in datasets:
        commits = sorted({str(r["git_commit"])[:7] for r in runs if r["dataset"] == dataset})
        if len(commits) > 1:
            study_flags.append(f"{dataset} runs come from {len(commits)} different git "
                               f"commits ({', '.join(commits)}).")
    unplanned = sorted(f"{d}_{c}_seed{s}" for d, c, s in done - planned)
    if unplanned:
        study_flags.append(f"Runs that are not in the plan (configs x seeds "
                           f"{SEEDS[0]}-{SEEDS[-1]}): {', '.join(unplanned)}")

    try:
        shown_dir = str(runs_dir.resolve().relative_to(ROOT))
    except ValueError:
        shown_dir = str(runs_dir)
    return {
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "runs_dir": shown_dir,
        "head_commit": head_commit(),
        "datasets": datasets,
        "conditions": conditions,
        "seeds": seeds,
        "metrics": METRICS,
        "planned": [list(p) for p in sorted(planned)],
        "runs": runs,
        "summary": summarize(runs),
        "study_flags": study_flags,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--open", action="store_true", help="open the dashboard in the browser")
    args = parser.parse_args()

    payload = build_payload(args.runs_dir)
    # "</" must not appear inside the inline <script> block.
    data = json.dumps(payload).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)

    flagged = sum(bool(r["flags"]) for r in payload["runs"])
    print(f"{len(payload['runs'])} runs found, {len(payload['planned'])} planned, "
          f"{flagged} flagged, {len(payload['study_flags'])} study-level warnings -> {args.out}")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()
