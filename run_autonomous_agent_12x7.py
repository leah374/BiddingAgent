"""Resumable 12-advertiser x 7-episode AuctionNet evaluation.

Protocol
--------
* Select two advertisers per category from the historical KB: the minimum and
  maximum traffic-weighted mean pValue. Advertiser 0 is excluded because it
  was used while developing the agent.
* Episode 0 acquires memory for each advertiser.
* Episodes 1..6 independently evaluate the agent from the same frozen
  episode-0 memory bank.
* BCQ, OnlineLP, IQL and MOPO are paired baselines on episodes 1..6.

Every run is written to a new attempt directory. A completed marker is atomic,
so restarting the command skips completed units and preserves failed attempts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

# Load this environment's OpenSSL DLLs before Torch or simulator imports can
# cache urllib without HTTPS support.
from autobid_agent.client import JSONClient
import torch

from autobid_agent.auctionnet import AutonomousAuctionNetStrategy, isolated_rng
from autobid_agent.contracts import TOOLS
from autobid_agent.memory import ExperienceStore, fingerprint, write_json
import run_auctionnet_agent_vs_gave_episode as experiment
import run_auctionnet_official_baselines as baselines
from run_autonomous_agent_smoke import asset_manifest, audit_strategy


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "autonomous_12x7_20261116_v6_qwen37plus"
ACQUISITION_EPISODE = 0
EVALUATION_EPISODES = tuple(range(1, 7))
BASELINES = tuple(TOOLS)


class IsolatedOfficialBaseline(baselines.TracedOfficialStrategy):
    """Keep policy inference RNG from consuming the environment RNG stream."""

    def __init__(self, strategy_key: str, seed: int, advertiser: int) -> None:
        super().__init__(strategy_key)
        self.seed = seed
        self.advertiser = advertiser

    def bidding(self, *args):
        with isolated_rng(self.seed, self.advertiser, args[0], self.strategy_key):
            return super().bidding(*args)


def select_advertisers(kb_path: Path) -> list[dict]:
    """Choose historical pValue extremes within each of the six categories."""
    query = """
        SELECT CAST(advertiserNumber AS INT) AS advertiser,
               SUM(avg_pvalue * traffic_num) / SUM(traffic_num) AS avg_pvalue,
               SUM(conversions) / SUM(traffic_num) AS conversion_rate,
               SUM(traffic_num) AS traffic
        FROM kb_market_advertiser
        GROUP BY advertiserNumber
        ORDER BY advertiserNumber
    """
    with sqlite3.connect(kb_path) as connection:
        rows = [dict(advertiser=int(row[0]), avg_pvalue=float(row[1]),
                     conversion_rate=float(row[2]), traffic=int(row[3]))
                for row in connection.execute(query)]
    selected: list[dict] = []
    for category in range(6):
        candidates = [row for row in rows
                      if row["advertiser"] // 8 == category and row["advertiser"] != 0]
        low = min(candidates, key=lambda row: (row["avg_pvalue"], row["advertiser"]))
        high = max(candidates, key=lambda row: (row["avg_pvalue"], -row["advertiser"]))
        if low["advertiser"] == high["advertiser"]:
            raise RuntimeError("Need two distinct advertisers per category")
        for tier, row in (("low", low), ("high", high)):
            selected.append(dict(row, category=category, selection_tier=tier))
    selected.sort(key=lambda row: row["advertiser"])
    if len(selected) != 12 or len({row["advertiser"] for row in selected}) != 12:
        raise RuntimeError("Advertiser selection must contain exactly 12 unique IDs")
    return selected


def next_attempt(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for child in directory.glob("attempt_*"):
        try:
            numbers.append(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    attempt = directory / ("attempt_%03d" % (max(numbers, default=0) + 1))
    attempt.mkdir()
    return attempt


def completed_markers(output: Path) -> dict[str, dict]:
    markers: dict[str, dict] = {}
    for path in output.glob("advertisers/**/completed.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("summary", {}).get("decision_steps") == 48:
                value["marker_path"] = str(path.resolve())
                previous = markers.get(value["run_key"])
                if previous is None or value["completed_at"] > previous["completed_at"]:
                    markers[value["run_key"]] = value
        except (OSError, ValueError, KeyError):
            continue
    return markers


def usage_delta(after: dict, before: dict) -> dict:
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in after}


def aggregate_records(markers: dict[str, dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for marker in markers.values():
        if marker["phase"] == "evaluation":
            groups[marker["strategy"]].append(marker["summary"])
    aggregates = {}
    for strategy, rows in sorted(groups.items()):
        total_budget = sum(float(row["budget"]) for row in rows)
        total_cost = sum(float(row["cost"]) for row in rows)
        aggregates[strategy] = {
            "runs": len(rows),
            "sum_score": sum(float(row["official_score_raw"]) for row in rows),
            "mean_score": sum(float(row["official_score_raw"]) for row in rows) / len(rows),
            "conversions": sum(float(row["conversions"]) for row in rows),
            "cost": total_cost,
            "budget_utilization": total_cost / total_budget if total_budget else 0.0,
        }
    paired = {}
    agent = {(m["advertiser"], m["episode"]): m["summary"] for m in markers.values()
             if m["phase"] == "evaluation" and m["strategy"] == "agent_memory"}
    for baseline in BASELINES:
        base = {(m["advertiser"], m["episode"]): m["summary"] for m in markers.values()
                if m["phase"] == "evaluation" and m["strategy"] == baseline}
        keys = sorted(set(agent) & set(base))
        deltas = [float(agent[key]["official_score_raw"]) -
                  float(base[key]["official_score_raw"]) for key in keys]
        paired[baseline] = {
            "pairs": len(keys),
            "mean_score_delta": sum(deltas) / len(deltas) if deltas else None,
            "wins": sum(delta > 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
        }
    return {"by_strategy": aggregates, "paired_vs_agent": paired}


def update_progress(output: Path, selected: list[dict], started_at: str) -> dict[str, dict]:
    markers = completed_markers(output)
    requested = len(selected) * (1 + len(EVALUATION_EPISODES) * (1 + len(BASELINES)))
    usage = defaultdict(int)
    for marker in markers.values():
        for key, value in marker.get("llm_usage", {}).items():
            usage[key] += int(value)
    progress = {
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed_runs": len(markers),
        "requested_runs": requested,
        "completion_ratio": len(markers) / requested,
        "agent_runs_completed": sum(m["strategy"].startswith("agent") for m in markers.values()),
        "baseline_runs_completed": sum(m["strategy"] in BASELINES for m in markers.values()),
        "llm_usage": dict(usage),
        "aggregates": aggregate_records(markers),
    }
    write_json(output / "progress.json", progress)
    flat = []
    for marker in sorted(markers.values(), key=lambda m: (m["advertiser"], m["episode"], m["strategy"])):
        summary = marker["summary"]
        flat.append({
            "advertiser": marker["advertiser"], "category": marker["category"],
            "selection_tier": marker["selection_tier"], "episode": marker["episode"],
            "phase": marker["phase"], "strategy": marker["strategy"],
            "score": summary["official_score_raw"], "cost": summary["cost"],
            "budget_utilization": summary["budget_utilization"],
            "conversions": summary["conversions"], "cpa": summary["cpa"],
            "expected_conversions": summary["expected_conversions"],
            "pv_signature": summary["pv_signature"], "marker_path": marker["marker_path"],
        })
    experiment.write_csv(output / "all_completed_runs.csv", flat)
    return markers


def record_completion(attempt: Path, run_key: str, phase: str, strategy: str,
                      selected_row: dict, episode: int, summary: dict,
                      llm_usage: dict | None = None, audit: dict | None = None,
                      memory_dir: Path | None = None) -> dict:
    marker = {
        "run_key": run_key,
        "completed_at": datetime.now().astimezone().isoformat(),
        "phase": phase,
        "strategy": strategy,
        "advertiser": selected_row["advertiser"],
        "category": selected_row["category"],
        "selection_tier": selected_row["selection_tier"],
        "episode": episode,
        "summary": summary,
        "llm_usage": llm_usage or {},
        "audit": audit,
        "memory_dir": str(memory_dir.resolve()) if memory_dir else None,
        "attempt_dir": str(attempt.resolve()),
    }
    write_json(attempt / "completed.json", marker)
    return marker


def run_agent(client: JSONClient, kb, output: Path, selected_row: dict,
              episode: int, cycle: int, environment: str,
              seed: int, pv_num: int, source_memory: Path | None) -> dict:
    advertiser = selected_row["advertiser"]
    phase = "acquisition" if episode == ACQUISITION_EPISODE else "evaluation"
    run_key = f"adv{advertiser:02d}:agent:{phase}:ep{episode:02d}"
    if run_key in completed_markers(output):
        return completed_markers(output)[run_key]
    base = output / "advertisers" / f"adv_{advertiser:02d}" / phase / f"episode_{episode:02d}" / "agent_memory"
    attempt = next_attempt(base)
    if source_memory is None:
        memory_dir = attempt / "memory_bank"
    else:
        memory_dir = attempt / "memory_snapshot"
        shutil.copytree(source_memory, memory_dir)
    before = client.usage()
    strategy = AutonomousAuctionNetStrategy(client, ExperienceStore(memory_dir), attempt,
                                            advertiser, cycle, environment, seed,
                                            read_memory=True, kb=kb)
    arm_name = f"agent_{phase}_adv{advertiser:02d}_ep{episode:02d}"
    try:
        summary, _ = experiment.run_arm(arm_name, strategy, attempt, episode, seed,
                                        pv_num, player_index=advertiser)
        strategy.agent.finish(summary)
        audit = audit_strategy(strategy, summary)
        audit["llm_usage"] = usage_delta(client.usage(), before)
        write_json(attempt / "agent_audit.json", audit)
        validity = {
            "at_least_100_successful_llm_requests": audit["llm_usage"]["successful_requests"] >= 100,
            "no_main_decision_fallback": audit["fallback_steps"] == 0,
            "skill_analysis_fallback_at_most_3": audit["analysis_fallback_steps"] <= 3,
            "no_reflection_failure": audit["reflection_failed_cards"] == 0,
            "all_four_experts_every_step": audit["complete_expert_panel_steps"] == 48,
            "all_steps_blend_multiple_experts": audit["multi_expert_blend_steps"] == 48,
        }
        write_json(attempt / "validity.json", validity)
        if not all(validity.values()):
            write_json(attempt / "invalid.json", {
                "reason": "Agent run failed pre-registered validity gates",
                "validity": validity,
                "audit": audit,
            })
            raise RuntimeError("Agent run failed validity gates: " +
                               ", ".join(key for key, passed in validity.items() if not passed))
        return record_completion(attempt, run_key, phase, "agent_memory", selected_row,
                                 episode, summary, audit["llm_usage"], audit, memory_dir)
    except Exception as error:
        write_json(attempt / "failure.json", {
            "error": type(error).__name__ + ": " + str(error),
            "failed_at": datetime.now().astimezone().isoformat(),
            "llm_usage": usage_delta(client.usage(), before),
        })
        raise


def run_baseline(output: Path, selected_row: dict, episode: int, strategy_key: str,
                 seed: int, pv_num: int) -> dict:
    advertiser = selected_row["advertiser"]
    run_key = f"adv{advertiser:02d}:baseline:{strategy_key}:ep{episode:02d}"
    existing = completed_markers(output)
    if run_key in existing:
        return existing[run_key]
    base = (output / "advertisers" / f"adv_{advertiser:02d}" / "evaluation" /
            f"episode_{episode:02d}" / "baselines" / strategy_key)
    attempt = next_attempt(base)
    strategy = IsolatedOfficialBaseline(strategy_key, seed, advertiser)
    arm_name = f"{strategy_key}_adv{advertiser:02d}_ep{episode:02d}"
    try:
        summary, _ = experiment.run_arm(arm_name, strategy, attempt, episode, seed,
                                        pv_num, player_index=advertiser)
        summary["baseline"] = strategy_key
        write_json(attempt / f"{arm_name}_summary.json", summary)
        return record_completion(attempt, run_key, "evaluation", strategy_key,
                                 selected_row, episode, summary)
    except Exception as error:
        write_json(attempt / "failure.json", {
            "error": type(error).__name__ + ": " + str(error),
            "failed_at": datetime.now().astimezone().isoformat(),
        })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pv-num", type=int, default=500_000)
    parser.add_argument("--llm-model", default="qwen3.7-plus-2026-05-26")
    parser.add_argument("--max-requests", type=int, default=20_000)
    parser.add_argument("--proxy", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    selected = select_advertisers(experiment.KB_PATH)
    started_at = datetime.now().astimezone().isoformat()
    manifest_path = output / "manifest.json"
    assets = asset_manifest()
    source_files = (sorted((ROOT / "autobid_agent").rglob("*.py")) +
                    sorted((ROOT / "autobid_agent").rglob("SKILL.md")) +
                    [Path(__file__), ROOT / "run_auctionnet_agent_vs_gave_episode.py",
                     ROOT / "run_auctionnet_official_baselines.py"])
    source_hashes = {str(path.relative_to(ROOT)): experiment.sha256_file(path)
                     for path in source_files}
    environment_spec = {
        "official_commit": experiment.official_commit(), "assets": assets,
        "pv_num": args.pv_num, "pv_generator": "neuripsPvGen",
        "pvalue_mean_base": 0.0005, "score_beta": 2, "steps": 48,
        "tools": list(TOOLS), "onlinelp_tool_table": "episode-0.csv",
        "kb_sha256": experiment.sha256_file(experiment.KB_PATH),
        "source_hashes": source_hashes, "model": args.llm_model,
    }
    environment = fingerprint(environment_spec)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["environment_fingerprint"] != environment:
            raise RuntimeError("Environment/source fingerprint changed; use a new output directory")
        started_at = manifest["started_at"]
    else:
        manifest = {
            **environment_spec,
            "environment_fingerprint": environment,
            "started_at": started_at,
            "seed": args.seed,
            "python": sys.version,
            "torch": torch.__version__,
            "selected_advertisers": selected,
            "selection_rule": "Within each category, min/max historical traffic-weighted avg_pvalue; exclude advertiser 0",
            "episode_protocol": {
                "acquisition": ACQUISITION_EPISODE,
                "frozen_memory_evaluation": list(EVALUATION_EPISODES),
            },
            "baselines": list(BASELINES),
            "requested_runs": len(selected) * (1 + len(EVALUATION_EPISODES) * (1 + len(BASELINES))),
            "guardrails": {"scale": [0.8, 1.2], "segments": [0.95, 1.05]},
            "simulator": "48-advertiser official market; selected player slot replaced",
        }
        write_json(manifest_path, manifest)
    client = JSONClient(args.llm_model, max_requests=args.max_requests, proxy=args.proxy)
    preflight_before = client.usage()
    preflight = client.ask_json(
        '{"phase":"preflight","instruction":"Return exactly {\\"ok\\":true}","context":{}}'
    )
    if preflight != {"ok": True}:
        raise RuntimeError("LLM preflight returned an unexpected response")
    write_json(output / "llm_preflight.json", {
        "checked_at": datetime.now().astimezone().isoformat(),
        "model": args.llm_model,
        "response": preflight,
        "usage": usage_delta(client.usage(), preflight_before),
    })
    kb = experiment.build_kb_tool()
    update_progress(output, selected, started_at)

    try:
        for selected_row in selected:
            advertiser = selected_row["advertiser"]
            acquisition_key = f"adv{advertiser:02d}:agent:acquisition:ep00"
            markers = completed_markers(output)
            if acquisition_key not in markers:
                run_agent(client, kb, output, selected_row, ACQUISITION_EPISODE, 0,
                          environment, args.seed, args.pv_num, None)
                markers = update_progress(output, selected, started_at)
            acquisition = markers[acquisition_key]
            acquisition_memory = Path(acquisition["memory_dir"])
            if not (acquisition_memory / "principles.json").exists():
                raise RuntimeError(f"Completed acquisition lacks principles: {acquisition_memory}")
            for episode in EVALUATION_EPISODES:
                run_agent(client, kb, output, selected_row, episode, 1, environment,
                          args.seed, args.pv_num, acquisition_memory)
                update_progress(output, selected, started_at)
                for strategy_key in BASELINES:
                    run_baseline(output, selected_row, episode, strategy_key,
                                 args.seed, args.pv_num)
                    update_progress(output, selected, started_at)
        markers = update_progress(output, selected, started_at)
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        progress["status"] = "complete"
        progress["completed_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "summary.json", progress)
        print(json.dumps({"status": "complete", "runs": len(markers),
                          "output": str(output)}, ensure_ascii=False), flush=True)
    except Exception as error:
        write_json(output / "last_failure.json", {
            "error": type(error).__name__ + ": " + str(error),
            "failed_at": datetime.now().astimezone().isoformat(),
            "resume_command": f"{sys.executable} {Path(__file__).name} --output-dir {output}",
        })
        update_progress(output, selected, started_at)
        raise


if __name__ == "__main__":
    main()
