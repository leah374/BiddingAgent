"""48 advertisers, one cold-start episode and one frozen-memory test episode.

Each advertiser runs agent, BCQ, OnlineLP, IQL and MOPO on both episodes.
Completed runs in the chosen output directory are skipped on resume.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

from autobid_agent.client import JSONClient
from autobid_agent.memory import fingerprint, write_json
from experiments import run_autonomous_agent_12x7 as prior


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "autonomous_48x2"
EPISODES = (0, 1)
MODEL = "qwen3.7-plus-2026-05-26"


def selected_advertisers() -> list[dict]:
    return [dict(advertiser=index, category=index // 8, selection_tier="all_48")
            for index in range(48)]


def source_manifest() -> dict:
    source_files = (sorted((ROOT / "autobid_agent").rglob("*.py")) +
                    sorted((ROOT / "autobid_agent").rglob("SKILL.md")) +
                    [ROOT / "launch.py",
                     ROOT / "experiments" / "run_autonomous_agent_12x7.py",
                     ROOT / "experiments" / "run_autonomous_agent_smoke.py",
                     ROOT / "compat" / "run_auctionnet_agent_vs_gave_episode.py",
                     ROOT / "compat" / "run_auctionnet_official_baselines.py",
                     ROOT / "compat" / "bidding_agent_demo_v2.py"])
    source_hashes = {str(path.relative_to(ROOT)): prior.experiment.sha256_file(path)
                     for path in source_files}
    environment_spec = {
        "official_commit": prior.experiment.official_commit(),
        "assets": prior.asset_manifest(),
        "pv_num": 500_000, "pv_generator": "neuripsPvGen",
        "pvalue_mean_base": 0.0005, "score_beta": 2, "steps": 48,
        "tools": list(prior.BASELINES), "onlinelp_tool_table": "episode-0.csv",
        "kb_sha256": prior.experiment.sha256_file(prior.experiment.KB_PATH),
        "source_hashes": source_hashes, "model": MODEL,
    }
    return {**environment_spec, "seed": 2026,
            "environment_fingerprint": fingerprint(environment_spec)}


def progress(output: Path, started_at: str) -> dict:
    markers = prior.completed_markers(output)
    usage = defaultdict(int)
    for marker in markers.values():
        for key, value in marker.get("llm_usage", {}).items():
            usage[key] += int(value)
    by_episode_strategy = defaultdict(list)
    rows = []
    for marker in sorted(markers.values(), key=lambda m: (m["advertiser"], m["episode"], m["strategy"])):
        summary = marker["summary"]
        by_episode_strategy[(marker["episode"], marker["strategy"])].append(summary)
        rows.append({
            "advertiser": marker["advertiser"], "category": marker["category"],
            "episode": marker["episode"], "phase": marker["phase"],
            "strategy": marker["strategy"], "score": summary["official_score_raw"],
            "cost": summary["cost"], "budget_utilization": summary["budget_utilization"],
            "conversions": summary["conversions"], "cpa": summary["cpa"],
            "pv_signature": summary["pv_signature"], "marker_path": marker["marker_path"],
            "reused_from": marker.get("reused_from", ""),
        })
    aggregates = {}
    for (episode, strategy), group in sorted(by_episode_strategy.items()):
        aggregates[f"episode_{episode}:{strategy}"] = {
            "runs": len(group),
            "mean_score": sum(x["official_score_raw"] for x in group) / len(group),
            "conversions": sum(x["conversions"] for x in group),
            "budget_utilization": sum(x["cost"] for x in group) / sum(x["budget"] for x in group),
        }
    state = {
        "started_at": started_at,
        "updated_at": datetime.now().astimezone().isoformat(),
        "completed_runs": len(markers), "requested_runs": 48 * 2 * 5,
        "completed_advertiser_episodes": len({(m["advertiser"], m["episode"])
                                              for m in markers.values() if m["strategy"] == "agent_memory"}),
        "requested_advertiser_episodes": 48 * 2,
        "llm_usage": dict(usage), "by_episode_strategy": aggregates,
    }
    write_json(output / "progress.json", state)
    prior.experiment.write_csv(output / "all_completed_runs.csv", rows)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-requests", type=int, default=20_000)
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prior.torch.set_num_threads(1)
    source = source_manifest()
    selected = selected_advertisers()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["engine_fingerprint"] != source["environment_fingerprint"]:
            raise RuntimeError("Engine fingerprint changed; cannot resume")
        started_at = manifest["started_at"]
    else:
        started_at = datetime.now().astimezone().isoformat()
        manifest = {
            "protocol": "48 advertisers x Episode 0 cold start + Episode 1 frozen Episode-0 memory",
            "selected_advertisers": selected, "episodes": list(EPISODES),
            "arms_each_episode": ["agent_memory", *prior.BASELINES],
            "requested_advertiser_episodes": 96, "requested_runs": 480,
            "model": MODEL, "seed": source["seed"], "pv_num": source["pv_num"],
            "official_commit": source["official_commit"], "assets": source["assets"],
            "kb_sha256": source["kb_sha256"],
            "engine_source_hashes": source["source_hashes"],
            "engine_fingerprint": source["environment_fingerprint"],
            "orchestrator_sha256": prior.experiment.sha256_file(Path(__file__)),
            "reused_source": None,
            "reuse_rule": "Resume completed runs only from this output directory",
            "development_advertiser": 0,
            "started_at": started_at,
        }
        write_json(manifest_path, manifest)
    progress(output, started_at)
    client = JSONClient(MODEL, max_requests=args.max_requests, proxy=args.proxy)
    preflight = client.ask_json('{"phase":"preflight","instruction":"Return exactly {\\"ok\\":true}","context":{}}')
    if preflight != {"ok": True}:
        raise RuntimeError("LLM preflight returned an unexpected response")
    kb = prior.experiment.build_kb_tool()
    try:
        for selected_row in selected:
            advertiser = selected_row["advertiser"]
            acquisition = prior.run_agent(client, kb, output, selected_row, 0, 0,
                                          source["environment_fingerprint"], source["seed"], source["pv_num"], None)
            progress(output, started_at)
            for baseline in prior.BASELINES:
                prior.run_baseline(output, selected_row, 0, baseline, source["seed"], source["pv_num"])
                progress(output, started_at)
            memory_dir = Path(acquisition["memory_dir"])
            if not (memory_dir / "principles.json").exists():
                raise RuntimeError("Completed cold start has no memory bank: " + str(memory_dir))
            prior.run_agent(client, kb, output, selected_row, 1, 1,
                            source["environment_fingerprint"], source["seed"], source["pv_num"], memory_dir)
            progress(output, started_at)
            for baseline in prior.BASELINES:
                prior.run_baseline(output, selected_row, 1, baseline, source["seed"], source["pv_num"])
                progress(output, started_at)
        final = progress(output, started_at)
        final["status"] = "complete"
        final["completed_at"] = datetime.now().astimezone().isoformat()
        write_json(output / "summary.json", final)
        print(json.dumps({"status": "complete", "completed_runs": final["completed_runs"]}), flush=True)
    except Exception as error:
        write_json(output / "last_failure.json", {
            "error": type(error).__name__ + ": " + str(error),
            "failed_at": datetime.now().astimezone().isoformat(),
            "resume_command": f"{sys.executable} launch.py --output-dir {output}",
        })
        progress(output, started_at)
        raise


if __name__ == "__main__":
    main()
