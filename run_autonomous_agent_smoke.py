"""One advertiser: 48-step acquisition, then paired memory/no-memory and BCQ.

Uses the four official tool policies and 47 untouched official opponents.
This is a live integration smoke test, not a statistical effectiveness study.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import time

from autobid_agent.client import JSONClient
from autobid_agent.auctionnet import AutonomousAuctionNetStrategy, IsolatedBCQBaseline
from autobid_agent.contracts import TOOLS
from autobid_agent.memory import ExperienceStore, fingerprint, write_json
import run_auctionnet_agent_vs_gave_episode as experiment
import torch

ROOT = Path(__file__).resolve().parent


def asset_manifest():
    official = experiment.AUCTIONNET_ROOT
    assets = official / "simul_bidding_env" / "strategy" / "official_agent"
    files = sorted(p for p in assets.rglob("*") if p.is_file() and p.suffix in (".pth", ".pt", ".pkl", ".csv"))
    return {str(path.relative_to(official)): experiment.sha256_file(path) for path in files}


def audit_strategy(strategy, summary):
    rows = strategy.trace
    called = Counter(event["tool"] for row in rows for event in row["tool_events"] if not event.get("cached"))
    weight_values = {name: [row["executed_action"]["tool_weights"][name] for row in rows] for name in TOOLS}
    return dict(steps=len(rows), analysis_skill_outputs=sum(len(row["analyses"]) for row in rows),
                cards=len(strategy.agent.cards), snapshot_principles=len(strategy.agent.snapshot),
                retrieved_steps=sum(bool(row["retrieved_principle_ids"]) for row in rows),
                applied_steps=sum(bool(row["applied_principle_ids"]) for row in rows),
                called_tools=dict(called),
                mean_tool_weights={name: sum(values) / len(values) if values else 0.0 for name, values in weight_values.items()},
                dominant_tools=dict(Counter(max(TOOLS, key=lambda name: row["executed_action"]["tool_weights"][name]) for row in rows)),
                multi_expert_blend_steps=sum(sum(value > 0 for value in row["executed_action"]["tool_weights"].values()) >= 2 for row in rows),
                material_multi_expert_steps=sum(sum(value >= .05 for value in row["executed_action"]["tool_weights"].values()) >= 2 for row in rows),
                complete_expert_panel_steps=sum({event["tool"] for event in row["tool_events"] if "result" in event} == set(TOOLS) for row in rows),
                fallback_steps=sum(row["fallback"] for row in rows),
                analysis_fallback_steps=sum(row["analysis_fallback"] for row in rows),
                reflection_failed_cards=sum(bool(card["reflection_error"]) for card in strategy.agent.cards),
                guardrail_steps=sum(row["guardrail"]["intervened"] for row in rows),
                environment_budget_intervention_steps=sum(row["environment_budget_intervention"] for row in rows),
                llm_invalid_or_failed=strategy.agent.errors,
                llm_deterministic_key_repairs=strategy.agent.repairs,
                card_assessments=dict(Counter(card["outcome"]["assessment"] for card in strategy.agent.cards)),
                lesson_types=dict(Counter(lesson["type"] for card in strategy.agent.cards for lesson in card["lessons"])),
                score=summary["official_score_raw"], cost=summary["cost"],
                budget_utilization=summary["budget_utilization"], conversions=summary["conversions"],
                expected_conversions=summary["expected_conversions"],
                cpa=summary["cpa"] if summary["conversions"] else None,
                expected_cpa=summary["expected_cpa"])


def report(output, results, audits, checks, usage, duration):
    payload = dict(results=results, audits=audits, checks=checks, llm_usage=usage,
                   duration_seconds=round(duration, 2),
                   interpretation="One advertiser and one acquisition/evaluation pair. Integration evidence only; score differences do not establish general improvement or causal memory benefit.")
    write_json(output / "summary.json", payload)
    lines = ["# Autonomous four-tool agent: minimal live smoke test", "",
             "One advertiser; original 48-agent market; 47 official opponents. Each arm has 48 steps.", "",
             "| Arm | Episode | Score (beta=2) | Cost | Budget used | Conversions | Expected conversions |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        lines.append("| %s | %s | %.4f | %.2f | %.2f%% | %.0f | %.4f |" % (
            name, result["episode"], result["official_score_raw"], result["cost"],
            result["budget_utilization"] * 100, result["conversions"], result["expected_conversions"]))
    lines.extend(["", "## Integration checks", ""])
    lines.extend("- %s: %s" % (name, "PASS" if passed else "FAIL") for name, passed in checks.items())
    lines.extend(["", "## Interpretation", "",
                  "Acquisition episode 0 differs from evaluation episode 1. Compare memory/no-memory/BCQ only within episode 1.",
                  "Same initial traffic and inference RNG protocol do not guarantee identical realized auctions after policies diverge.",
                  "The no-memory ablation disables long-term retrieval but retains the same skills, recent memory and reflection calls.",
                  "This single paired example is not evidence of statistical significance. Frozen episode-start memory prevents current-cycle leakage.",
                  "", "LLM usage: `%s`" % json.dumps(usage), ""])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--advertiser", type=int, default=0)
    parser.add_argument("--pv-num", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--llm-model", default="qwen3.5-flash")
    parser.add_argument("--max-requests", type=int, default=700)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--proxy", default=None)
    args = parser.parse_args()
    if not 0 <= args.advertiser < 48:
        parser.error("advertiser must be 0..47")
    output = args.output_dir or ROOT / "outputs" / ("autonomous_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if output.exists() and any(output.iterdir()):
        parser.error("Use a new output directory; completed experiments will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    client = JSONClient(args.llm_model, max_requests=args.max_requests, proxy=args.proxy)
    official_assets = asset_manifest()
    source_files = sorted((ROOT / "autobid_agent").rglob("*.py")) + sorted((ROOT / "autobid_agent").rglob("SKILL.md"))
    source_files += [Path(__file__), ROOT / "run_auctionnet_agent_vs_gave_episode.py"]
    source_hashes = {str(path.relative_to(ROOT)): experiment.sha256_file(path) for path in source_files}
    environment_spec = dict(official_commit=experiment.official_commit(), assets=official_assets,
                            pv_num=args.pv_num, pv_generator="neuripsPvGen", pvalue_mean_base=.0005,
                            score_beta=2, steps=48, tools=list(TOOLS), onlinelp_tool_table="episode-0.csv",
                            kb_sha256=experiment.sha256_file(experiment.KB_PATH),
                            source_hashes=source_hashes, model=args.llm_model)
    environment = fingerprint(environment_spec)
    manifest = dict(environment_spec, environment_fingerprint=environment, advertiser=args.advertiser,
                    seed=args.seed, python=sys.version, torch=torch.__version__,
                    official_checkout_clean_before=experiment.official_checkout_clean(),
                    episodes={"acquisition": 0, "evaluation": 1},
                    simulator="original official 48-agent market, one slot replaced",
                    llm_provider="Aliyun DashScope compatible API", max_requests=args.max_requests,
                    guardrails=dict(expert_weights="nonnegative, normalized, at least two positive",
                                    scale=[.8, 1.2], high_cpa_scale_upper=1., segments=[.95, 1.05]))
    write_json(output / "manifest.json", manifest)
    results, audits = {}, {}
    kb = experiment.build_kb_tool()

    def run_agent(name, episode, cycle, memory_dir, read):
        directory = output / name
        directory.mkdir()
        before = client.usage()
        strategy = AutonomousAuctionNetStrategy(client, ExperienceStore(memory_dir), directory,
                                                args.advertiser, cycle, environment,
                                                args.seed, read_memory=read, kb=kb)
        summary, _ = experiment.run_arm(name, strategy, directory, episode, args.seed, args.pv_num, player_index=args.advertiser)
        strategy.agent.finish(summary)
        audit = audit_strategy(strategy, summary)
        audit["llm_usage"] = {key: value - before.get(key, 0) for key, value in client.usage().items()}
        write_json(directory / "agent_audit.json", audit)
        results[name], audits[name] = summary, audit
        write_json(output / "progress.json", dict(completed=list(results), latest_audit=audit, usage=client.usage()))
        return strategy

    try:
        acquisition = run_agent("acquisition", 0, 0, output / "memory_acquisition", True)
        # Preserve the exact acquired bank. The two evaluation arms write to
        # separate stores and cannot observe each other's newly created cards.
        shutil.copytree(output / "memory_acquisition", output / "memory_reuse")
        shutil.copytree(output / "memory_acquisition", output / "memory_disabled")
        reuse = run_agent("with_memory", 1, 1, output / "memory_reuse", True)
        no_memory = run_agent("without_memory", 1, 1, output / "memory_disabled", False)
        baseline_dir = output / "bcq_baseline"
        baseline_dir.mkdir()
        baseline = IsolatedBCQBaseline(args.seed, args.advertiser)
        results["bcq_baseline"], _ = experiment.run_arm("bcq_baseline", baseline, baseline_dir,
                                                       1, args.seed, args.pv_num, player_index=args.advertiser)
        checks = {
            "four_official_tools_only": all(set(audit["called_tools"]) <= set(TOOLS) for audit in audits.values()),
            "all_four_experts_run_every_step": all(audit["complete_expert_panel_steps"] == 48 for audit in audits.values()),
            "agent_synthesizes_multiple_experts_every_step": all(audit["multi_expert_blend_steps"] == 48 for audit in audits.values()),
            "all_144_steps_have_three_skill_outputs": all(a["steps"] == 48 and a["analysis_skill_outputs"] == 144 for a in audits.values()),
            "twelve_cards_per_episode": all(a["cards"] == 12 for a in audits.values()),
            "real_llm_decisions_no_fallback": all(a["fallback_steps"] == 0 for a in audits.values()),
            "all_skill_analyses_valid": all(a["analysis_fallback_steps"] == 0 for a in audits.values()),
            "all_reflections_valid": all(a["reflection_failed_cards"] == 0 for a in audits.values()),
            "acquisition_has_no_previous_experience": audits["acquisition"]["retrieved_steps"] == 0,
            "next_cycle_retrieves_prior_experience": audits["with_memory"]["retrieved_steps"] == 48,
            "next_cycle_applies_some_experience": audits["with_memory"]["applied_steps"] > 0,
            "disabled_memory_retrieves_nothing": audits["without_memory"]["retrieved_steps"] == 0,
            "positive_and_negative_principles_retained": {p["type"] for p in acquisition.agent.store.principles} == {"guiding", "cautionary"},
            "paired_evaluation_traffic_identical": len({results[n]["pv_signature"] for n in ("with_memory", "without_memory", "bcq_baseline")}) == 1,
            "acquisition_and_evaluation_traffic_differ": results["acquisition"]["pv_signature"] != results["with_memory"]["pv_signature"],
            "official_47_opponents": all(r["official_opponent_count"] == 47 for r in results.values()),
            "no_budget_overspend": all(r["cost"] <= r["budget"] + 1e-6 for r in results.values()),
            "official_source_unchanged": experiment.official_checkout_clean(),
            "official_assets_unchanged": official_assets == asset_manifest(),
        }
        report(output, results, audits, checks, client.usage(), time.perf_counter() - started)
        print(json.dumps(dict(checks=checks, usage=client.usage(), output=str(output)), ensure_ascii=False), flush=True)
        if not all(checks.values()):
            raise RuntimeError("Smoke evidence incomplete; inspect REPORT.md and failed checks")
    except Exception as error:
        write_json(output / "failure.json", dict(error=type(error).__name__ + ": " + str(error),
                                                completed=list(results), usage=client.usage()))
        raise


if __name__ == "__main__":
    main()
