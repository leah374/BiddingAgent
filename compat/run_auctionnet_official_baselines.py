"""Evaluate AuctionNet's nine official baseline policies in one market protocol.

Each policy sequentially replaces the requested advertiser positions while the
other 47 bidders remain the untouched official AuctionNet lineup.  All models
and lookup tables are loaded locally; no LLM or external API is used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from compat import run_auctionnet_agent_vs_gave_episode as experiment
from simul_bidding_env.strategy.bc_bidding_strategy import BcBiddingStrategy
from simul_bidding_env.strategy.bcq_bidding_strategy import BcqBiddingStrategy
from simul_bidding_env.strategy.cql_bidding_strategy import CqlBiddingStrategy
from simul_bidding_env.strategy.iql_bidding_strategy import IqlBiddingStrategy
from simul_bidding_env.strategy.mbrl_combomicro_bidding_strategy import (
    MbrlComboMicroBiddingStrategy,
)
from simul_bidding_env.strategy.mbrl_mopo_bidding_strategy import MbrlMopoBiddingStrategy
from simul_bidding_env.strategy.onlinelp_bidding_strategy import OnlineLpBiddingStrategy
from simul_bidding_env.strategy.pid_bidding_strategy import PidBiddingStrategy
from simul_bidding_env.strategy.td3_bc_bidding_strategy import TD3_BCBiddingStrategy


ROOT = Path(__file__).resolve().parents[1]


STRATEGIES: Dict[str, Callable[[], Any]] = {
    "pid": lambda: PidBiddingStrategy(exp_tempral_ratio=np.ones(48)),
    "onlinelp": lambda: OnlineLpBiddingStrategy(episode=0),
    "bc": BcBiddingStrategy,
    "bcq": BcqBiddingStrategy,
    "cql": CqlBiddingStrategy,
    "iql": IqlBiddingStrategy,
    "td3_bc": TD3_BCBiddingStrategy,
    "mopo": MbrlMopoBiddingStrategy,
    "combo_micro": MbrlComboMicroBiddingStrategy,
}


class TracedOfficialStrategy(experiment.BaseBiddingStrategy):
    """Delegate bidding to one official strategy and expose common trace hooks."""

    def __init__(self, strategy_key: str) -> None:
        super().__init__(budget=2900, name=f"official_{strategy_key}", cpa=100, category=0)
        self.strategy_key = strategy_key
        self.delegate = STRATEGIES[strategy_key]()
        self.trace: List[Dict[str, Any]] = []
        self.pending: Dict[str, Any] | None = None

    def _sync_delegate(self) -> None:
        self.budget = float(self.budget)
        self.cpa = float(self.cpa)
        self.category = int(self.category)
        self.remaining_budget = float(self.remaining_budget)
        self.delegate.budget = self.budget
        self.delegate.cpa = self.cpa
        self.delegate.category = self.category
        self.delegate.remaining_budget = self.remaining_budget

    def reset(self) -> None:
        self.remaining_budget = float(self.budget)
        self._sync_delegate()
        self.delegate.reset()
        self.trace = []
        self.pending = None

    def bidding(
        self,
        timeStepIndex,
        pValues,
        pValueSigmas,
        historyPValueInfo,
        historyBid,
        historyAuctionResult,
        historyImpressionResult,
        historyLeastWinningCost,
    ):
        self._sync_delegate()
        bids = np.asarray(
            self.delegate.bidding(
                timeStepIndex,
                pValues,
                pValueSigmas,
                historyPValueInfo,
                historyBid,
                historyAuctionResult,
                historyImpressionResult,
                historyLeastWinningCost,
            ),
            dtype=float,
        ).reshape(-1)
        pvalues = np.asarray(pValues, dtype=float).reshape(-1)
        denom = float(np.dot(pvalues, pvalues))
        alpha = float(np.dot(bids, pvalues) / denom) if denom > 0 else 0.0
        self.pending = {
            "time_step": int(timeStepIndex),
            "base_alpha": alpha,
            "final_alpha": alpha,
            "adjustment": 0.0,
            "direction": "hold",
            "decision_source": f"auctionnet_official_{self.strategy_key}",
            "reason": "Official AuctionNet baseline action.",
            "llm_called": False,
            "segment_strategy": {},
        }
        return bids

    def record_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(self.pending or {})
        row.update(outcome)
        self.trace.append(row)
        self.pending = None
        return row

    def record_inactive(self, time_step: int, outcome: Dict[str, Any]) -> Dict[str, Any]:
        self.pending = {
            "time_step": int(time_step),
            "base_alpha": 0.0,
            "final_alpha": 0.0,
            "adjustment": 0.0,
            "direction": "hold",
            "decision_source": "budget_exhausted",
            "reason": "Remaining budget below AuctionNet minimum.",
            "llm_called": False,
            "segment_strategy": {},
        }
        return self.record_outcome(outcome)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pv-num", type=int, default=500_000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=4)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=list(STRATEGIES),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_budget = sum(float(row["budget"]) for row in rows)
    total_cost = sum(float(row["cost"]) for row in rows)
    return {
        "evaluated_runs": len(rows),
        "total_budget": total_budget,
        "total_cost": total_cost,
        "budget_utilization": total_cost / total_budget if total_budget else 0.0,
        "total_conversions": sum(float(row["conversions"]) for row in rows),
        "total_expected_conversions": sum(float(row["expected_conversions"]) for row in rows),
        "sum_official_score_raw": sum(float(row["official_score_raw"]) for row in rows),
        "mean_official_score_raw": (
            sum(float(row["official_score_raw"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.start_index <= args.end_index <= 47:
        raise ValueError("Expected 0 <= start-index <= end-index <= 47")
    output_dir = args.output_dir or (
        ROOT / "outputs" / f"auctionnet_official_baselines_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    by_strategy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for strategy_key in args.strategies:
        for advertiser_index in range(args.start_index, args.end_index + 1):
            arm_name = f"{strategy_key}_adv{advertiser_index:02d}"
            arm_dir = output_dir / strategy_key / arm_name
            arm_dir.mkdir(parents=True, exist_ok=True)
            summary_path = arm_dir / f"{arm_name}_summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                print(f"[resume] strategy={strategy_key} advertiser={advertiser_index}", flush=True)
            else:
                strategy = TracedOfficialStrategy(strategy_key)
                summary, _ = experiment.run_arm(
                    arm_name,
                    strategy,
                    arm_dir,
                    args.episode,
                    args.seed,
                    args.pv_num,
                    player_index=advertiser_index,
                )
                summary["baseline"] = strategy_key
                experiment.write_json(summary_path, summary)
            summaries.append(summary)
            by_strategy[strategy_key].append(summary)
            experiment.write_csv(output_dir / "all_runs.csv", summaries)
            progress = {
                "auctionnet_commit": experiment.official_commit(),
                "episode": args.episode,
                "seed": args.seed,
                "pv_num": args.pv_num,
                "advertiser_indices": [args.start_index, args.end_index],
                "uses_llm": False,
                "completed_runs": len(summaries),
                "requested_runs": len(args.strategies) * (args.end_index - args.start_index + 1),
                "strategies": {
                    key: aggregate(strategy_rows) for key, strategy_rows in by_strategy.items()
                },
            }
            experiment.write_json(output_dir / "summary.json", progress)
            print(
                f"[progress] {len(summaries)}/{progress['requested_runs']} "
                f"strategy={strategy_key} advertiser={advertiser_index} "
                f"score={summary['official_score_raw']:.3f} cost={summary['cost']:.2f}",
                flush=True,
            )

    experiment.write_json(
        output_dir / "run_config.json",
        {
            "episode": args.episode,
            "seed": args.seed,
            "pv_num": args.pv_num,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "strategies": args.strategies,
            "uses_llm": False,
            "evaluation": "each baseline replaces one advertiser; other 47 are official opponents",
        },
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
