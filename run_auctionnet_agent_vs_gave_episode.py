"""Paired 48-step AuctionNet rollout comparing the project agent with GAVE.

The official AuctionNet checkout stays unmodified.  Each arm uses the official
48-advertiser market and replaces advertiser 0 with the evaluated policy; the
other 47 advertisers are the official AuctionNet strategies and checkpoints.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# AuctionNet's official runner still imports ``Iterable`` from ``collections``.
# Python 3.10+ moved it to ``collections.abc``; expose the compatibility alias
# before importing the untouched official package.
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parent
AUCTIONNET_ROOT = ROOT / "third_party" / "AuctionNet"
GAVE_CODE_DIR = WORKSPACE_ROOT / "gave" / "code"
CHECKPOINT_DIR = ROOT / "checkpoints" / "gave_full_steps300000_seed2026_20260510_202440"
CHECKPOINT = CHECKPOINT_DIR / "200000.pt"
KB_PATH = ROOT / "cache" / "experience_kb_v1_3fa1b08629.sqlite"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs"

sys.path.insert(0, str(GAVE_CODE_DIR))
sys.path.insert(0, str(AUCTIONNET_ROOT / "strategy_train_env"))
sys.path.insert(0, str(AUCTIONNET_ROOT))

import bidding_agent_demo_v2 as agentlib  # noqa: E402
from run.run_test import adjust_over_cost, get_winner  # noqa: E402
from simul_bidding_env.Controller.Controller import Controller  # noqa: E402
from simul_bidding_env.strategy.base_bidding_strategy import BaseBiddingStrategy  # noqa: E402


EXPECTED_OFFICIAL_OPPONENTS = Counter(
    {
        "PidBiddingStrategy": 5,
        "IqlBiddingStrategy": 6,
        "TD3_BCBiddingStrategy": 6,
        "OnlineLpBiddingStrategy": 12,
        "CqlBiddingStrategy": 3,
        "BcBiddingStrategy": 3,
        "MbrlMopoBiddingStrategy": 6,
        "BcqBiddingStrategy": 3,
        "MbrlComboMicroBiddingStrategy": 3,
    }
)

EXPECTED_FULL_OFFICIAL_LINEUP = EXPECTED_OFFICIAL_OPPONENTS + Counter(
    {"PidBiddingStrategy": 1}
)


def replaced_official_strategy(player_index: int) -> str:
    """Return the official strategy type replaced at a controller index."""
    slot = player_index % 8
    category = player_index // 8
    if category % 2 == 0:
        pattern = (
            "PidBiddingStrategy",
            "IqlBiddingStrategy",
            "TD3_BCBiddingStrategy",
            "OnlineLpBiddingStrategy",
            "OnlineLpBiddingStrategy",
            "CqlBiddingStrategy",
            "BcBiddingStrategy",
            "MbrlMopoBiddingStrategy",
        )
    else:
        pattern = (
            "PidBiddingStrategy",
            "BcqBiddingStrategy",
            "MbrlMopoBiddingStrategy",
            "OnlineLpBiddingStrategy",
            "OnlineLpBiddingStrategy",
            "TD3_BCBiddingStrategy",
            "IqlBiddingStrategy",
            "MbrlComboMicroBiddingStrategy",
        )
    return pattern[slot]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(AUCTIONNET_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def official_checkout_clean() -> bool:
    return not bool(
        subprocess.check_output(
            ["git", "-C", str(AUCTIONNET_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )


def strategy_name(agent: object) -> str:
    underlying = getattr(agent, "player_agent", agent)
    return type(underlying).__name__


def score(reward: float, cpa: float, cpa_constraint: float, beta: int = 2) -> float:
    penalty = 1.0
    if cpa > cpa_constraint:
        penalty = (cpa_constraint / (cpa + 1e-10)) ** beta
    return penalty * reward


class DashScopeJSONClient:
    """Small thread-safe OpenAI-compatible client with explicit proxy support."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        proxy: Optional[str],
        timeout: int,
        retries: int,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.proxy = proxy
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        self.request_count = 0
        self.failed_attempts = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _opener(self) -> urllib.request.OpenerDirector:
        if self.proxy:
            return urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": self.proxy, "https": self.proxy})
            )
        return urllib.request.build_opener()

    def ask_json(self, prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
            "enable_thinking": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert RTB auto-bidding agent. Return strict JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                with self._opener().open(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                with self._lock:
                    self.request_count += 1
                    self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    self.completion_tokens += int(usage.get("completion_tokens") or 0)
                return agentlib.extract_json_object(content)
            except Exception as exc:
                last_error = exc
                with self._lock:
                    self.failed_attempts += 1
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries + 1} attempts: {last_error}")

    def usage(self) -> Dict[str, int]:
        with self._lock:
            return {
                "successful_requests": self.request_count,
                "failed_attempts": self.failed_attempts,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            }


class ExistingKBSource:
    """Metadata shim allowing the compact KB to be used without the 100+ GB raw DB."""

    def __init__(self, kb_path: Path) -> None:
        import sqlite3

        with sqlite3.connect(kb_path) as conn:
            metadata = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        self.sqlite_path = Path(metadata["source_sqlite"])
        self._rows = int(metadata["source_rows_loaded"])

    def row_count(self) -> int:
        return self._rows

    def source_rows_loaded(self) -> int:
        return self._rows


def build_kb_tool() -> agentlib.ExperienceKnowledgeTool:
    source = ExistingKBSource(KB_PATH)
    return agentlib.ExperienceKnowledgeTool(source, KB_PATH, force_rebuild=False)


def build_model_param() -> Dict[str, Any]:
    model_param = json.loads((CHECKPOINT_DIR / "train_param.json").read_text(encoding="utf-8"))
    model_param["device"] = "cpu"
    model_param["save_dir"] = str(CHECKPOINT_DIR)
    model_param["gave_target_return"] = 0.01
    return model_param


def activate_gave_namespace() -> None:
    """Resolve AuctionNet/GAVE's shared ``bidding_train_env`` package name.

    AuctionNet's official strategy classes are imported above and retain their
    resolved class references.  GAVE inference is then loaded from the sibling
    GAVE checkout under the same upstream package name.
    """

    for module_name in list(sys.modules):
        if module_name == "bidding_train_env" or module_name.startswith("bidding_train_env."):
            del sys.modules[module_name]
    gave_path = str(GAVE_CODE_DIR)
    sys.path[:] = [entry for entry in sys.path if entry != gave_path]
    sys.path.insert(0, gave_path)


def history_dict(
    history_pvalue_infos: List[np.ndarray],
    history_bids: List[np.ndarray],
    history_auction_results: List[np.ndarray],
    history_impression_results: List[np.ndarray],
    history_least_winning_costs: List[np.ndarray],
) -> Dict[str, list]:
    return {
        "historyPValueInfo": history_pvalue_infos,
        "historyBids": history_bids,
        "historyAuctionResult": history_auction_results,
        "historyImpressionResult": history_impression_results,
        "historyLeastWinningCost": history_least_winning_costs,
    }


class GaveAuctionNetStrategy(BaseBiddingStrategy):
    def __init__(self, checkpoint: Path, model_param: Dict[str, Any]) -> None:
        super().__init__(budget=2900, name="GAVE", cpa=100, category=0)
        activate_gave_namespace()
        self.gave_tool = agentlib.GaveTool(checkpoint, model_param)
        self.trace: List[Dict[str, Any]] = []
        self.pending: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        # Controller assigns NumPy scalar configuration values.  Convert them
        # once so the same values can be embedded in LLM JSON prompts.
        self.budget = float(self.budget)
        self.cpa = float(self.cpa)
        self.category = int(self.category)
        self.remaining_budget = float(self.budget)
        self.trace = []
        self.pending = None

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        historyPValueInfo: List[np.ndarray],
        historyBid: List[np.ndarray],
        historyAuctionResult: List[np.ndarray],
        historyImpressionResult: List[np.ndarray],
        historyLeastWinningCost: List[np.ndarray],
    ) -> np.ndarray:
        self.remaining_budget = float(self.remaining_budget)
        history = history_dict(
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )
        alpha, bids = self.gave_tool.bid(
            timeStepIndex, pValues, self.budget, self.cpa, self.remaining_budget, history
        )
        self.pending = {
            "time_step": timeStepIndex,
            "base_alpha": alpha,
            "final_alpha": alpha,
            "adjustment": 0.0,
            "direction": "hold",
            "decision_source": "gave",
            "reason": "Pure GAVE action.",
            "llm_called": False,
            "segment_strategy": {},
        }
        return np.asarray(bids, dtype=float)

    def record_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(self.pending or {})
        row.update(outcome)
        self.trace.append(row)
        self.pending = None
        return row

    def record_inactive(self, time_step: int, outcome: Dict[str, Any]) -> Dict[str, Any]:
        self.pending = {
            "time_step": time_step,
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


class AgentAuctionNetStrategy(GaveAuctionNetStrategy):
    def __init__(
        self,
        checkpoint: Path,
        model_param: Dict[str, Any],
        kb_tool: agentlib.ExperienceKnowledgeTool,
        llm_client: DashScopeJSONClient,
        decision_interval: int,
        scale_calibrator: Optional[Any] = None,
    ) -> None:
        super().__init__(checkpoint, model_param)
        self.name = "GAVE_KB_LLM_AGENT"
        self.kb_tool = kb_tool
        self.adjustment_agent = agentlib.AdjustmentAnalysisAgent(
            llm_client=llm_client, print_agent_outputs=False
        )
        self.final_agent = agentlib.FinalBidAgent(min_alpha=0.0)
        self.scale_calibrator = scale_calibrator
        self.decision_interval = max(int(decision_interval), 1)
        self.last_decision: Optional[agentlib.AdjustmentDecision] = None
        self.last_sql_summary: Optional[Dict[str, Any]] = None
        self.decision_memory_rows: List[Dict[str, Any]] = []
        self.cumulative_cost = 0.0
        self.cumulative_conversions = 0.0
        self.cumulative_expected_conversions = 0.0

    def reset(self) -> None:
        super().reset()
        self.last_decision = None
        self.last_sql_summary = None
        self.decision_memory_rows = []
        self.cumulative_cost = 0.0
        self.cumulative_conversions = 0.0
        self.cumulative_expected_conversions = 0.0

    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        historyPValueInfo: List[np.ndarray],
        historyBid: List[np.ndarray],
        historyAuctionResult: List[np.ndarray],
        historyImpressionResult: List[np.ndarray],
        historyLeastWinningCost: List[np.ndarray],
    ) -> np.ndarray:
        self.remaining_budget = float(self.remaining_budget)
        history = history_dict(
            historyPValueInfo,
            historyBid,
            historyAuctionResult,
            historyImpressionResult,
            historyLeastWinningCost,
        )
        raw_gave_alpha, base_bids = self.gave_tool.bid(
            timeStepIndex, pValues, self.budget, self.cpa, self.remaining_budget, history
        )
        base_alpha = float(raw_gave_alpha)
        scale_calibration: Dict[str, Any] = {}
        if self.scale_calibrator is not None:
            base_alpha, scale_calibration = self.scale_calibrator.calibrate(
                raw_alpha=base_alpha,
                time_step_index=timeStepIndex,
                budget=self.budget,
                cpa=self.cpa,
                remaining_budget=self.remaining_budget,
                cumulative_cost=self.cumulative_cost,
                cumulative_expected_conversions=self.cumulative_expected_conversions,
                history_pvalue_info=historyPValueInfo,
                history_least_winning_cost=historyLeastWinningCost,
            )
            base_bids = float(base_alpha) * np.asarray(pValues, dtype=float)
        should_call_llm = (
            self.last_decision is None
            or self.last_sql_summary is None
            or timeStepIndex % self.decision_interval == 0
        )
        if should_call_llm:
            pvalue_stats = {
                "current_pvalue_mean": float(np.mean(pValues)) if len(pValues) else 0.0,
                "current_pv_count": int(len(pValues)),
                "current_budget": float(self.budget),
                "current_pvalue_min": float(np.min(pValues)) if len(pValues) else 0.0,
                "current_pvalue_p25": float(np.quantile(pValues, 0.25)) if len(pValues) else 0.0,
                "current_pvalue_p50": float(np.quantile(pValues, 0.50)) if len(pValues) else 0.0,
                "current_pvalue_p75": float(np.quantile(pValues, 0.75)) if len(pValues) else 0.0,
                "current_pvalue_p90": float(np.quantile(pValues, 0.90)) if len(pValues) else 0.0,
                "current_pvalue_max": float(np.max(pValues)) if len(pValues) else 0.0,
            }
            sql_summary = self.kb_tool.summarize(
                advertiser_number=0.0,
                category=float(self.category),
                time_step_index=timeStepIndex,
                **pvalue_stats,
            )
            agentlib.attach_current_segment_mix(sql_summary, pValues)
            sql_summary = json_safe(sql_summary)
            current_state = agentlib.state_to_dict(
                agentlib.build_gave_state(
                    time_step_index=timeStepIndex,
                    p_values=pValues,
                    budget=self.budget,
                    remaining_budget=self.remaining_budget,
                    history=history,
                )
            )
            current_state["score_status"] = agentlib.build_score_status(
                cost_so_far=self.cumulative_cost,
                conversions_so_far=self.cumulative_conversions,
                expected_conversions_so_far=self.cumulative_expected_conversions,
                cpa_constraint=self.cpa,
            )
            current_state["score_optimization_hint"] = (
                "AuctionNet online rollout: optimize official beta=2 score while respecting CPA and budget."
            )
            current_state = json_safe(current_state)
            memory = json_safe(agentlib.build_decision_memory(self.decision_memory_rows))
            decision = self.adjustment_agent.decide(
                base_alpha=base_alpha,
                sql_summary=sql_summary,
                time_step_index=timeStepIndex,
                budget=self.budget,
                cpa=self.cpa,
                remaining_budget=self.remaining_budget,
                history=history,
                current_state=current_state,
                decision_memory=memory,
            )
            self.last_decision = decision
            self.last_sql_summary = sql_summary
        else:
            assert self.last_decision is not None and self.last_sql_summary is not None
            decision = agentlib.AdjustmentDecision(
                self.last_decision.direction,
                self.last_decision.adjustment_range,
                self.last_decision.adjustment,
                (
                    f"Reused timestep {timeStepIndex - timeStepIndex % self.decision_interval} "
                    f"LLM strategy: {self.last_decision.reason}"
                )[:360],
                dict(self.last_decision.pvalue_multipliers),
                "interval_reuse_last_llm",
            )
            sql_summary = self.last_sql_summary
            current_state = agentlib.state_to_dict(
                agentlib.build_gave_state(
                    time_step_index=timeStepIndex,
                    p_values=pValues,
                    budget=self.budget,
                    remaining_budget=self.remaining_budget,
                    history=history,
                )
            )
            current_state["score_status"] = agentlib.build_score_status(
                cost_so_far=self.cumulative_cost,
                conversions_so_far=self.cumulative_conversions,
                expected_conversions_so_far=self.cumulative_expected_conversions,
                cpa_constraint=self.cpa,
            )
            current_state = json_safe(current_state)
            memory = json_safe(agentlib.build_decision_memory(self.decision_memory_rows))

        final_alpha, bids, final_source, segment_strategy = self.final_agent.finalize(
            base_alpha,
            base_bids,
            decision,
            sql_summary=sql_summary,
            budget=self.budget,
            remaining_budget=self.remaining_budget,
            cpa=self.cpa,
            time_step_index=timeStepIndex,
            p_values=pValues,
        )
        self.pending = {
            "time_step": timeStepIndex,
            "raw_gave_alpha": float(raw_gave_alpha),
            "base_alpha": float(base_alpha),
            "final_alpha": float(final_alpha),
            "adjustment": float(decision.adjustment),
            "direction": decision.direction,
            "decision_source": decision.source,
            "final_source": final_source,
            "reason": decision.reason,
            "llm_called": should_call_llm,
            "pvalue_multipliers": dict(decision.pvalue_multipliers),
            "value_segment": sql_summary.get("pvalue_context", {})
            .get("decision_basis", {})
            .get("segment", "unknown"),
            "segment_strategy": segment_strategy,
            "state": current_state,
            "decision_memory": memory,
            "knowledge_backend": sql_summary.get("query_scope", "unknown"),
            "scale_calibration": scale_calibration,
        }
        return np.asarray(bids, dtype=float)

    def record_outcome(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        row = super().record_outcome(outcome)
        self.cumulative_cost = float(outcome["cumulative_cost"])
        self.cumulative_conversions = float(outcome["cumulative_conversions"])
        self.cumulative_expected_conversions = float(outcome["cumulative_expected_conversions"])
        self.decision_memory_rows.append(
            {
                "timeStepIndex": row["time_step"],
                "value_segment": row.get("value_segment", "unknown"),
                "base_alpha": row.get("base_alpha", 0.0),
                "adjustment": row.get("adjustment", 0.0),
                "final_alpha": row.get("final_alpha", 0.0),
                "multipliers": row.get("pvalue_multipliers", {}),
                "reason": row.get("reason", ""),
                "cost": outcome["cost"],
                "win_count": outcome["wins"],
                "conversions": outcome["conversions"],
                "expected_conversions": outcome["expected_conversions"],
                "remaining_budget_ratio_after": outcome["remaining_budget_ratio"],
                "cpa_so_far_after": outcome["cpa"],
                "expected_cpa_so_far_after": outcome["expected_cpa"],
                "score_so_far_after": outcome["score"],
                "score1_so_far_after": outcome["score_beta5"],
                "expected_score_so_far_after": outcome["expected_score"],
                "expected_score1_so_far_after": outcome["expected_score_beta5"],
                "cpa_constraint": self.cpa,
            }
        )
        return row


def pv_signature(controller: Controller) -> str:
    digest = hashlib.sha256()
    for pvalues, sigmas in zip(
        controller.pvGenerator.pv_values, controller.pvGenerator.pValueSigmas
    ):
        digest.update(np.ascontiguousarray(pvalues).tobytes())
        digest.update(np.ascontiguousarray(sigmas).tobytes())
    return digest.hexdigest()


def validate_lineup(controller: Controller, player_index: int = 0) -> Dict[str, int]:
    if len(controller.agents) != 48:
        raise AssertionError(f"Expected 48 advertisers, got {len(controller.agents)}")
    category_counts = Counter(int(agent.category) for agent in controller.agents)
    if category_counts != Counter({category: 8 for category in range(6)}):
        raise AssertionError(f"Unexpected category layout: {category_counts}")
    opponent_counts = Counter(
        strategy_name(agent)
        for index, agent in enumerate(controller.agents)
        if index != player_index
    )
    expected_opponents = EXPECTED_FULL_OFFICIAL_LINEUP.copy()
    expected_opponents[replaced_official_strategy(player_index)] -= 1
    if opponent_counts != expected_opponents:
        raise AssertionError(
            f"Official opponent lineup changed: expected={expected_opponents}, "
            f"actual={opponent_counts}"
        )
    return dict(sorted(opponent_counts.items()))


def run_arm(
    arm_name: str,
    strategy: GaveAuctionNetStrategy,
    output_dir: Path,
    episode: int,
    seed: int,
    pv_num: int,
    player_index: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    agentlib.set_seed(seed)
    random.seed(seed)
    started = time.perf_counter()
    controller = Controller(
        player_index=player_index,
        player_agent=strategy,
        num_tick=48,
        num_agent_category=8,
        num_category=6,
        pv_num=pv_num,
        pv_generator_type="neuripsPvGen",
    )
    official_opponents = validate_lineup(controller, player_index)
    controller.reset(episode=episode)
    signature = pv_signature(controller)

    agents = controller.agents
    env = controller.biddingEnv
    pv_generator = controller.pvGenerator
    history_pvalue_infos: List[np.ndarray] = []
    history_bids: List[np.ndarray] = []
    history_auction_results: List[np.ndarray] = []
    history_impression_results: List[np.ndarray] = []
    history_least_winning_costs: List[np.ndarray] = []
    cumulative_cost = 0.0
    cumulative_conversions = 0.0
    cumulative_expected_conversions = 0.0
    rows: List[Dict[str, Any]] = []
    decision_path = output_dir / f"{arm_name}_decision_trace.jsonl"
    tick_path = output_dir / f"{arm_name}_ticks.csv"
    if decision_path.exists():
        decision_path.unlink()

    for tick_index in range(48):
        tick_started = time.perf_counter()
        pv_values = pv_generator.pv_values[tick_index]
        pvalue_sigmas = pv_generator.pValueSigmas[tick_index]
        bid_columns = []
        player_was_active = agents[player_index].remaining_budget >= env.min_remaining_budget
        for index, competitor in enumerate(agents):
            if competitor.remaining_budget < env.min_remaining_budget and not (
                index == player_index and getattr(strategy, "decide_when_inactive", False)
            ):
                bid_columns.append(np.zeros(pv_values.shape[0], dtype=float))
                continue
            bid_columns.append(
                competitor.bidding(
                    tick_index,
                    pv_values[:, index],
                    pvalue_sigmas[:, index],
                    [item[index] for item in history_pvalue_infos],
                    [item[index] for item in history_bids],
                    [item[index] for item in history_auction_results],
                    [item[index] for item in history_impression_results],
                    history_least_winning_costs,
                )
            )
        bids = np.asarray(bid_columns, dtype=float).T
        if bids.shape != pv_values.shape:
            raise AssertionError(f"Bid shape {bids.shape} != PV shape {pv_values.shape}")
        if not np.isfinite(bids).all():
            raise AssertionError(f"{arm_name} produced non-finite bids at tick {tick_index}")
        bids[bids < 0] = 0

        remaining_before = np.asarray([competitor.remaining_budget for competitor in agents], dtype=float)
        ratio_max: Optional[float] = None
        auction_iterations = 0
        while ratio_max is None or ratio_max > 0:
            if ratio_max is not None and ratio_max > 0:
                over_cost_ratio = np.maximum(
                    (step_cost - remaining_before) / (step_cost + 1e-4), 0
                )
                adjust_over_cost(bids, over_cost_ratio, env.slot_coefficients, winner_pit)
            (
                xi_pit,
                slot_pit,
                cost_pit,
                is_exposed_pit,
                conversion_action_pit,
                least_winning_cost_pit,
                _market_price_pit,
            ) = env.simulate_ad_bidding(pv_values, pvalue_sigmas, bids)
            auction_iterations += 1
            real_cost = cost_pit * is_exposed_pit
            step_cost = real_cost.sum(axis=1)
            winner_pit = get_winner(slot_pit)
            over_cost_ratio = np.maximum(
                (step_cost - remaining_before) / (step_cost + 1e-4), 0
            )
            ratio_max = float(over_cost_ratio.max())

        for index, competitor in enumerate(agents):
            competitor.remaining_budget = float(
                competitor.remaining_budget - float(step_cost[index])
            )

        player_cost = float(step_cost[player_index])
        player_wins = int(xi_pit[player_index].sum())
        player_exposures = int(is_exposed_pit[player_index].sum())
        player_conversions = float(conversion_action_pit[player_index].sum())
        player_expected_conversions = float(
            np.sum(pv_values[:, player_index] * is_exposed_pit[player_index])
        )
        cumulative_cost += player_cost
        cumulative_conversions += player_conversions
        cumulative_expected_conversions += player_expected_conversions
        player_cpa = cumulative_cost / (cumulative_conversions + 1e-10)
        player_expected_cpa = cumulative_cost / (cumulative_expected_conversions + 1e-10)
        outcome = {
            "pv_count": int(pv_values.shape[0]),
            "cost": player_cost,
            "wins": player_wins,
            "exposures": player_exposures,
            "conversions": player_conversions,
            "expected_conversions": player_expected_conversions,
            "remaining_budget": float(agents[player_index].remaining_budget),
            "remaining_budget_ratio": float(
                agents[player_index].remaining_budget / agents[player_index].budget
            ),
            "cumulative_cost": cumulative_cost,
            "cumulative_conversions": cumulative_conversions,
            "cumulative_expected_conversions": cumulative_expected_conversions,
            "cpa": player_cpa,
            "expected_cpa": player_expected_cpa,
            "score": score(cumulative_conversions, player_cpa, agents[player_index].cpa, beta=2),
            "score_beta5": score(cumulative_conversions, player_cpa, agents[player_index].cpa, beta=5),
            "expected_score": score(
                cumulative_expected_conversions,
                player_expected_cpa,
                agents[player_index].cpa,
                beta=2,
            ),
            "expected_score_beta5": score(
                cumulative_expected_conversions,
                player_expected_cpa,
                agents[player_index].cpa,
                beta=5,
            ),
            "bid_mean_after_budget_enforcement": float(np.mean(bids[:, player_index])),
            "bid_max_after_budget_enforcement": float(np.max(bids[:, player_index])),
            "auction_iterations": auction_iterations,
        }
        if hasattr(strategy, "record_executed_bids"):
            strategy.record_executed_bids(tick_index, bids[:, player_index].copy())
        if player_was_active or getattr(strategy, "decide_when_inactive", False):
            decision_row = strategy.record_outcome(outcome)
        else:
            decision_row = strategy.record_inactive(tick_index, outcome)
        append_jsonl(decision_path, decision_row)

        row = {
            "strategy": arm_name,
            "episode": episode,
            "advertiser_index": player_index,
            "time_step": tick_index,
            "pv_count": outcome["pv_count"],
            "active": int(player_was_active),
            "llm_called": int(bool(decision_row.get("llm_called"))),
            "base_alpha": decision_row.get("base_alpha", 0.0),
            "adjustment": decision_row.get("adjustment", 0.0),
            "final_alpha": decision_row.get("final_alpha", 0.0),
            "bid_mean": outcome["bid_mean_after_budget_enforcement"],
            "wins": player_wins,
            "exposures": player_exposures,
            "conversions": player_conversions,
            "expected_conversions": player_expected_conversions,
            "cost": player_cost,
            "cumulative_cost": cumulative_cost,
            "remaining_budget": outcome["remaining_budget"],
            "cumulative_conversions": cumulative_conversions,
            "cumulative_expected_conversions": cumulative_expected_conversions,
            "cpa": player_cpa,
            "expected_cpa": player_expected_cpa,
            "score": outcome["score"],
            "expected_score": outcome["expected_score"],
            "decision_source": decision_row.get("decision_source", "unknown"),
            "duration_seconds": round(time.perf_counter() - tick_started, 4),
        }
        rows.append(row)
        write_csv(tick_path, rows)

        pvalue_info = np.stack((pv_values.T, pvalue_sigmas.T), axis=-1)
        history_pvalue_infos.append(pvalue_info)
        history_bids.append(bids.T.copy())
        history_auction_results.append(np.stack((xi_pit, slot_pit, cost_pit), axis=-1))
        history_impression_results.append(
            np.stack((is_exposed_pit, conversion_action_pit), axis=-1)
        )
        history_least_winning_costs.append(least_winning_cost_pit.copy())
        print(
            f"[{arm_name}] step {tick_index + 1:02d}/48 "
            f"cost={player_cost:.2f} cum_cost={cumulative_cost:.2f} "
            f"conv={player_conversions:.0f} cum_conv={cumulative_conversions:.0f} "
            f"alpha={float(decision_row.get('final_alpha', 0.0)):.4f} "
            f"elapsed={time.perf_counter() - tick_started:.1f}s",
            flush=True,
        )

    final_cpa = cumulative_cost / (cumulative_conversions + 1e-10)
    final_expected_cpa = cumulative_cost / (cumulative_expected_conversions + 1e-10)
    summary = {
        "strategy": arm_name,
        "episode": episode,
        "decision_steps": 48,
        "advertiser_index": player_index,
        "category": int(agents[player_index].category),
        "budget": float(agents[player_index].budget),
        "cpa_constraint": float(agents[player_index].cpa),
        "total_pv": int(sum(row["pv_count"] for row in rows)),
        "wins": int(sum(row["wins"] for row in rows)),
        "exposures": int(sum(row["exposures"] for row in rows)),
        "conversions": cumulative_conversions,
        "expected_conversions": cumulative_expected_conversions,
        "cost": cumulative_cost,
        "remaining_budget": float(agents[player_index].remaining_budget),
        "budget_utilization": cumulative_cost / float(agents[player_index].budget),
        "cpa": final_cpa,
        "expected_cpa": final_expected_cpa,
        "cpa_exceedance_rate": (
            final_cpa - agents[player_index].cpa
        ) / agents[player_index].cpa,
        "official_score_raw": score(
            cumulative_conversions, final_cpa, agents[player_index].cpa, beta=2
        ),
        "official_score_normalized": score(
            cumulative_conversions, final_cpa, agents[player_index].cpa, beta=2
        )
        / 20000.0,
        "expected_score_raw": score(
            cumulative_expected_conversions,
            final_expected_cpa,
            agents[player_index].cpa,
            beta=2,
        ),
        "last_active_step": max((row["time_step"] for row in rows if row["wins"] > 0), default=-1),
        "pv_signature": signature,
        "official_opponent_count": 47,
        "official_opponent_strategy_counts": official_opponents,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "tick_csv": str(tick_path),
        "decision_trace_jsonl": str(decision_path),
    }
    write_json(output_dir / f"{arm_name}_summary.json", summary)
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--pv-num", type=int, default=500_000)
    parser.add_argument("--llm-model", default="qwen3.5-flash")
    parser.add_argument("--llm-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--llm-decision-interval", type=int, default=4)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--proxy", default="http://127.0.0.1:6952")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--only", choices=["both", "gave", "agent"], default="both")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (AUCTIONNET_ROOT, CHECKPOINT, CHECKPOINT_DIR / "normalize_dict.pkl", KB_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
    if args.only in {"both", "agent"} and not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or LLM_API_KEY is required for the agent arm")

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"auctionnet_agent_vs_gave_ep{args.episode}_{datetime.now():%Y%m%d_%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_param = build_model_param()
    llm_client = DashScopeJSONClient(
        api_key=api_key or "",
        model=args.llm_model,
        base_url=args.llm_base_url,
        proxy=args.proxy or None,
        timeout=args.llm_timeout,
        retries=args.llm_retries,
    )
    experiment_started = time.perf_counter()
    summaries: List[Dict[str, Any]] = []
    all_ticks: List[Dict[str, Any]] = []

    # A failed agent/API attempt should not force the deterministic GAVE arm to
    # be recomputed.  Reuse its fully written artifacts when resuming the same
    # output directory with ``--only agent``.
    existing_gave_summary = output_dir / "gave_summary.json"
    existing_gave_ticks = output_dir / "gave_ticks.csv"
    if args.only == "agent" and existing_gave_summary.exists() and existing_gave_ticks.exists():
        summaries.append(json.loads(existing_gave_summary.read_text(encoding="utf-8")))
        with existing_gave_ticks.open("r", newline="", encoding="utf-8-sig") as handle:
            all_ticks.extend(list(csv.DictReader(handle)))
        print(f"[resume] reused completed GAVE arm from {output_dir}", flush=True)

    if args.only in {"both", "gave"}:
        gave_strategy = GaveAuctionNetStrategy(CHECKPOINT, model_param)
        gave_summary, gave_ticks = run_arm(
            "gave", gave_strategy, output_dir, args.episode, args.seed, args.pv_num
        )
        summaries.append(gave_summary)
        all_ticks.extend(gave_ticks)

    if args.only in {"both", "agent"}:
        kb_tool = build_kb_tool()
        agent_strategy = AgentAuctionNetStrategy(
            CHECKPOINT,
            model_param,
            kb_tool,
            llm_client,
            decision_interval=args.llm_decision_interval,
        )
        agent_summary, agent_ticks = run_arm(
            "agent", agent_strategy, output_dir, args.episode, args.seed, args.pv_num
        )
        summaries.append(agent_summary)
        all_ticks.extend(agent_ticks)

    write_csv(output_dir / "comparison.csv", summaries)
    write_csv(output_dir / "all_tick_trajectories.csv", all_ticks)
    comparison: Dict[str, Any] = {
        "auctionnet_commit": official_commit(),
        "official_checkout_clean": official_checkout_clean(),
        "environment": {
            "episode": args.episode,
            "seed": args.seed,
            "decision_steps": 48,
            "configured_total_pv": args.pv_num,
            "advertiser_count": 48,
            "evaluated_advertiser": 0,
            "official_opponents": 47,
        },
        "gave": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "target_return": model_param["gave_target_return"],
        },
        "agent": {
            "base": "same GAVE checkpoint",
            "knowledge_base": str(KB_PATH),
            "knowledge_source_fold": "period-10 held out; periods 7-9 and 11-13 are experience",
            "llm_model": args.llm_model,
            "decision_interval": args.llm_decision_interval,
            "between_decisions": "reuse_last",
            "llm_usage": llm_client.usage(),
        },
        "results": summaries,
        "paired_environment_match": (
            len(summaries) == 2 and summaries[0]["pv_signature"] == summaries[1]["pv_signature"]
        ),
        "total_duration_seconds": round(time.perf_counter() - experiment_started, 3),
    }
    if len(summaries) == 2:
        gave = next(item for item in summaries if item["strategy"] == "gave")
        agent = next(item for item in summaries if item["strategy"] == "agent")
        comparison["agent_minus_gave"] = {
            key: float(agent[key]) - float(gave[key])
            for key in (
                "conversions",
                "expected_conversions",
                "cost",
                "budget_utilization",
                "cpa",
                "expected_cpa",
                "official_score_raw",
                "expected_score_raw",
            )
        }
    write_json(output_dir / "summary.json", comparison)
    print(json.dumps(json_safe(comparison), ensure_ascii=False, indent=2), flush=True)
    print(f"[done] outputs={output_dir}", flush=True)


if __name__ == "__main__":
    main()
