"""Adapters for the unmodified official AuctionNet strategies and simulator."""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from . import client as _https_bootstrap
from .contracts import TOOLS
from .core import BiddingAgent
from .memory import append_json, fingerprint
import run_auctionnet_agent_vs_gave_episode as experiment
import run_auctionnet_official_baselines as baselines


@contextmanager
def isolated_rng(seed, advertiser, step, tool):
    """Extra candidate calls cannot consume opponents'/environment's RNG stream."""
    np_state, py_state = np.random.get_state(), random.getstate()
    tool_seed = int(fingerprint([seed, advertiser, step, tool])[:8], 16)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(tool_seed)
            np.random.seed(tool_seed)
            random.seed(tool_seed)
            yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


class OfficialToolbox:
    def __init__(self, seed, advertiser):
        self.seed, self.advertiser = seed, advertiser
        self.policies = {name: baselines.STRATEGIES[name]() for name in TOOLS}
        self.bids = {}

    def prepare(self, owner, arguments):
        self.bids = {}
        self.arguments = arguments
        for policy in self.policies.values():
            # All four consume the same REAL history before their bids are blended.
            # These official implementations do not keep recurrent policy state.
            for field in ("budget", "remaining_budget", "category", "cpa"):
                setattr(policy, field, getattr(owner, field))

    def call(self, name):
        if name not in TOOLS:
            raise ValueError("Only BCQ, OnlineLP, IQL, MOPO are enabled")
        if name not in self.bids:
            with isolated_rng(self.seed, self.advertiser, self.arguments[0], name), torch.no_grad():
                result = np.asarray(self.policies[name].bidding(*self.arguments), dtype=float).reshape(-1)
            pvalues = self.arguments[1]
            if result.shape != pvalues.shape or not np.isfinite(result).all():
                raise ValueError("Invalid official policy output")
            self.bids[name] = np.maximum(result, 0)
        bids = self.bids[name]
        pvalues = self.arguments[1]
        denom = float(np.dot(pvalues, pvalues))
        return dict(valid=True, tool=name,
                    alpha=float(np.dot(bids, pvalues) / denom) if denom else 0.0,
                    bid_mean=float(bids.mean()) if bids.size else 0.0,
                    bid_max=float(bids.max()) if bids.size else 0.0,
                    count=int(bids.size), bid_sha256=hashlib.sha256(bids.tobytes()).hexdigest(),
                    evidence="official_checkpoint_or_lookup_output; no counterfactual outcome")


class AutonomousAuctionNetStrategy(experiment.BaseBiddingStrategy):
    decide_when_inactive = True

    def __init__(self, client, store, output_dir, advertiser, cycle, environment,
                 seed, read_memory=True, kb=None):
        super().__init__(budget=2900, name="autonomous_four_tools", cpa=100, category=0)
        self.agent = BiddingAgent(client, store, output_dir, advertiser, cycle, environment, read_memory)
        self.toolbox = OfficialToolbox(seed, advertiser)
        self.kb = kb or experiment.build_kb_tool()
        self.output_dir = Path(output_dir)
        (self.output_dir / "bid_arrays").mkdir(parents=True, exist_ok=True)
        self.trace, self.pending = [], None
        self.cumulative_cost = self.cumulative_conversions = self.cumulative_expected = 0.0

    def reset(self):
        if self.trace:
            raise RuntimeError("Create a fresh strategy per episode; do not overwrite a completed trace")
        self.remaining_budget = float(self.budget)

    def _state(self, step, pvalues):
        quantiles = np.quantile(pvalues, [0, .25, .5, .75, .9, 1]).tolist() if pvalues.size else [0.] * 6
        mean = float(pvalues.mean()) if pvalues.size else 0.0
        summary = self.kb.summarize(
            advertiser_number=float(self.agent.advertiser), category=float(self.category),
            time_step_index=step, current_pvalue_mean=mean, current_pv_count=len(pvalues),
            current_budget=float(self.budget), current_pvalue_min=quantiles[0],
            current_pvalue_p25=quantiles[1], current_pvalue_p50=quantiles[2],
            current_pvalue_p75=quantiles[3], current_pvalue_p90=quantiles[4], current_pvalue_max=quantiles[5])
        markets = {scope: {key: data.get(key) for key in
                          ("traffic_num", "avg_pvalue", "avg_market_price", "win_rate", "quantile_mode")}
                   for scope, data in summary.get("market_context", {}).items()}
        winning = {scope: {key: data.get(key) for key in ("win_count", "reference_alpha", "quantile_mode")}
                   for scope, data in summary.get("success_context", {}).get("win_reference_by_scope", {}).items()}
        cpa = self.cumulative_cost / self.cumulative_conversions if self.cumulative_conversions else None
        expected_cpa = self.cumulative_cost / self.cumulative_expected if self.cumulative_expected else None
        return dict(step=step, episode_steps=48, advertiser_id=self.agent.advertiser,
                    category=int(self.category), pv_count=len(pvalues), pvalue_mean=mean,
                    pvalue_quantiles=quantiles, budget=float(self.budget),
                    remaining_budget=float(self.remaining_budget), remaining_budget_ratio=float(self.remaining_budget / self.budget),
                    spent_ratio=float(1 - self.remaining_budget / self.budget), elapsed_ratio=step / 48,
                    cpa_constraint=float(self.cpa), cpa=cpa, expected_cpa=expected_cpa,
                    cumulative_cost=self.cumulative_cost, conversions=self.cumulative_conversions,
                    expected_conversions=self.cumulative_expected,
                    score=experiment.score(self.cumulative_conversions, cpa or 0, self.cpa),
                    min_remaining_budget=0.1,
                    historical_market=markets, historical_winning=winning,
                    kb_caveat="Historical aggregate references; xi=1 means win, not conversion; quantiles are mean proxies. No current competitor bids or future observations are exposed.")

    def bidding(self, timeStepIndex, pValues, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost):
        values = np.asarray(pValues, dtype=float)
        args = (timeStepIndex, values, pValueSigmas, historyPValueInfo, historyBid,
                historyAuctionResult, historyImpressionResult, historyLeastWinningCost)
        self.toolbox.prepare(self, args)
        decision = self.agent.decide(self._state(timeStepIndex, values), self.toolbox.call)
        action = decision["executed_action"]
        base = np.zeros_like(values)
        for name, weight in action["tool_weights"].items():
            if name in self.toolbox.bids:
                base = base + float(weight) * self.toolbox.bids[name]
        boundaries = np.quantile(values, [.25, .5, .75]) if len(values) else np.zeros(3)
        segments = np.searchsorted(boundaries, values, side="right")
        proposed = base * action["scale"] * np.asarray(action["segment_multipliers"])[segments]
        bids = np.clip(proposed, 0, max(0., float(self.remaining_budget)))
        if not np.array_equal(bids, proposed):
            decision["guardrail"]["intervened"] = True
            decision["guardrail"]["reasons"].append("per_bid_remaining_budget_cap")
        denominator = float(np.dot(values, values))
        base_alpha = float(np.dot(base, values) / denominator) if denominator else 0.
        final_alpha = float(np.dot(bids, values) / denominator) if denominator else 0.
        self.pending = dict(decision, time_step=timeStepIndex, base_alpha=base_alpha,
                            final_alpha=final_alpha, adjustment=final_alpha - base_alpha,
                            llm_called=True, decision_source="autonomous_tool_loop",
                            reason=action["reason"], bid_array_path=str(self.output_dir / "bid_arrays" / ("step_%02d.npz" % timeStepIndex)))
        self._proposed_bids = bids.copy()
        return bids

    def record_executed_bids(self, step, bids):
        """Called AFTER AuctionNet's own budget enforcement."""
        if self.pending is None:
            raise RuntimeError("No pending decision")
        changed = not np.array_equal(bids, self._proposed_bids)
        self.pending["environment_budget_intervention"] = changed
        self.pending["guardrail"]["environment_budget_intervention"] = changed
        self.pending["executed_bid_sha256"] = hashlib.sha256(np.asarray(bids, dtype=float).tobytes()).hexdigest()
        np.savez_compressed(self.pending["bid_array_path"], submitted=self._proposed_bids,
                            executed=bids, blended_base=sum(
                                self.pending["executed_action"]["tool_weights"].get(key, 0.0) * value
                                for key, value in self.toolbox.bids.items()),
                            **{"tool_" + key: value for key, value in self.toolbox.bids.items()})

    def record_outcome(self, outcome):
        row = self.pending
        if row is None:
            raise RuntimeError("Outcome without decision")
        self.cumulative_cost = float(outcome["cumulative_cost"])
        self.cumulative_conversions = float(outcome["cumulative_conversions"])
        self.cumulative_expected = float(outcome["cumulative_expected_conversions"])
        clean_outcome = dict(outcome)
        # The legacy reporter uses epsilon denominators. Agent memory uses null
        # for undefined CPA so zero conversions cannot be misread as a real CPA.
        clean_outcome["cpa"] = self.cumulative_cost / self.cumulative_conversions if self.cumulative_conversions else None
        clean_outcome["expected_cpa"] = self.cumulative_cost / self.cumulative_expected if self.cumulative_expected else None
        row["outcome"] = clean_outcome
        append_json(self.output_dir / "decisions.jsonl", row)
        self.agent.observe(row, clean_outcome)
        row.update(outcome)
        self.trace.append(row)
        self.pending = None
        return row

    def record_inactive(self, step, outcome):
        # The integration runner must still call the agent when the budget is gone.
        return self.record_outcome(outcome)


class IsolatedBCQBaseline(baselines.TracedOfficialStrategy):
    def __init__(self, seed, advertiser):
        super().__init__("bcq")
        self.seed, self.advertiser = seed, advertiser

    def bidding(self, *args):
        with isolated_rng(self.seed, self.advertiser, args[0], "bcq"):
            return super().bidding(*args)
