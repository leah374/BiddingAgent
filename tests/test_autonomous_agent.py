"""Behavioral contracts. Scripted LLM fixtures are NOT performance experiments."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

from autobid_agent.core import BiddingAgent, FINAL_FORMAT
from autobid_agent.contracts import SKILLS, TOOLS
from autobid_agent.memory import ExperienceStore, assess_window


def state(step=0):
    return dict(step=step, remaining_budget=2900., remaining_budget_ratio=1.,
                spent_ratio=0., elapsed_ratio=step / 48, pvalue_mean=.0005,
                pvalue_quantiles=[0, .0002, .0005, .0007, .001, .002],
                expected_conversions=0., expected_cpa=None, cpa=None, cpa_constraint=100.,
                score=0., historical_market={}, historical_winning={})


def outcome(cost=100., conv=2., expected=2.):
    return dict(cost=cost, conversions=conv, expected_conversions=expected,
                wins=100, score=conv, cpa=cost / conv if conv else None)


class ScriptedClient:
    """Controlled tool decisions and both polarities, for mechanism testing only."""
    def __init__(self, scale=1., broken=False, illegal=False, weights=None, alias_keys=False,
                 fake_principles=False, bad_evidence=False):
        self.scale, self.broken, self.illegal = scale, broken, illegal
        self.weights = weights or {"bcq": 4., "onlinelp": 3., "iql": 2., "mopo": 1.}
        self.alias_keys = alias_keys
        self.fake_principles = fake_principles
        self.bad_evidence = bad_evidence
        self.calls = []

    def ask_json(self, prompt):
        request = json.loads(prompt)
        self.calls.append(request)
        if self.broken:
            raise RuntimeError("injected provider outage")
        phase, context = request["phase"], request["context"]
        if phase == "analyze":
            result = {"analyses": [dict(skill=name, conclusion="测试分析结论", data=dict(signal="mixed", direction="hold"),
                                        evidence_ids=["pvalue_mean"], uncertainties=[]) for name in SKILLS]}
            if self.alias_keys:
                result["analyses"][1]["concentration"] = result["analyses"][1].pop("conclusion")
                result["analyses"][2]["data"]["方向"] = result["analyses"][2]["data"].pop("direction")
            if self.bad_evidence:
                result["analyses"][0]["evidence_ids"] = ["pvalue_mean", "step_29_data", "recent_memory[0].outcome"]
            return result
        if phase == "decide":
            if self.illegal:
                return dict(action="finalize", tool_weights={"bcq": .5, "onlinelp": .5, "iql": 0, "gave": 1},
                            scale=1., segment_multipliers=[1.] * 4, applied_principle_ids=[],
                            plan={"intent": "illegal", "reconsider_if": "always"}, reason="injected disallowed tool")
            result = copy.deepcopy(FINAL_FORMAT)
            result.update(tool_weights=copy.deepcopy(self.weights), scale=self.scale, reason="synthesize the four observed expert bids",
                          applied_principle_ids=(["invented_principle"] if self.fake_principles else
                                                 [p["principle_id"] for p in context["principles"][:1]]))
            return result
        if phase == "reflect":
            return dict(summary="Synthetic fixture, not an auction finding", lessons=[
                dict(type="guiding", description="测试：满足条件时组合专家", conditions=["synthetic_condition"], action="组合",
                     tool_weight_advice={name: "hold" for name in TOOLS}, scale_advice="hold", segment_advice=["hold"] * 4, limitation="仅为测试样本"),
                dict(type="cautionary", description="测试：风险高时避免盲目加价", conditions=["synthetic_risk"], action="避免",
                     tool_weight_advice={name: "decrease" for name in TOOLS}, scale_advice="down", segment_advice=["down"] * 4, limitation="仅为测试样本")])
        if phase == "deduplicate":
            return {"equivalent": False}
        raise AssertionError(phase)


class AgentContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = ExperienceStore(self.root / "memory")
        self.called = []

    def tearDown(self):
        self.temporary.cleanup()

    def tool(self, name):
        self.called.append(name)
        return dict(valid=True, tool=name, alpha=100.)

    def agent(self, cycle=0, client=None, read=True, adv=0, env="env1"):
        return BiddingAgent(client or ScriptedClient(), self.store, self.root / ("cycle_%s_%s" % (cycle, adv)), adv, cycle, env, read)

    def roll(self, agent, steps=4, observed=None):
        for step in range(steps):
            decision = agent.decide(state(step), self.tool)
            agent.observe(decision, observed or outcome())

    def test_expert_panel_calls_all_tools_and_blends(self):
        agent = self.agent()
        decision = agent.decide(state(), self.tool)
        self.assertEqual(self.called, list(TOOLS))
        self.assertEqual(decision["executed_action"]["tool_weights"],
                         {"bcq": .4, "onlinelp": .3, "iql": .2, "mopo": .1})
        self.assertEqual(decision["main_llm_calls"], 1)
        self.assertEqual(len(decision["analyses"]), 3)

    def test_negative_weights_are_guarded_and_normalized(self):
        weights = {"bcq": -1., "onlinelp": 2., "iql": 1., "mopo": 1.}
        decision = self.agent(client=ScriptedClient(weights=weights)).decide(state(), self.tool)
        self.assertEqual(decision["executed_action"]["tool_weights"],
                         {"bcq": 0., "onlinelp": .5, "iql": .25, "mopo": .25})
        self.assertIn("expert_weights_nonnegative_normalized", decision["guardrail"]["reasons"])

    def test_unambiguous_analysis_key_aliases_are_repaired_and_logged(self):
        agent = self.agent(client=ScriptedClient(alias_keys=True))
        result = agent.decide(state(), self.tool)
        self.assertFalse(result["analysis_fallback"])
        calls = [json.loads(line) for line in (agent.output_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(calls[0]["repaired"])

    def test_unsupported_evidence_is_removed_and_audited_without_fallback(self):
        agent = self.agent(client=ScriptedClient(bad_evidence=True))
        result = agent.decide(state(), self.tool)
        self.assertFalse(result["analysis_fallback"])
        first = result["analyses"][0]
        self.assertEqual(first["evidence_ids"], ["pvalue_mean", "recent_memory"])
        self.assertIn("unsupported_evidence_reference_removed", first["uncertainties"])
        calls = [json.loads(line) for line in (agent.output_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertTrue(calls[0]["repaired"])
        self.assertIn("step_29_data", calls[0]["raw_response"]["analyses"][0]["evidence_ids"])

    def test_unretrieved_principle_ids_are_removed_without_losing_weights(self):
        result = self.agent(client=ScriptedClient(fake_principles=True)).decide(state(), self.tool)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["proposed_action"]["applied_principle_ids"], ["invented_principle"])
        self.assertEqual(result["executed_action"]["applied_principle_ids"], [])
        self.assertIn("unretrieved_principle_ids_removed", result["guardrail"]["reasons"])

    def test_guardrail_logs_proposal_and_clipped_execution(self):
        decision = self.agent(client=ScriptedClient(scale=9)).decide(state(), self.tool)
        self.assertEqual(decision["proposed_action"]["scale"], 9)
        self.assertEqual(decision["executed_action"]["scale"], 1.2)
        self.assertTrue(decision["guardrail"]["intervened"])

    def test_high_cpa_risk_blocks_upscale(self):
        current = state()
        current.update(expected_conversions=2, expected_cpa=150.)
        result = self.agent(client=ScriptedClient(scale=1.2)).decide(current, self.tool)
        self.assertEqual(result["executed_action"]["scale"], 1.)
        self.assertIn("CPA_risk_no_upscale", result["guardrail"]["reasons"])

    def test_exhausted_budget_still_decides_but_cannot_spend(self):
        current = state()
        current["remaining_budget"] = 0.
        result = self.agent().decide(current, self.tool)
        self.assertEqual(result["executed_action"]["scale"], 0.)
        self.assertGreater(result["main_llm_calls"], 0)

    def test_provider_failure_has_bounded_official_fallback(self):
        result = self.agent(client=ScriptedClient(broken=True)).decide(state(), self.tool)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["main_llm_calls"], 1)
        self.assertEqual(result["executed_action"]["tool_weights"], {name: .25 for name in TOOLS})

    def test_gave_and_unknown_tools_are_rejected(self):
        result = self.agent(client=ScriptedClient(illegal=True)).decide(state(), self.tool)
        self.assertTrue(result["fallback"])
        self.assertEqual(self.called, list(TOOLS))
        self.assertEqual(set(TOOLS), {"bcq", "onlinelp", "iql", "mopo"})

    def test_all_tools_failed_returns_zero_safe_action(self):
        def failed(name):
            raise ValueError("injected invalid candidate")
        result = self.agent(client=ScriptedClient(broken=True)).decide(state(), failed)
        self.assertEqual(result["executed_action"]["tool_weights"], {name: 0. for name in TOOLS})

    def test_card_created_after_four_outcomes_not_before(self):
        agent = self.agent()
        self.roll(agent, 3)
        self.assertEqual(agent.cards, [])
        result = agent.decide(state(3), self.tool)
        agent.observe(result, outcome())
        self.assertEqual(agent.cards[0]["step_range"], [0, 3])
        self.assertEqual(agent.cards[0]["schema_version"], 2)
        self.assertEqual(set(agent.cards[0]["ensemble_summary"]["tool_weights"]), set(TOOLS))
        self.assertEqual(len(list((self.store.directory / "cards").glob("*.json"))), 1)
        with self.assertRaises(ValueError):
            agent.observe(result, outcome())

    def test_both_polarities_persist_and_snapshot_is_frozen(self):
        agent = self.agent()
        self.roll(agent)
        with self.assertRaises(ValueError):
            agent.finish({"fixture": True})
        agent.finish({"fixture": True}, expected_steps=4)
        self.assertEqual(agent.snapshot, [])
        reloaded = ExperienceStore(self.store.directory)
        self.assertEqual({p["type"] for p in reloaded.snapshot(0, 1, "env1")}, {"guiding", "cautionary"})
        self.assertEqual(reloaded.snapshot(0, 0, "env1"), [])
        self.assertEqual(reloaded.snapshot(1, 1, "env1"), [])
        self.assertEqual(reloaded.snapshot(0, 1, "different_scale"), [])

    def test_negative_source_does_not_penalize_new_warning(self):
        agent = self.agent()
        self.roll(agent, observed=outcome(cost=200, conv=0, expected=1))
        self.assertEqual(agent.cards[0]["outcome"]["assessment"], "negative")
        agent.finish({"fixture": True}, expected_steps=4)
        for principle in self.store.principles:
            self.assertEqual(principle["evaluated_windows"], 0)

    def test_only_applied_principle_receives_reuse_credit(self):
        first = self.agent()
        self.roll(first)
        first.finish({"fixture": True}, expected_steps=4)
        second = self.agent(1)
        self.roll(second)
        retrieved_only = next(p for p in self.store.principles if p["applied_count"] == 0)
        self.assertEqual(retrieved_only["retrieval_count"], 4)
        self.assertEqual(retrieved_only["evaluated_windows"], 0)
        self.assertEqual(sum(p["evaluated_windows"] for p in self.store.principles), 1)

    def test_full_two_cycle_contract_has_24_nonoverlapping_cards(self):
        first = self.agent()
        self.roll(first, 48)
        first.finish({"fixture": True})
        second = self.agent(1)
        self.roll(second, 48)
        second.finish({"fixture": True})
        self.assertEqual(len(first.cards), 12)
        self.assertEqual(len(second.cards), 12)
        self.assertEqual([c["step_range"] for c in second.cards], [[x, x + 3] for x in range(0, 48, 4)])
        self.assertTrue(all(row["retrieved_principle_ids"] for row in second.rows))
        self.assertEqual(self.agent(2, read=False).snapshot, [])

    def test_zero_conversion_is_uncertain_and_cpa_undefined(self):
        rows = [dict(state=state(), outcome=outcome(cost=1, conv=0, expected=.01)) for _ in range(4)]
        assessment = assess_window(rows, 100.)
        self.assertIsNone(assessment["cpa"])
        self.assertEqual(assessment["assessment"], "uncertain")


class OfficialToolContracts(unittest.TestCase):
    def test_four_real_tools_share_state_and_preserve_global_rng(self):
        import numpy as np
        import torch
        from autobid_agent.auctionnet import OfficialToolbox, IsolatedBCQBaseline
        pool = OfficialToolbox(seed=2026, advertiser=0)
        class Owner:
            budget, remaining_budget, cpa, category = 2900., 2900., 100., 0
        pvalues = np.linspace(.0001, .001, 100)
        args = (0, pvalues, pvalues * .1, [], [], [], [], [])
        pool.prepare(Owner(), args)
        before_np, before_py, before_torch = np.random.get_state(), random.getstate(), torch.get_rng_state().clone()
        results = {name: pool.call(name) for name in TOOLS}
        self.assertEqual(set(results), set(TOOLS))
        self.assertTrue(all(result["valid"] and result["count"] == 100 for result in results.values()))
        self.assertTrue(np.array_equal(before_np[1], np.random.get_state()[1]))
        self.assertEqual(before_np[2:], np.random.get_state()[2:])
        self.assertEqual(before_py, random.getstate())
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        # Baseline and BCQ tool must use the same inference RNG convention.
        baseline = IsolatedBCQBaseline(2026, 0)
        baseline.reset()
        self.assertTrue(np.array_equal(baseline.bidding(*args), pool.bids["bcq"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
