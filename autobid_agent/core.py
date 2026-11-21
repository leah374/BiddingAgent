"""A bounded observe/expert-panel/act loop; the LLM synthesizes expert bids."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import time

from .contracts import SKILLS, TOOLS, validate_analyses, validate_decision, validate_reflection
from .memory import ExperienceStore, append_json, assess_window, fingerprint


TOOL_DESCRIPTIONS = {
    "bcq": "Official BCQ checkpoint: generates actions near the offline behavior distribution, then ranks perturbed candidates with Q estimates. Useful as a distribution-constrained reference; neither low prices nor optimality are guaranteed. Returns actual alpha/bid statistics, not a calibrated confidence score.",
    "onlinelp": "Official OnlineLP lookup policy: looks up category/time cumulative cost versus remaining budget, uses the associated realCPA as alpha, capped at 1.5 times CPA target. This is a historical table heuristic, not solving a fresh LP or observing current competitor bids. Its table version is fixed in the manifest.",
    "iql": "Official IQL checkpoint: offline expectile value estimation and advantage-weighted actor training; inference outputs the actor's action from the observed history. Offers a different value/behavior tradeoff from BCQ. No inference-time Q ranking or calibrated confidence is exposed.",
    "mopo": "Official MOPO policy checkpoint: trained with model-generated rollouts and uncertainty penalties. Runtime uses the learned policy only; it does not expose model uncertainty or run counterfactual auctions. Its behavior may shift under a different state distribution."
}

ANALYSIS_FORMAT = {"analyses": [{"skill": "one of the three skill names", "conclusion": "short Chinese explanation",
                    "data": {"signal": "favorable|unfavorable|mixed|unknown", "direction": "up|down|hold|uncertain"},
                    "evidence_ids": ["exact top-level evidence key"], "uncertainties": ["short limitation"]}]}
FINAL_FORMAT = {"action": "finalize", "tool_weights": {"bcq": 0.25, "onlinelp": 0.25,
                 "iql": 0.25, "mopo": 0.25}, "scale": 1.0,
                "segment_multipliers": [1, 1, 1, 1], "applied_principle_ids": [],
                "plan": {"intent": "intent for up to the next four steps", "reconsider_if": "replan trigger"},
                "reason": "short evidence-grounded Chinese rationale"}


def brief_rows(rows):
    """Keep exact recent outcomes without repeating static KB blocks in every call."""
    keys = ("step", "remaining_budget_ratio", "pvalue_mean", "cpa_constraint", "expected_cpa", "score")
    return [dict(state={key: row["state"][key] for key in keys}, outcome=row["outcome"],
                 executed_action=row["executed_action"], guardrail=row["guardrail"],
                 candidate_alphas={event["tool"]: event.get("result", {}).get("alpha")
                                   for event in row.get("tool_events", []) if "result" in event},
                 applied_principle_ids=row["applied_principle_ids"]) for row in rows]


class BiddingAgent:
    def __init__(self, client, store, output_dir, advertiser, cycle, environment,
                 read_memory=True, max_tool_rounds=3):
        self.client, self.store = client, store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.advertiser, self.cycle, self.environment = advertiser, cycle, environment
        self.max_tool_rounds = max_tool_rounds
        self.snapshot = store.snapshot(advertiser, cycle, environment) if read_memory else []
        self.snapshot_hash = fingerprint(self.snapshot)
        self.rows, self.cards, self.plan = [], [], None
        self.errors = 0
        self.repairs = 0
        self.current_step = -1
        directory = Path(__file__).parent / "skills"
        self.skills = {name: (directory / name / "SKILL.md").read_text(encoding="utf-8") for name in SKILLS}
        self.skill_hashes = {name: fingerprint(text) for name, text in self.skills.items()}

    def _request(self, phase, instruction, context, validator):
        feedback = ""
        for attempt in range(2):
            prompt = json.dumps(dict(phase=phase, instruction=instruction, context=context,
                                     validation_feedback=feedback), ensure_ascii=False, allow_nan=False)
            started = time.perf_counter()
            event = dict(phase=phase, step=self.current_step, cycle=self.cycle, attempt=attempt, prompt=prompt)
            before_usage = self.client.usage() if hasattr(self.client, "usage") else {}
            try:
                value = self.client.ask_json(prompt)
                event["raw_response"] = copy.deepcopy(value)
                validator(value)
                event["response"] = value
                event["repaired"] = event["raw_response"] != value
                self.repairs += int(event["repaired"])
                event["valid"] = True
                return value
            except Exception as error:
                self.errors += 1
                feedback = type(error).__name__ + ": " + str(error)[:200]
                event.update(valid=False, error=feedback)
                if attempt == 1:
                    raise
            finally:
                event["duration_seconds"] = round(time.perf_counter() - started, 4)
                if hasattr(self.client, "usage"):
                    event["usage"] = {key: value - before_usage.get(key, 0) for key, value in self.client.usage().items()}
                append_json(self.output_dir / "llm_calls.jsonl", event)

    def decide(self, state, execute_tool):
        """Return validated action parameters; executor never receives LLM code."""
        self.current_step = state["step"]
        if state["step"] != len(self.rows):
            raise ValueError("Decisions must be sequential and outcomes recorded exactly once")
        retrieved = ExperienceStore.retrieve(self.snapshot, state)
        ids = [p["principle_id"] for p in retrieved]
        evidence = copy.deepcopy(state)
        context = dict(evidence=evidence, recent_memory=brief_rows(self.rows[-4:]), principles=retrieved,
                       previous_plan=self.plan)
        instruction = dict(skills=self.skills, output=ANALYSIS_FORMAT,
                           rules="Execute all three skills. Return three separate analyses in one JSON request. Numeric facts are supplied by the program; cite their evidence IDs, do not invent measurements. Keep each conclusion under 100 Chinese characters.")
        def check_analyses(value):
            # Repair only unambiguous key-name slips observed from the provider;
            # values and conclusions are never synthesized by this adapter.
            for item in value.get("analyses", []) if isinstance(value, dict) else []:
                data = item.get("data")
                if isinstance(data, dict) and "direction" not in data and "方向" in data:
                    data["direction"] = data.pop("方向")
                if "conclusion" not in item:
                    expected = {"skill", "conclusion", "data", "evidence_ids", "uncertainties"}
                    extras = [key for key in item if key not in expected and isinstance(item[key], str)]
                    if len(extras) == 1:
                        item["conclusion"] = item.pop(extras[0])
            allowed_roots = set(evidence) | {"recent_memory", "principles", "previous_plan"}
            validate_analyses(value)
            for item in value["analyses"]:
                supported = []
                removed = False
                for reference in item["evidence_ids"]:
                    root = reference.split(".", 1)[0].split("[", 1)[0]
                    if root in allowed_roots:
                        if root not in supported:
                            supported.append(root)
                    else:
                        removed = True
                item["evidence_ids"] = supported
                if removed and "unsupported_evidence_reference_removed" not in item["uncertainties"]:
                    item["uncertainties"] = (item["uncertainties"][:7] +
                                             ["unsupported_evidence_reference_removed"])
            validate_analyses(value)
        analysis_fallback = False
        try:
            analyses = self._request("analyze", instruction, context, check_analyses)["analyses"]
        except Exception:
            analysis_fallback = True
            analyses = [dict(skill=name, conclusion="分析不可用，按已观测数据保守决策。",
                             data=dict(signal="unknown", direction="uncertain"), evidence_ids=[],
                             uncertainties=["analysis_failed"]) for name in SKILLS]
        # Exact numeric metrics are attached by code rather than re-generated by a model.
        metric_keys = {"market_value": ("pvalue_mean", "pvalue_quantiles", "historical_market"),
                       "budget_control": ("spent_ratio", "elapsed_ratio", "cpa", "expected_cpa", "cpa_constraint"),
                       "experience_review": ("historical_winning",)}
        for item in analyses:
            item["data"]["metrics"] = {key: state.get(key) for key in metric_keys[item["skill"]]}
        candidates, events = {}, []

        def call_tool(name):
            if name in candidates:
                events.append(dict(tool=name, cached=True))
                return
            try:
                result = execute_tool(name)
                if result.get("valid") is not True:
                    raise ValueError("Invalid tool result")
                candidates[name] = result
                events.append(dict(tool=name, cached=False, result=result))
            except Exception as error:
                events.append(dict(tool=name, cached=False, error=type(error).__name__))

        # The user asked for an expert committee: all four official proposals are
        # observations. The LLM synthesizes weights rather than choosing one arm.
        for name in TOOLS:
            call_tool(name)
        policy_context = dict(state=evidence, analyses=analyses, principles=retrieved,
                              recent_memory=brief_rows(self.rows[-4:]), previous_plan=self.plan,
                              tools=TOOL_DESCRIPTIONS, candidates=candidates)
        instruction = dict(
            objective="Maximize final official beta=2 score. Budget utilization alone is not the objective. No future or other advertisers' hidden state is available.",
            finalize=FINAL_FORMAT,
            rules="All four official experts have already returned current bids. Synthesize their evidence into ONE weighted bid vector; do not mechanically select one expert. Assign finite weights to exactly bcq/onlinelp/iql/mopo, with positive weight on at least two successful experts and non-positive weight on failed experts. The program normalizes nonnegative weights. Then choose overall scale and four pValue-segment factors. Default scale near 1; guardrails clip to [0.8,1.2], risk-high upper 1.0, segment factors [0.95,1.05]. applied_principle_ids may contain ONLY exact IDs present in context.principles; when principles is empty it MUST be []. A warning may be applied by down-weighting a risky expert pattern. Reassess the short plan every step. No extra fields.")
        main_calls = 1
        try:
            decision = self._request("decide", instruction, policy_context,
                                     lambda value: validate_decision(value, candidates, ids))
        except Exception:
            decision = None
        fallback = decision is None
        if fallback:
            equal = 1.0 / len(candidates) if candidates else 0.0
            decision = dict(action="finalize", tool_weights={name: equal if name in candidates else 0.0 for name in TOOLS}, scale=1.0,
                            segment_multipliers=[1.0] * 4, applied_principle_ids=[],
                            plan=dict(intent="Use an equal-weight valid expert panel conservatively", reconsider_if="Next step or restored LLM"),
                            reason="Fallback to an equal-weight blend of available official proposals; zero bid if all tools failed.")
        proposed = copy.deepcopy(decision)
        decision["applied_principle_ids"] = [pid for pid in decision["applied_principle_ids"] if pid in ids]
        raw_weights = {name: max(0.0, float(decision["tool_weights"].get(name, 0.0)))
                       if name in candidates else 0.0 for name in TOOLS}
        weight_total = sum(raw_weights.values())
        if weight_total > 0:
            if abs(weight_total - 1.0) <= 1e-9:
                decision["tool_weights"] = raw_weights
            else:
                decision["tool_weights"] = {name: raw_weights[name] / weight_total for name in TOOLS}
        else:
            decision["tool_weights"] = {name: 0.0 for name in TOOLS}
        risk = state["expected_conversions"] >= 1 and state["expected_cpa"] is not None and state["expected_cpa"] > 1.2 * state["cpa_constraint"]
        upper = 1.0 if risk else 1.2
        decision["scale"] = min(upper, max(0.8, decision["scale"]))
        decision["segment_multipliers"] = [min(1.05, max(0.95, x)) for x in decision["segment_multipliers"]]
        reasons = []
        if proposed["applied_principle_ids"] != decision["applied_principle_ids"]:
            reasons.append("unretrieved_principle_ids_removed")
        if any(abs(float(proposed["tool_weights"].get(name, 0.0)) - decision["tool_weights"][name]) > 1e-9
               for name in TOOLS):
            reasons.append("expert_weights_nonnegative_normalized")
        if proposed["scale"] != decision["scale"]:
            reasons.append("CPA_risk_no_upscale" if risk else "relative_scale_limit")
        if proposed["segment_multipliers"] != decision["segment_multipliers"]:
            reasons.append("segment_multiplier_limit")
        if state["remaining_budget"] < state.get("min_remaining_budget", 0.1):
            decision["scale"] = 0.0
            reasons.append("budget_exhausted_zero_bid")
        self.plan = dict(decision["plan"], made_at=state["step"], valid_until=min(state["step"] + 3, 47))
        return dict(state=state, analyses=analyses, retrieved_principle_ids=ids,
                    applied_principle_ids=decision["applied_principle_ids"], tool_events=events,
                    proposed_action=proposed, executed_action=decision, plan=self.plan,
                    guardrail=dict(intervened=bool(reasons), reasons=reasons, risk_high=risk),
                    fallback=fallback, analysis_fallback=analysis_fallback,
                    main_llm_calls=main_calls, snapshot_hash=self.snapshot_hash)

    def observe(self, decision, outcome):
        step = decision["state"]["step"]
        if step != len(self.rows):
            raise ValueError("Duplicate or out-of-order outcome")
        row = dict(state=decision["state"], outcome=outcome,
                   analyses=decision["analyses"], tool_events=decision["tool_events"],
                   proposed_action=decision["proposed_action"], executed_action=decision["executed_action"],
                   guardrail=decision["guardrail"], retrieved_principle_ids=decision["retrieved_principle_ids"],
                   applied_principle_ids=decision["applied_principle_ids"])
        self.rows.append(row)
        if len(self.rows) % 4:
            return None
        window = self.rows[-4:]
        assessment = assess_window(window, row["state"]["cpa_constraint"])
        instruction = dict(
            rules="Reflect on these four actual ensemble-bidding steps. Keep both useful strategies and mistakes. Analyze the executed expert weights, overall scale, four segment factors, guardrails and observed outcomes. Numerical assessment is observational, not causal proof. All four candidates were generated but only their blend was executed, so do NOT invent counterfactual score for an expert alone. Winning is not conversion; expected conversions are not realized outcomes. Return up to TWO conditional guiding/cautionary principles, or none if evidence is insufficient. Each lesson must give qualitative advice for all four expert weights, scale and four segments. Evidence from one window is limited; preserve this limitation. Chinese descriptions.",
            output={"summary": "short factual ensemble summary", "lessons": [{"type": "guiding|cautionary", "description": "conditional lesson", "conditions": ["applicable condition"], "action": "what to do or avoid", "tool_weight_advice": {"bcq": "increase|decrease|hold|unknown", "onlinelp": "increase|decrease|hold|unknown", "iql": "increase|decrease|hold|unknown", "mopo": "increase|decrease|hold|unknown"}, "scale_advice": "up|down|hold|unknown", "segment_advice": ["up|down|hold|unknown", "up|down|hold|unknown", "up|down|hold|unknown", "up|down|hold|unknown"], "limitation": "evidence limitation"}]})
        try:
            reflection = self._request("reflect", instruction, dict(window=brief_rows(window), assessment=assessment), validate_reflection)
            error = None
        except Exception:
            reflection = dict(summary="反思调用不可用；保留原始窗口事实，不生成策略建议。", lessons=[])
            error = "reflection_failed"
        weights = {name: [item["executed_action"]["tool_weights"][name] for item in window] for name in TOOLS}
        ensemble_summary = dict(
            tool_weights={name: {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}
                          for name, values in weights.items()},
            dominant_tool_by_step=[max(TOOLS, key=lambda name: item["executed_action"]["tool_weights"][name]) for item in window],
            effective_expert_count_by_step=[sum(value >= 0.05 for value in item["executed_action"]["tool_weights"].values()) for item in window],
            scale_by_step=[item["executed_action"]["scale"] for item in window],
            segment_multipliers_by_step=[item["executed_action"]["segment_multipliers"] for item in window])
        card = dict(schema_version=2, card_id="adv_%02d_cycle_%03d_steps_%02d_%02d" % (self.advertiser, self.cycle, step - 3, step),
                    advertiser_id=self.advertiser, cycle_id=self.cycle,
                    environment=self.environment, step_range=[step - 3, step],
                    context={key: window[0]["state"][key] for key in ("step", "remaining_budget_ratio", "pvalue_mean")},
                    trajectory_refs=["decisions.jsonl#step=%s" % item["state"]["step"] for item in window],
                    trajectory_file=str((self.output_dir / "decisions.jsonl").resolve()),
                    outcome=assessment, ensemble_summary=ensemble_summary,
                    expert_candidates_by_step=[{event["tool"]: event.get("result", {}) for event in item.get("tool_events", []) if "result" in event} for item in window],
                    summary=reflection["summary"], lessons=reflection["lessons"],
                    retrieved_by_step=[item["retrieved_principle_ids"] for item in window],
                    applied_by_step=[item["applied_principle_ids"] for item in window],
                    reflection_error=error)
        self.store.record_card(card)
        self.cards.append(card)
        return card

    def finish(self, summary, expected_steps=48):
        if len(self.rows) != expected_steps or expected_steps % 4:
            raise ValueError("Only a completed cycle can publish reusable principles")
        def judge(old, new):
            try:
                result = self._request("deduplicate", "Return {\"equivalent\":true|false}. True only for the same recommendation AND compatible applicability conditions. A conditional contradiction is NOT equivalent.",
                                       dict(old=old, new=new),
                                       lambda value: (_ for _ in ()).throw(ValueError("Invalid equivalence"))
                                       if set(value) != {"equivalent"} or not isinstance(value["equivalent"], bool) else None)
                return result["equivalent"]
            except Exception:
                return False
        self.store.consolidate(self.cards, judge)
        self.store.finish_cycle(self.cycle, self.advertiser, summary)
