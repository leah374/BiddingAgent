"""Strict boundary validation without adding runtime dependencies."""
from __future__ import annotations
import math

TOOLS = ("bcq", "onlinelp", "iql", "mopo")
SKILLS = ("market_value", "budget_control", "experience_review")
SIGNALS = ("favorable", "unfavorable", "mixed", "unknown")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def strings(value, maximum=8):
    return isinstance(value, list) and len(value) <= maximum and all(isinstance(x, str) for x in value)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_analyses(value):
    require(set(value) == {"analyses"}, "Only analyses is allowed")
    items = value["analyses"]
    require(isinstance(items, list) and len(items) == 3, "Need all three skill analyses")
    require({x.get("skill") for x in items if isinstance(x, dict)} == set(SKILLS), "Wrong skill set")
    for item in items:
        require(set(item) == {"skill", "conclusion", "data", "evidence_ids", "uncertainties"}, "Invalid analysis fields")
        require(isinstance(item["conclusion"], str) and 0 < len(item["conclusion"]) <= 600, "Invalid conclusion")
        require(isinstance(item["data"], dict) and set(item["data"]) == {"signal", "direction"}, "Invalid skill data")
        require(item["data"]["signal"] in SIGNALS, "Invalid signal")
        require(item["data"]["direction"] in ("up", "down", "hold", "uncertain"), "Invalid direction")
        require(strings(item["evidence_ids"]) and strings(item["uncertainties"]), "Invalid analysis references")


def validate_decision(value, candidates, principle_ids):
    require(isinstance(value.get("reason"), str) and len(value["reason"]) <= 600, "Missing concise reason")
    require(value.get("action") == "finalize", "Expected ensemble finalize")
    require(set(value) == {"action", "tool_weights", "scale", "segment_multipliers",
                           "applied_principle_ids", "plan", "reason"}, "Invalid final decision fields")
    weights = value["tool_weights"]
    require(isinstance(weights, dict) and set(weights) == set(TOOLS), "Need weights for exactly four official tools")
    require(all(number(weights[name]) for name in TOOLS), "Tool weights must be finite")
    require(sum(weights[name] > 0 for name in candidates) >= min(2, len(candidates)),
            "Blend at least two successful expert proposals")
    require(all(weights[name] <= 0 for name in set(TOOLS) - set(candidates)), "Failed tools cannot receive positive weight")
    require(number(value["scale"]), "Scale must be finite")
    mult = value["segment_multipliers"]
    require(isinstance(mult, list) and len(mult) == 4 and all(number(x) for x in mult), "Invalid four segment multipliers")
    require(strings(value["applied_principle_ids"]), "Invalid applied experience IDs")
    plan = value["plan"]
    require(isinstance(plan, dict) and set(plan) == {"intent", "reconsider_if"}, "Invalid plan")
    require(all(isinstance(x, str) and 0 < len(x) <= 400 for x in plan.values()), "Invalid plan text")


def validate_reflection(value):
    require(set(value) == {"summary", "lessons"}, "Invalid reflection fields")
    require(isinstance(value["summary"], str) and len(value["summary"]) <= 700, "Invalid summary")
    require(isinstance(value["lessons"], list) and len(value["lessons"]) <= 2, "At most two lessons per card")
    for item in value["lessons"]:
        require(set(item) == {"type", "description", "conditions", "action", "tool_weight_advice",
                              "scale_advice", "segment_advice", "limitation"}, "Invalid lesson fields")
        require(item["type"] in ("guiding", "cautionary"), "Invalid principle type")
        require(strings(item["conditions"], 4) and item["conditions"], "Need applicability conditions")
        for field in ("description", "action", "limitation"):
            require(isinstance(item[field], str) and 0 < len(item[field]) <= 500, "Invalid principle text")
        advice = item["tool_weight_advice"]
        require(isinstance(advice, dict) and set(advice) == set(TOOLS), "Need advice for all four tools")
        require(all(value in ("increase", "decrease", "hold", "unknown") for value in advice.values()), "Invalid tool weight advice")
        require(item["scale_advice"] in ("up", "down", "hold", "unknown"), "Invalid scale advice")
        require(isinstance(item["segment_advice"], list) and len(item["segment_advice"]) == 4 and
                all(value in ("up", "down", "hold", "unknown") for value in item["segment_advice"]),
                "Invalid segment advice")
