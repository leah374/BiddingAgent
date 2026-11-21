"""Immutable four-step cards plus a versioned, advertiser-scoped principle index."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def append_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def assess_window(rows, cpa_constraint):
    """Descriptive local assessment, never a causal improvement claim."""
    cost = sum(row["outcome"]["cost"] for row in rows)
    conversions = sum(row["outcome"]["conversions"] for row in rows)
    expected = sum(row["outcome"]["expected_conversions"] for row in rows)
    cpa = cost / conversions if conversions else None
    expected_cpa = cost / expected if expected else None
    if conversions > 0 and cpa <= cpa_constraint and expected_cpa is not None and expected_cpa <= cpa_constraint:
        label = "positive"
    elif expected >= 1 and expected_cpa > 1.2 * cpa_constraint:
        label = "negative"
    elif cost == 0 or expected < 1:
        label = "uncertain"
    else:
        label = "mixed"
    return dict(assessment=label, cost=cost, conversions=conversions,
                expected_conversions=expected, cpa=cpa, expected_cpa=expected_cpa,
                wins=sum(row["outcome"]["wins"] for row in rows),
                score_before=rows[0]["state"]["score"], score_after=rows[-1]["outcome"]["score"],
                causal_status="observational_not_a_baseline_comparison")


class ExperienceStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "principles.json"
        self.principles = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []

    def snapshot(self, advertiser, cycle, environment):
        return copy.deepcopy([p for p in self.principles if p["advertiser_id"] == advertiser
                              and p["available_from_cycle"] <= cycle
                              and p["environment"] == environment and not p.get("archived", False)])

    @staticmethod
    def retrieve(snapshot, state, top_k=3):
        def rank(principle):
            context = principle["context"]
            phase = abs(context["step"] - state["step"]) / 48
            budget = abs(context["remaining_budget_ratio"] - state["remaining_budget_ratio"])
            quality = (principle["successful_windows"] + 1) / (principle["evaluated_windows"] + 2)
            return 1 - phase - 0.25 * budget + 0.1 * quality
        return sorted(copy.deepcopy(snapshot), key=rank, reverse=True)[:top_k]

    def record_card(self, card):
        path = self.directory / "cards" / (card["card_id"] + ".json")
        if path.exists():
            raise ValueError("Duplicate card: refuse to overwrite completed experience")
        write_json(path, card)
        # Only applied principles get outcome credit. Mixed/uncertain windows do
        # not become automatic failures. Counts refer to windows, not conversions.
        for principle in self.principles:
            pid = principle["principle_id"]
            principle["retrieval_count"] += sum(pid in ids for ids in card["retrieved_by_step"])
            principle["applied_count"] += sum(pid in ids for ids in card["applied_by_step"])
            if any(pid in ids for ids in card["applied_by_step"]):
                label = card["outcome"]["assessment"]
                if label in ("positive", "negative"):
                    principle["evaluated_windows"] += 1
                    principle["successful_windows"] += int(label == "positive")
                principle["reuse_evidence"].append(card["card_id"])
        write_json(self.path, self.principles)

    def consolidate(self, cards, judge=None):
        """Retain every card; cautiously merge equivalent principles, not cases."""
        from difflib import SequenceMatcher
        for card in cards:
            for index, lesson in enumerate(card["lessons"]):
                candidates = [p for p in self.principles if
                              p["environment"] == card["environment"] and
                              p["advertiser_id"] == card["advertiser_id"] and
                              p["type"] == lesson["type"]]
                same = None
                for candidate in candidates:
                    if candidate["description"] == lesson["description"] and candidate["conditions"] == lesson["conditions"]:
                        same = candidate
                        break
                if same is None and candidates and judge:
                    candidate = max(candidates, key=lambda p: SequenceMatcher(None, p["description"], lesson["description"]).ratio())
                    similarity = SequenceMatcher(None, candidate["description"], lesson["description"]).ratio()
                    if similarity >= 0.45 and judge(candidate, lesson):
                        same = candidate
                if same is not None:
                    same["source_cards"].append(card["card_id"])
                    # Source outcomes are evidence; they are NOT reuse failures.
                    continue
                principle = dict(lesson)
                principle.update(principle_id="p_" + fingerprint([card["card_id"], index])[:16],
                                 advertiser_id=card["advertiser_id"], environment=card["environment"],
                                 available_from_cycle=card["cycle_id"] + 1,
                                 context=card["context"], source_cards=[card["card_id"]],
                                 retrieval_count=0, applied_count=0, evaluated_windows=0,
                                 successful_windows=0, reuse_evidence=[], evidence_strength="limited",
                                 archived=False)
                self.principles.append(principle)
        write_json(self.path, self.principles)

    def finish_cycle(self, cycle, advertiser, summary):
        write_json(self.directory / ("cycle_%s_adv_%s.json" % (cycle, advertiser)), summary)
