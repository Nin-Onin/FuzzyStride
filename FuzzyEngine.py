import numpy as np
import skfuzzy as fuzz
from dataclasses import dataclass, field
from typing import Dict, List

from FuzzyModel import FuzzyModel


@dataclass
class EvaluationResult:
    heart_rate:      float
    pacing:          float
    distance:        float
    activated_rules: List[str]
    agg_values:      Dict[str, np.ndarray]
    aggregated_mf:   np.ndarray
    crisp_value:     float
    status:          str
    recommendation:  str
    membership_degrees: Dict[str, float] = field(default_factory=dict)


class FuzzyEngine:
    ADVICE_MAP = {
        "Undertraining": "Increase intensity or distance gradually.",
        "Normal":        "Maintain current training regimen.",
        "Overtraining":  "Rest or reduce intensity to prevent injury.",
    }

    def __init__(self, model: FuzzyModel):
        self.model = model


    def evaluate(self, hr: float, pc: float, dist: float) -> EvaluationResult:
        m      = self.model
        ts_uni = m.training_status.universe

        agg_values = {
            "undertraining": np.zeros_like(ts_uni),
            "normal":        np.zeros_like(ts_uni),
            "overtraining":  np.zeros_like(ts_uni),
        }

        activated_rules: List[str] = []

        for idx, (dist_term, pace_term, hr_term, out_term) in \
                enumerate(m.rule_definitions, start=1):

            deg_dist = fuzz.interp_membership(
                m.distance.universe, m.distance[dist_term].mf, dist)
            deg_pace = fuzz.interp_membership(
                m.pacing.universe, m.pacing[pace_term].mf, pc)
            deg_hr   = fuzz.interp_membership(
                m.heart_rate.universe, m.heart_rate[hr_term].mf, hr)

            firing_strength = np.fmin(np.fmin(deg_dist, deg_pace), deg_hr)

            if firing_strength > 0:
                activated_rules.append(
                    f"Rule {idx}: IF Distance is {dist_term.capitalize()} "
                    f"AND Pacing is {pace_term.capitalize()} "
                    f"AND Heart Rate is {hr_term.capitalize()} "
                    f"THEN Training Status is {out_term.capitalize()}"
                )

            clipped = np.fmin(firing_strength, m.training_status[out_term].mf)
            agg_values[out_term] = np.fmax(agg_values[out_term], clipped)

        aggregated_mf = np.fmax(
            agg_values['undertraining'],
            np.fmax(agg_values['normal'], agg_values['overtraining'])
        )

        crisp_value = fuzz.defuzz(ts_uni, aggregated_mf, 'centroid')

        membership_degrees = self._membership_degrees(crisp_value)
        status             = max(membership_degrees, key=membership_degrees.get)
        recommendation     = self.ADVICE_MAP[status]

        return EvaluationResult(
            heart_rate      = hr,
            pacing          = pc,
            distance        = dist,
            activated_rules = activated_rules,
            agg_values      = agg_values,
            aggregated_mf   = aggregated_mf,
            crisp_value     = crisp_value,
            status          = status,
            recommendation  = recommendation,
            membership_degrees = membership_degrees,
        )


    def _membership_degrees(self, crisp: float) -> Dict[str, float]:
        ts = self.model.training_status
        return {
            "Undertraining": fuzz.interp_membership(
                ts.universe, ts['undertraining'].mf, crisp),
            "Normal":        fuzz.interp_membership(
                ts.universe, ts['normal'].mf,        crisp),
            "Overtraining":  fuzz.interp_membership(
                ts.universe, ts['overtraining'].mf,  crisp),
        }


def validate_input(value_str: str, min_val: float,
                   max_val: float, param_name: str) -> float:
  
    if not value_str.strip():
        raise ValueError(f"{param_name} cannot be empty.")
    try:
        value = float(value_str.strip())
    except ValueError:
        raise ValueError(f"{param_name} must be a number (integer or decimal).")
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{param_name} must be between {min_val} and {max_val}.")
    return value