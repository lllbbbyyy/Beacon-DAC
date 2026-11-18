"""Configuration coverage, without attributing joint outcomes to one parameter."""

import math


def summarize_parameter_coverage(records, domains, base_hardware):
    records = list(records)
    for row in records:
        if not math.isfinite(row["objective"]) or row["objective"] <= 0:
            raise ValueError("Coverage requires positive finite objectives")
    result = {}
    for parameter, candidates in domains.items():
        base_value = base_hardware.get(parameter)
        other_base = {k: v for k, v in base_hardware.items() if k != parameter}
        values = []
        for value in candidates:
            matches = [r for r in records if r["hardware"].get(parameter) == value]
            controlled = [r for r in matches if
                          {k: v for k, v in r["hardware"].items() if k != parameter} == other_base]
            best = min(matches, key=lambda r: r["objective"]) if matches else None
            exact = min(controlled, key=lambda r: r["objective"]) if controlled else None
            values.append({
                "value": value,
                "evaluations": len(matches),
                "best_observed": None if best is None else {
                    "iteration": best["iteration"], "objective": best["objective"]},
                "same_other_parameters": None if exact is None else {
                    "iteration": exact["iteration"], "objective": exact["objective"]},
            })
        result[parameter] = {"base_value": base_value, "values": values}
    return {
        "interpretation": (
            "Best observed values may come from different configurations and are not "
            "single-parameter effects. same_other_parameters holds all other hardware "
            "fields equal to the selected base. Zero evaluations means untested, not "
            "ineffective. A failed increase does not establish that a decrease fails."
        ),
        "parameters": result,
    }
