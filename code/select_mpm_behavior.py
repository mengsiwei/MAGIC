#!/usr/bin/env python3
"""
select_mpm_behavior.py
======================

Adaptive MPM behavior/scenario selector for the PhyMAGIC pipeline.

Given:
  • physics.yaml  — VLM-inferred physical properties (from infer_physics.py)
  • text instruction — user's natural-language motion description

Outputs:
  • material_key  — one of the 10 keys in MATERIALS (elastic/jelly/rigid/metal/
                    sand/foam/snow/plasticine/non_newtonian/fluid)
  • behavior_key  — one of the 7 single-object behaviors (free_fall/side_push/
                    compress/spin/gravity_push/stretch/top_push) OR a
                    multi-material scenario name
  • scenario_type — "single" | "multi"
  • probe_sequence — ordered list of probe behaviors for confidence-guided
                    iterative refinement (from PhyMAGIC paper Sec. 3.2)

Behavior selection strategy (hybrid rule-based + optional GPT fallback)
------------------------------------------------------------------------
  1. Keyword matching on the instruction text (fast, no API cost)
  2. Material-behavior compatibility check (e.g. sand+spin is nonsensical)
  3. Multi-material hint detection (two-object or substrate keywords)
  4. GPT-4o-mini disambiguation if confidence is too low (--use_gpt flag)

Confidence-guided probe sequence (PhyMAGIC paper, γ=0.6)
---------------------------------------------------------
  Different low-confidence parameters map to different probe motions:
    mass / density              → free_fall   (trajectory reveals inertia)
    E / youngsModulus           → compress    (stiffness visible at impact)
    frictionAngle               → gravity_push (slope reveals friction)
    plastic_viscosity           → top_push    (viscous flow rate visible)
    material (classification)   → free_fall + compress
    poissonsRatio / yieldStress → compress

Usage
-----
  python select_mpm_behavior.py \\
      --physics_yaml data/yellowcar/physics.yaml \\
      --instruction  "The car slides sideways after a collision"

  python select_mpm_behavior.py \\
      --physics_yaml data/jelly/physics.yaml \\
      --instruction  "The jelly bounces after being dropped" \\
      --use_gpt --apikey configs/openai_apikey

  # Importable API:
  from select_mpm_behavior import select_behavior
  result = select_behavior(physics_yaml_path, instruction)
  # result = {material_key, behavior_key, scenario_type, probe_sequence, reason}
"""

import argparse
import json
import re
import sys
from pathlib import Path
from ruamel.yaml import YAML

# ── Confidence threshold (PhyMAGIC paper γ = 0.6) ──────────────────────────────
CONFIDENCE_THRESHOLD = 0.6

# ── Material name → material_key mapping ───────────────────────────────────────
# Maps VLM-inferred material strings → the 10 canonical scenario material keys.
_MATERIAL_KEY_MAP = {
    # Elastic family
    "elastic":          "elastic",
    "rubber":           "elastic",
    "silicone":         "elastic",
    "bouncy":           "elastic",
    "jelly":            "jelly",
    "soft":             "jelly",
    "gel":              "jelly",
    "putty":            "jelly",

    # Rigid / metal family
    "rigid":            "rigid",
    "hard":             "rigid",
    "glass":            "rigid",
    "ceramic":          "rigid",
    "wood":             "rigid",
    "plastic":          "rigid",
    "metal":            "metal",
    "steel":            "metal",
    "iron":             "metal",
    "aluminum":         "metal",
    "copper":           "metal",
    "tin":              "metal",

    # Granular
    "sand":             "sand",
    "soil":             "sand",
    "gravel":           "sand",
    "granular":         "sand",
    "powder":           "sand",
    "dirt":             "sand",
    "grain":            "sand",

    # Compressible / crushable
    "foam":             "foam",
    "sponge":           "foam",
    "cushion":          "foam",
    "snow":             "snow",
    "ice":              "snow",
    "frozen":           "snow",

    # Perfectly plastic
    "plasticine":       "plasticine",
    "clay":             "plasticine",
    "dough":            "plasticine",
    "wax":              "plasticine",

    # Viscoplastic / fluid
    "non_newtonian":    "non_newtonian",
    "non-newtonian":    "non_newtonian",
    "mud":              "non_newtonian",
    "slime":            "non_newtonian",
    "paste":            "non_newtonian",
    "viscous":          "non_newtonian",
    "fluid":            "fluid",
    "liquid":           "fluid",
    "water":            "fluid",
    "oil":              "fluid",
    "juice":            "fluid",
    "milk":             "fluid",
    "blood":            "fluid",
}

# ── Behavior keyword triggers ───────────────────────────────────────────────────
# Each behavior maps to a list of (pattern, weight) tuples.
# Patterns are compiled regex fragments matched against the lowercased instruction.
# Weight: higher = stronger signal.
_BEHAVIOR_PATTERNS = {
    "free_fall": [
        (r"\bfall(s|ing)?\b",           2.0),
        (r"\bdrop(s|ping)?\b",          2.0),
        (r"\bplummet",                  2.0),
        (r"\bbounce(s|d|ing)?\b",       1.5),
        (r"\bgravity\b",                1.0),
        (r"\blands?\b",                 1.0),
        (r"\bcrash(es|ed|ing)? down\b", 1.5),
        (r"\bthrown?\b",                0.5),
    ],
    "side_push": [
        (r"\bpush(es|ed|ing)?\b",       1.5),
        (r"\bslide(s|d|ing)?\b",        2.0),
        (r"\bkick(s|ed|ing)?\b",        2.0),
        (r"\bshove(s|d|ing)?\b",        2.0),
        (r"\bnudge(s|d|ing)?\b",        1.5),
        (r"\blateral\b",                1.5),
        (r"\bsideways?\b",              2.0),
        (r"\bhorizontal\b",             1.0),
        (r"\bmove(s|d|ing)? (left|right)\b", 2.0),
        (r"\bcollision\b",              1.0),
        (r"\bslam(s|med|ming)?\b",      1.5),
    ],
    "compress": [
        (r"\bsqueeze(s|d|ing)?\b",      2.0),
        (r"\bpress(es|ed|ing)?\b",      1.5),
        (r"\bcompress(es|ed|ion|ing)?\b", 2.5),
        (r"\bflatten(s|ed|ing)?\b",     1.5),
        (r"\bsquish(es|ed|ing)?\b",     2.0),
        (r"\bcrush(es|ed|ing)?\b",      1.5),
        (r"\bpunch(es|ed|ing)?\b",      1.0),
        (r"\bimpact\b",                 1.0),
    ],
    "spin": [
        (r"\bspin(s|ning)?\b",          3.0),
        (r"\brotate(s|d|ing)?\b",       3.0),
        (r"\btwirl(s|ed|ing)?\b",       2.5),
        (r"\btwist(s|ed|ing)?\b",       2.0),
        (r"\bwhirl(s|ed|ing)?\b",       2.0),
        (r"\bspiral(s|ed|ing)?\b",      1.5),
        (r"\borbits?\b",                1.5),
        (r"\bangular\b",                1.0),
    ],
    "gravity_push": [
        (r"\bslope(s|d|ing)?\b",        3.0),
        (r"\bincline(s|d|ing)?\b",      3.0),
        (r"\bslide(s|d|ing)? down\b",   2.5),
        (r"\broll(s|ed|ing)? down\b",   2.0),
        (r"\bhill\b",                   2.0),
        (r"\bramp\b",                   2.0),
        (r"\btilted? gravity\b",        2.5),
        (r"\bdownhill\b",               2.0),
    ],
    "stretch": [
        (r"\bstretch(es|ed|ing)?\b",    3.0),
        (r"\bpull(s|ed|ing)?\b",        2.0),
        (r"\belongate(s|d|ing)?\b",     2.5),
        (r"\bextend(s|ed|ing)?\b",      1.5),
        (r"\btug(s|ged|ging)?\b",       2.0),
        (r"\bstrain(s|ed|ing)?\b",      1.5),
        (r"\btension\b",                1.5),
    ],
    "top_push": [
        (r"\bstomp(s|ed|ing)?\b",                   2.0),
        (r"\btop.?push\b",                           3.0),
        (r"\bpancake(s|d|ing)?\b",                   2.0),
        (r"\bpressed? (from (the )?)?top\b",         2.5),
        (r"\bfrom (the )?top\b",                     2.0),
        (r"\bfrom above\b",                          2.0),
        (r"\bpressed? (from (the )?)?above\b",       2.0),
        (r"\bpress(es|ed|ing)? (down|flat)\b",       2.0),
        (r"\bflatten(ed)? (from|on) (the )?(top|above)\b", 2.5),
        (r"\bflat(ten)?(s|ed|ing)?\b",               1.0),
        (r"\bslow(ly)? compre",                      2.0),
        (r"\bsustained press\b",                     2.0),
    ],
}

# ── Multi-material trigger keywords ────────────────────────────────────────────
_MULTI_PATTERNS = {
    "multi_elastic_collision": [
        (r"\bcollide(s|d|ing)?\b",      2.0),
        (r"\bhead.?on\b",               2.5),
        (r"\btwo .*(ball|cube|block)",  2.0),
        (r"\bcrash(es|ed|ing)? into\b", 2.0),
        (r"\belastic collision\b",      3.0),
        (r"\bbounce off each other\b",  3.0),
    ],
    "multi_water_elastic": [
        (r"\b(drop|fall|plunge) into water\b", 3.0),
        (r"\bfloat(s|ing)? on (water|fluid|liquid)\b", 3.0),
        (r"\bobject.*water\b",          2.0),
        (r"\bwater.*elastic\b",         2.0),
    ],
    "multi_water_rigid": [
        (r"\bsink(s|ing)? into (water|fluid|liquid)\b", 3.0),
        (r"\bdrop(s|ping)? into water\b", 2.0),
        (r"\bsubmerge(s|d|ing)?\b",     2.5),
        (r"\bheavy.*water\b",           2.0),
    ],
    "multi_sand_rigid": [
        (r"\bfall(s|ing)? onto sand\b",     3.0),
        (r"\bdrop(s|ping)? on(to)? sand\b", 3.0),
        (r"\bsand.*impact\b",               2.0),
        (r"\bland(s|ing)? on sand\b",        3.0),
        (r"\bon(to)? sand\b",               2.0),
        (r"\binto (the )?sand\b",           2.5),
    ],
}

# ── Material–behavior compatibility ────────────────────────────────────────────
# For each material, lists PREFERRED behaviors (order = priority) and
# FORBIDDEN behaviors (physically nonsensical or visually boring).
_MATERIAL_BEHAVIOR_COMPAT = {
    "elastic":      {"preferred": ["free_fall","compress","stretch","spin","side_push"],
                     "forbidden": []},
    "jelly":        {"preferred": ["free_fall","compress","stretch","spin","side_push"],
                     "forbidden": []},
    "rigid":        {"preferred": ["free_fall","spin","side_push","compress"],
                     "forbidden": ["stretch"]},          # rigid can't stretch
    "metal":        {"preferred": ["free_fall","compress","stretch","spin"],
                     "forbidden": []},
    "sand":         {"preferred": ["free_fall","side_push","gravity_push","compress"],
                     "forbidden": ["spin","stretch","top_push"]},  # granular can't spin/stretch
    "foam":         {"preferred": ["free_fall","compress","top_push","side_push"],
                     "forbidden": ["stretch","spin"]},
    "snow":         {"preferred": ["free_fall","compress","spin","side_push"],
                     "forbidden": ["stretch"]},
    "plasticine":   {"preferred": ["free_fall","compress","stretch","top_push"],
                     "forbidden": ["spin"]},
    "non_newtonian":{"preferred": ["free_fall","compress","gravity_push","top_push"],
                     "forbidden": ["spin","stretch"]},
    "fluid":        {"preferred": ["free_fall","compress","side_push","top_push"],
                     "forbidden": ["spin","stretch"]},
}

# ── Probe sequence table ────────────────────────────────────────────────────────
# Maps low-confidence parameter names → recommended probe behavior.
# (PhyMAGIC paper Sec. 3.2: iterative confidence-guided refinement)
PROBE_FOR_PARAM = {
    "mass":             "free_fall",
    "density":          "free_fall",
    "material":         "free_fall",          # probe 1: fall for basic classification
    "youngsModulus":    "compress",
    "E":                "compress",
    "young_modulus":    "compress",
    "poissonsRatio":    "compress",
    "nu":               "compress",
    "yieldStress":      "compress",
    "yield_stress":     "compress",
    "frictionAngle":    "gravity_push",
    "friction_angle":   "gravity_push",
    "plasticViscosity": "top_push",
    "plastic_viscosity":"top_push",
    "fluidViscosity":   "top_push",
    "bulkModulus":      "compress",
    "shearModulus":     "compress",
    "externalForce":    "side_push",
    "external_force":   "side_push",
    "initialVelocity":  "side_push",
    "initial_velocity": "side_push",
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _score_patterns(text, pattern_dict):
    """Return {key: score} by summing regex-match weights."""
    text_lower = text.lower()
    scores = {}
    for key, patterns in pattern_dict.items():
        s = 0.0
        for pat, weight in patterns:
            if re.search(pat, text_lower):
                s += weight
        scores[key] = s
    return scores


def _load_physics(yaml_path):
    """Load physics.yaml and return first object dict."""
    yp = YAML()
    with open(yaml_path, "r") as f:
        data = yp.load(f)
    if isinstance(data, list):
        return data[0]
    return data


def _get_material_key(physics):
    """Map VLM-inferred material string → canonical material_key."""
    llm_mat = str(physics.get("material", "elastic")).lower().strip()
    # exact lookup
    if llm_mat in _MATERIAL_KEY_MAP:
        return _MATERIAL_KEY_MAP[llm_mat]
    # partial match
    for alias, key in _MATERIAL_KEY_MAP.items():
        if alias in llm_mat or llm_mat in alias:
            return key
    return "jelly"  # safe fallback


def _get_low_confidence_params(physics, threshold=CONFIDENCE_THRESHOLD):
    """Return list of parameter names with confidence below threshold."""
    low = []
    for k, v in physics.items():
        if k.endswith("_confidence") and isinstance(v, (int, float)) and v < threshold:
            param = k[: -len("_confidence")]
            low.append(param)
    return low


def _build_probe_sequence(low_params, material_key):
    """
    Build an ordered probe sequence from low-confidence parameters.
    Returns a deduplicated list respecting material compatibility.
    """
    forbidden = _MATERIAL_BEHAVIOR_COMPAT.get(material_key, {}).get("forbidden", [])
    seen = set()
    probes = []
    for param in low_params:
        probe = PROBE_FOR_PARAM.get(param)
        if probe and probe not in seen and probe not in forbidden:
            probes.append(probe)
            seen.add(probe)
    # Always probe material classification if not already covered
    if not probes:
        fallback = "free_fall"
        if fallback not in forbidden:
            probes.append(fallback)
    return probes


def _gpt_select_behavior(instruction, material_key, physics, apikey_path):
    """
    Ask GPT-4o-mini to select the best behavior when rule-based scoring is
    ambiguous (top score < 2.0 or top-1 and top-2 differ by < 0.5).
    Returns (behavior_key, reason_str).
    """
    try:
        import openai
        with open(apikey_path) as f:
            openai.api_key = f.read().strip()

        behaviors = list(_BEHAVIOR_PATTERNS.keys())
        forbidden = _MATERIAL_BEHAVIOR_COMPAT.get(material_key, {}).get("forbidden", [])
        allowed = [b for b in behaviors if b not in forbidden]

        prompt = (
            f"You are a physics simulation expert. "
            f"A {material_key} object is described by the following instruction:\n\n"
            f'  "{instruction}"\n\n'
            f"Choose the single most appropriate MPM simulation behavior from this list:\n"
            f"  {allowed}\n\n"
            f"Behavior descriptions:\n"
            f"  free_fall   – object drops under gravity, impacts floor\n"
            f"  side_push   – lateral impulse pushes object sideways\n"
            f"  compress    – downward impulse compresses object from above\n"
            f"  spin        – object rotates, then is released\n"
            f"  gravity_push – tilted gravity (slope effect) causes lateral sliding\n"
            f"  stretch     – bottom is pinned, top is pulled upward\n"
            f"  top_push    – bottom is pinned, top is slowly pressed down\n\n"
            f"Reply with exactly one JSON object: "
            f'{{\"behavior\": \"<name>\", \"reason\": \"<one sentence>\"}}'
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        text = resp["choices"][0]["message"]["content"]
        cleaned = re.sub(r"```json|```", "", text).strip()
        result = json.loads(cleaned)
        return result.get("behavior", allowed[0]), result.get("reason", "GPT selection")
    except Exception as e:
        print(f"[select_behavior] GPT fallback failed: {e}. Using free_fall.")
        return "free_fall", f"GPT error: {e}"


# ── Public API ──────────────────────────────────────────────────────────────────

def select_behavior(
    physics_yaml_path,
    instruction,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    use_gpt=False,
    apikey_path="configs/openai_apikey",
):
    """
    Main entry point.

    Parameters
    ----------
    physics_yaml_path : str
        Path to physics.yaml produced by infer_physics.py.
    instruction : str
        Natural-language motion description (e.g. "The car slides sideways").
    confidence_threshold : float
        Below this value a parameter is considered low-confidence (default 0.6).
    use_gpt : bool
        Fall back to GPT-4o-mini when keyword scores are ambiguous.
    apikey_path : str
        Path to OpenAI API key file (used only when use_gpt=True).

    Returns
    -------
    dict with keys:
        material_key   : str   – one of 10 canonical material names
        behavior_key   : str   – one of 7 behaviors OR a multi-material scenario
        scenario_type  : str   – "single" | "multi"
        probe_sequence : list  – ordered probe behaviors for iterative refinement
        low_conf_params: list  – parameter names below confidence threshold
        reason         : str   – human-readable selection rationale
    """
    physics = _load_physics(physics_yaml_path)
    material_key = _get_material_key(physics)
    low_conf = _get_low_confidence_params(physics, confidence_threshold)
    probe_seq = _build_probe_sequence(low_conf, material_key)

    # ── 1. Check for multi-material scenario hints ────────────────────────────
    multi_scores = _score_patterns(instruction, _MULTI_PATTERNS)
    best_multi = max(multi_scores, key=multi_scores.get)
    if multi_scores[best_multi] >= 3.0:
        # Strong multi-material signal
        return {
            "material_key":    material_key,
            "behavior_key":    best_multi,
            "scenario_type":   "multi",
            "probe_sequence":  probe_seq,
            "low_conf_params": low_conf,
            "reason": (
                f"Multi-material scenario '{best_multi}' detected "
                f"(score={multi_scores[best_multi]:.1f})"
            ),
        }

    # ── 2. Score single-object behaviors ──────────────────────────────────────
    behavior_scores = _score_patterns(instruction, _BEHAVIOR_PATTERNS)

    # Zero out forbidden behaviors for this material
    forbidden = _MATERIAL_BEHAVIOR_COMPAT.get(material_key, {}).get("forbidden", [])
    for fb in forbidden:
        behavior_scores[fb] = -1.0

    sorted_behaviors = sorted(behavior_scores.items(), key=lambda x: -x[1])
    best_behavior, best_score = sorted_behaviors[0]
    second_score = sorted_behaviors[1][1] if len(sorted_behaviors) > 1 else 0.0

    # ── 3. GPT fallback if ambiguous ──────────────────────────────────────────
    reason = f"Keyword match: score={best_score:.1f}"
    if use_gpt and (best_score < 2.0 or (best_score - second_score) < 0.5):
        gpt_behavior, gpt_reason = _gpt_select_behavior(
            instruction, material_key, physics, apikey_path
        )
        best_behavior = gpt_behavior
        reason = f"GPT: {gpt_reason}"
    elif best_score <= 0.0:
        # No keyword match at all → use material's top preferred behavior
        preferred = _MATERIAL_BEHAVIOR_COMPAT.get(material_key, {}).get(
            "preferred", ["free_fall"]
        )
        best_behavior = preferred[0]
        reason = f"No keyword match; default for {material_key}"

    return {
        "material_key":    material_key,
        "behavior_key":    best_behavior,
        "scenario_type":   "single",
        "probe_sequence":  probe_seq,
        "low_conf_params": low_conf,
        "reason":          reason,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Select MPM behavior/scenario from physics.yaml + instruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--physics_yaml", required=True,
        help="Path to physics.yaml (output of infer_physics.py)",
    )
    parser.add_argument(
        "--instruction", required=True,
        help="Natural-language motion description",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=CONFIDENCE_THRESHOLD,
        help=f"Confidence below this triggers probe (default {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--use_gpt", action="store_true",
        help="Use GPT-4o-mini for ambiguous cases",
    )
    parser.add_argument(
        "--apikey", default="configs/openai_apikey",
        help="Path to OpenAI API key file",
    )
    parser.add_argument(
        "--output_json", default=None,
        help="Write result to this JSON file",
    )
    args = parser.parse_args()

    result = select_behavior(
        physics_yaml_path=args.physics_yaml,
        instruction=args.instruction,
        confidence_threshold=args.confidence_threshold,
        use_gpt=args.use_gpt,
        apikey_path=args.apikey,
    )

    print(json.dumps(result, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[select_behavior] Saved to {args.output_json}")


if __name__ == "__main__":
    main()
