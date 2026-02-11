#!/usr/bin/env python3
"""
Statistical analysis for ALME audio-text conflict experiments.

Provides paired comparison of natural vs TTS evaluation results using
McNemar's test, Wilson score CIs, Cohen's h effect sizes, and
chi-squared tests for bucket/flip-type independence.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Data loading & pairing
# ---------------------------------------------------------------------------

def load_trial_results(path: str) -> List[Dict]:
    """Load per-trial JSON results array."""
    with open(path) as f:
        return json.load(f)


def load_stimuli_metadata(jsonl_path: str) -> Dict[str, Dict]:
    """Load stimulus JSONL, return dict keyed by stimulus_id."""
    lookup = {}
    with open(jsonl_path) as f:
        for line in f:
            stim = json.loads(line)
            lookup[stim["stimulus_id"]] = stim
    return lookup


def _base_stimulus_id(sid: str) -> str:
    """Strip tts_ prefix to get the base stimulus id for pairing."""
    if sid.startswith("tts_"):
        return sid[4:]
    return sid


def _get_language(stim_meta: Dict) -> str:
    """Get language from stimulus metadata.

    Checks 'language' field first, falls back to 'accent_bucket' for old data,
    stripping TTS_ prefix if present.
    """
    lang = stim_meta.get("language")
    if lang:
        if lang.startswith("TTS_"):
            return lang[4:]
        return lang
    # Fallback to accent_bucket for old-format data
    bucket = stim_meta.get("accent_bucket", "UNKNOWN")
    if bucket.startswith("TTS_"):
        return bucket[4:]
    return bucket


# Backward-compat alias
_original_bucket = _get_language


def pair_conflict_trials(
    natural_trials: List[Dict],
    tts_trials: List[Dict],
) -> List[Dict]:
    """Pair natural and TTS conflict trials by base stimulus_id.

    Returns list of dicts with keys:
        base_id, natural, tts
    where natural/tts are the trial dicts for audio_text_conflict condition.
    """
    nat_conflict = {
        t["stimulus_id"]: t
        for t in natural_trials
        if t["condition"] == "audio_text_conflict"
    }
    tts_conflict = {
        _base_stimulus_id(t["stimulus_id"]): t
        for t in tts_trials
        if t["condition"] == "audio_text_conflict"
    }

    pairs = []
    for base_id in sorted(nat_conflict.keys()):
        if base_id in tts_conflict:
            pairs.append({
                "base_id": base_id,
                "natural": nat_conflict[base_id],
                "tts": tts_conflict[base_id],
            })
    return pairs


def pair_audio_only_trials(
    natural_trials: List[Dict],
    tts_trials: List[Dict],
) -> List[Dict]:
    """Pair natural and TTS audio_only trials by base stimulus_id."""
    nat_ao = {
        t["stimulus_id"]: t
        for t in natural_trials
        if t["condition"] == "audio_only"
    }
    tts_ao = {
        _base_stimulus_id(t["stimulus_id"]): t
        for t in tts_trials
        if t["condition"] == "audio_only"
    }

    pairs = []
    for base_id in sorted(nat_ao.keys()):
        if base_id in tts_ao:
            pairs.append({
                "base_id": base_id,
                "natural": nat_ao[base_id],
                "tts": tts_ao[base_id],
            })
    return pairs


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def mcnemar_test(table: np.ndarray) -> Dict[str, Any]:
    """McNemar's test for paired nominal data.

    Args:
        table: 2x2 contingency table [[a, b], [c, d]] where
            a = both correct, b = nat correct / tts incorrect,
            c = nat incorrect / tts correct, d = both incorrect

    Uses exact binomial test when b + c < 25, otherwise chi-squared
    with continuity correction.
    """
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    n_discordant = b + c

    if n_discordant == 0:
        return {
            "statistic": None,
            "p_value": None,
            "method": "not_applicable",
            "b": int(b),
            "c": int(c),
            "n_discordant": 0,
            "note": "No discordant pairs; McNemar test not applicable",
        }

    if n_discordant < 25:
        # Exact binomial test: H0 is b/(b+c) = 0.5
        result = sp_stats.binomtest(int(b), int(n_discordant), 0.5)
        return {
            "statistic": None,
            "p_value": float(result.pvalue),
            "method": "exact_binomial",
            "b": int(b),
            "c": int(c),
            "n_discordant": int(n_discordant),
        }

    # Chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - sp_stats.chi2.cdf(chi2, df=1)
    return {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "method": "chi_squared_cc",
        "b": int(b),
        "c": int(c),
        "n_discordant": int(n_discordant),
    }


def wilson_score_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    if n == 0:
        return (0.0, 0.0)

    z = sp_stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denom = 1 + z ** 2 / n
    centre = p_hat + z ** 2 / (2 * n)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n)

    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return (max(0.0, lower), min(1.0, upper))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for difference between two proportions.

    h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))

    Magnitude thresholds:
        < 0.20  negligible
        0.20–0.49  small
        0.50–0.79  medium
        >= 0.80  large
    """
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def cohens_h_magnitude(h: float) -> str:
    """Interpret Cohen's h magnitude."""
    ah = abs(h)
    if ah < 0.20:
        return "negligible"
    elif ah < 0.50:
        return "small"
    elif ah < 0.80:
        return "medium"
    else:
        return "large"


def chi_squared_independence(observed: np.ndarray) -> Dict[str, Any]:
    """Chi-squared test of independence with Cramér's V.

    Falls back to Fisher's exact test when any expected frequency < 5.

    Args:
        observed: r×c contingency table (numpy 2D array)
    """
    # Check expected frequencies
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()

    if total == 0:
        return {
            "statistic": None,
            "p_value": None,
            "method": "not_applicable",
            "cramers_v": None,
            "note": "Empty table",
        }

    expected = np.outer(row_sums, col_sums) / total
    min_expected = expected.min()

    # For 2x2 tables with small expected, use Fisher's exact
    if min_expected < 5 and observed.shape == (2, 2):
        _, p_value = sp_stats.fisher_exact(observed)
        return {
            "statistic": None,
            "p_value": float(p_value),
            "method": "fisher_exact",
            "min_expected": float(min_expected),
            "cramers_v": None,
            "note": "Fisher's exact test used due to small expected frequencies",
        }

    # For larger tables with small expected, note the issue
    use_fisher_note = None
    if min_expected < 5:
        use_fisher_note = (
            f"Warning: minimum expected frequency = {min_expected:.1f} < 5. "
            "Chi-squared approximation may be unreliable."
        )

    chi2, p_value, dof, _ = sp_stats.chi2_contingency(observed, correction=False)

    # Cramér's V
    k = min(observed.shape)
    cramers_v = math.sqrt(chi2 / (total * (k - 1))) if total > 0 and k > 1 else 0.0

    result = {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "method": "chi_squared",
        "min_expected": float(min_expected),
        "cramers_v": float(cramers_v),
    }
    if use_fisher_note:
        result["note"] = use_fisher_note
    return result


# ---------------------------------------------------------------------------
# Composite analyses
# ---------------------------------------------------------------------------

def _classify_conflict_trial(trial: Dict) -> str:
    """Classify a conflict trial as 'audio', 'text', or 'other'."""
    if trial["correct"]:
        return "audio"
    elif trial.get("followed_text"):
        return "text"
    else:
        return "other"


def _build_mcnemar_table(
    pairs: List[Dict],
    outcome_fn,
    exclude_other: bool = True,
) -> Tuple[np.ndarray, int, int]:
    """Build 2x2 McNemar table from paired trials.

    Args:
        pairs: list of {"natural": trial, "tts": trial, ...}
        outcome_fn: function(trial) -> bool (True = positive outcome)
        exclude_other: if True, exclude pairs where either side is "other"

    Returns:
        (table, n_included, n_excluded)
    """
    a = b = c = d = 0
    n_excluded = 0

    for pair in pairs:
        nat = pair["natural"]
        tts = pair["tts"]

        if exclude_other:
            nat_class = _classify_conflict_trial(nat)
            tts_class = _classify_conflict_trial(tts)
            if nat_class == "other" or tts_class == "other":
                n_excluded += 1
                continue

        nat_pos = outcome_fn(nat)
        tts_pos = outcome_fn(tts)

        if nat_pos and tts_pos:
            a += 1
        elif nat_pos and not tts_pos:
            b += 1
        elif not nat_pos and tts_pos:
            c += 1
        else:
            d += 1

    table = np.array([[a, b], [c, d]])
    n_included = a + b + c + d
    return table, n_included, n_excluded


def analyze_tdr_comparison(
    pairs: List[Dict],
    alpha: float = 0.005,
) -> Dict[str, Any]:
    """Compare TDR between natural and TTS conditions.

    Primary analysis: exclude pairs where either side is "other".
    Outcome: followed_text = True.
    """
    # "Followed text" as positive outcome for TDR
    def followed_text(trial):
        return bool(trial.get("followed_text"))

    table, n_included, n_excluded = _build_mcnemar_table(
        pairs, followed_text, exclude_other=True
    )

    mcnemar = mcnemar_test(table)

    # Compute TDR for each side
    nat_text = sum(1 for p in pairs
                   if _classify_conflict_trial(p["natural"]) == "text")
    nat_audio = sum(1 for p in pairs
                    if _classify_conflict_trial(p["natural"]) == "audio")
    nat_other = sum(1 for p in pairs
                    if _classify_conflict_trial(p["natural"]) == "other")
    tts_text = sum(1 for p in pairs
                   if _classify_conflict_trial(p["tts"]) == "text")
    tts_audio = sum(1 for p in pairs
                    if _classify_conflict_trial(p["tts"]) == "audio")
    tts_other = sum(1 for p in pairs
                    if _classify_conflict_trial(p["tts"]) == "other")

    nat_denom = nat_text + nat_audio
    tts_denom = tts_text + tts_audio
    nat_tdr = nat_text / nat_denom if nat_denom > 0 else None
    tts_tdr = tts_text / tts_denom if tts_denom > 0 else None

    nat_ci = wilson_score_interval(nat_text, nat_denom) if nat_denom > 0 else (None, None)
    tts_ci = wilson_score_interval(tts_text, tts_denom) if tts_denom > 0 else (None, None)

    h = cohens_h(tts_tdr, nat_tdr) if (nat_tdr is not None and tts_tdr is not None) else None

    return {
        "natural": {
            "tdr": nat_tdr,
            "followed_text": nat_text,
            "followed_audio": nat_audio,
            "other": nat_other,
            "n_clear": nat_denom,
            "ci_95": list(nat_ci),
        },
        "tts": {
            "tdr": tts_tdr,
            "followed_text": tts_text,
            "followed_audio": tts_audio,
            "other": tts_other,
            "n_clear": tts_denom,
            "ci_95": list(tts_ci),
        },
        "mcnemar": mcnemar,
        "cohens_h": float(h) if h is not None else None,
        "cohens_h_magnitude": cohens_h_magnitude(h) if h is not None else None,
        "n_pairs": len(pairs),
        "n_included": n_included,
        "n_excluded_other": n_excluded,
        "alpha": alpha,
        "significant": mcnemar["p_value"] < alpha if mcnemar["p_value"] is not None else None,
    }


def analyze_tdr_comparison_sensitivity(
    pairs: List[Dict],
    alpha: float = 0.005,
) -> Dict[str, Any]:
    """Sensitivity analysis: treat 'other' as 'not-text' (i.e., not followed text).

    Uses all pairs, no exclusions.
    """
    def followed_text(trial):
        return bool(trial.get("followed_text"))

    table, n_included, _ = _build_mcnemar_table(
        pairs, followed_text, exclude_other=False
    )

    mcnemar = mcnemar_test(table)

    nat_text = sum(1 for p in pairs if bool(p["natural"].get("followed_text")))
    tts_text = sum(1 for p in pairs if bool(p["tts"].get("followed_text")))
    n = len(pairs)

    nat_tdr = nat_text / n if n > 0 else None
    tts_tdr = tts_text / n if n > 0 else None

    nat_ci = wilson_score_interval(nat_text, n) if n > 0 else (None, None)
    tts_ci = wilson_score_interval(tts_text, n) if n > 0 else (None, None)

    h = cohens_h(tts_tdr, nat_tdr) if (nat_tdr is not None and tts_tdr is not None) else None

    return {
        "note": "Sensitivity: 'other' treated as not-text, all pairs included",
        "natural_tdr": nat_tdr,
        "tts_tdr": tts_tdr,
        "natural_ci_95": list(nat_ci),
        "tts_ci_95": list(tts_ci),
        "n_pairs": n,
        "mcnemar": mcnemar,
        "cohens_h": float(h) if h is not None else None,
        "cohens_h_magnitude": cohens_h_magnitude(h) if h is not None else None,
        "alpha": alpha,
        "significant": mcnemar["p_value"] < alpha if mcnemar["p_value"] is not None else None,
    }


def analyze_audio_only_comparison(
    pairs: List[Dict],
    alpha: float = 0.005,
) -> Dict[str, Any]:
    """Compare audio-only accuracy between natural and TTS."""
    def correct(trial):
        return bool(trial.get("correct"))

    table, n_included, _ = _build_mcnemar_table(
        pairs, correct, exclude_other=False
    )

    mcnemar = mcnemar_test(table)

    nat_correct = sum(1 for p in pairs if p["natural"].get("correct"))
    tts_correct = sum(1 for p in pairs if p["tts"].get("correct"))
    n = len(pairs)

    nat_acc = nat_correct / n if n > 0 else None
    tts_acc = tts_correct / n if n > 0 else None

    nat_ci = wilson_score_interval(nat_correct, n) if n > 0 else (None, None)
    tts_ci = wilson_score_interval(tts_correct, n) if n > 0 else (None, None)

    h = cohens_h(tts_acc, nat_acc) if (nat_acc is not None and tts_acc is not None) else None

    return {
        "natural_accuracy": nat_acc,
        "tts_accuracy": tts_acc,
        "natural_correct": nat_correct,
        "tts_correct": tts_correct,
        "n_pairs": n,
        "natural_ci_95": list(nat_ci),
        "tts_ci_95": list(tts_ci),
        "mcnemar": mcnemar,
        "cohens_h": float(h) if h is not None else None,
        "cohens_h_magnitude": cohens_h_magnitude(h) if h is not None else None,
        "alpha": alpha,
        "significant": mcnemar["p_value"] < alpha if mcnemar["p_value"] is not None else None,
    }


def _stratified_tdr_analysis(
    pairs: List[Dict],
    stimuli_meta: Dict[str, Dict],
    group_fn,
    alpha_omnibus: float = 0.05,
    alpha_pairwise: float = 0.005,
) -> Dict[str, Any]:
    """Generic stratified TDR analysis with omnibus + pairwise tests.

    Args:
        pairs: conflict trial pairs
        stimuli_meta: stimulus metadata lookup (keyed by natural stimulus_id)
        group_fn: function(stim_meta) -> group label
        alpha_omnibus: significance level for omnibus chi-squared
        alpha_pairwise: Bonferroni-adjusted alpha for pairwise McNemar
    """
    # Group pairs by stratum
    groups: Dict[str, List[Dict]] = {}
    for pair in pairs:
        base_id = pair["base_id"]
        meta = stimuli_meta.get(base_id, {})
        group = group_fn(meta)
        groups.setdefault(group, []).append(pair)

    # Per-group TDR (excluding "other")
    group_results = {}
    for group_name in sorted(groups.keys()):
        grp_pairs = groups[group_name]
        # Use natural side for TDR stratification
        nat_text = 0
        nat_audio = 0
        nat_other = 0
        for p in grp_pairs:
            cls = _classify_conflict_trial(p["natural"])
            if cls == "text":
                nat_text += 1
            elif cls == "audio":
                nat_audio += 1
            else:
                nat_other += 1

        nat_denom = nat_text + nat_audio
        tdr = nat_text / nat_denom if nat_denom > 0 else None
        ci = wilson_score_interval(nat_text, nat_denom) if nat_denom > 0 else (None, None)

        group_results[group_name] = {
            "n_pairs": len(grp_pairs),
            "n_clear": nat_denom,
            "followed_text": nat_text,
            "followed_audio": nat_audio,
            "other": nat_other,
            "tdr": tdr,
            "ci_95": list(ci),
        }

    # Omnibus chi-squared: TDR independence across groups
    # Build contingency table: rows = groups, cols = [followed_text, followed_audio]
    group_names = sorted(groups.keys())
    if len(group_names) >= 2:
        contingency = np.array([
            [group_results[g]["followed_text"], group_results[g]["followed_audio"]]
            for g in group_names
        ])
        # Remove rows with zero totals
        row_sums = contingency.sum(axis=1)
        valid_mask = row_sums > 0
        valid_names = [g for g, v in zip(group_names, valid_mask) if v]
        contingency_valid = contingency[valid_mask]

        if len(valid_names) >= 2:
            omnibus = chi_squared_independence(contingency_valid)
        else:
            omnibus = {
                "statistic": None,
                "p_value": None,
                "method": "not_applicable",
                "note": "Fewer than 2 groups with data",
            }
    else:
        omnibus = {
            "statistic": None,
            "p_value": None,
            "method": "not_applicable",
            "note": "Fewer than 2 groups",
        }

    omnibus["alpha"] = alpha_omnibus
    omnibus["significant"] = (
        omnibus["p_value"] < alpha_omnibus
        if omnibus["p_value"] is not None
        else None
    )

    # Pairwise McNemar (only if omnibus significant)
    pairwise = {}
    if omnibus.get("significant"):
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonf_alpha = alpha_omnibus / n_comparisons if n_comparisons > 0 else alpha_pairwise

        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                key = f"{g1}_vs_{g2}"
                # For pairwise, compare TDR within each group using natural trials
                # This is a proportions comparison, not paired McNemar
                r1 = group_results[g1]
                r2 = group_results[g2]
                if r1["n_clear"] > 0 and r2["n_clear"] > 0:
                    # Two-proportion z-test
                    n1, n2 = r1["n_clear"], r2["n_clear"]
                    p1 = r1["tdr"]
                    p2 = r2["tdr"]
                    p_pool = (r1["followed_text"] + r2["followed_text"]) / (n1 + n2)
                    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if p_pool > 0 and p_pool < 1 else 0
                    if se > 0:
                        z_stat = (p1 - p2) / se
                        p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_stat)))
                    else:
                        z_stat = 0.0
                        p_val = 1.0

                    h = cohens_h(p1, p2)
                    pairwise[key] = {
                        "z_statistic": float(z_stat),
                        "p_value": float(p_val),
                        "bonferroni_alpha": float(bonf_alpha),
                        "significant": p_val < bonf_alpha,
                        "tdr_1": p1,
                        "tdr_2": p2,
                        "cohens_h": float(h),
                        "cohens_h_magnitude": cohens_h_magnitude(h),
                    }
                else:
                    pairwise[key] = {
                        "note": "Insufficient data for comparison",
                    }

    return {
        "groups": group_results,
        "omnibus": omnibus,
        "pairwise": pairwise if pairwise else None,
    }


def analyze_language_effect(
    pairs: List[Dict],
    stimuli_meta: Dict[str, Dict],
    alpha_omnibus: float = 0.05,
) -> Dict[str, Any]:
    """Chi-squared test for TDR × language independence.

    With 8 languages, there are 28 pairwise comparisons
    (gated by omnibus chi-squared significance).
    """
    return _stratified_tdr_analysis(
        pairs, stimuli_meta,
        group_fn=lambda meta: _get_language(meta),
        alpha_omnibus=alpha_omnibus,
    )


# Backward-compat alias
analyze_bucket_effect = analyze_language_effect


def analyze_flip_type_effect(
    pairs: List[Dict],
    stimuli_meta: Dict[str, Dict],
    alpha_omnibus: float = 0.05,
) -> Dict[str, Any]:
    """Chi-squared test for TDR × flip type independence."""
    return _stratified_tdr_analysis(
        pairs, stimuli_meta,
        group_fn=lambda meta: meta.get("question", {}).get("flip_type", "UNKNOWN"),
        alpha_omnibus=alpha_omnibus,
    )


# ---------------------------------------------------------------------------
# Per-group natural vs TTS McNemar comparisons
# ---------------------------------------------------------------------------

def analyze_tdr_by_group(
    pairs: List[Dict],
    stimuli_meta: Dict[str, Dict],
    group_fn,
    alpha: float = 0.005,
) -> Dict[str, Dict]:
    """Run McNemar natural-vs-TTS comparison within each group."""
    groups: Dict[str, List[Dict]] = {}
    for pair in pairs:
        base_id = pair["base_id"]
        meta = stimuli_meta.get(base_id, {})
        group = group_fn(meta)
        groups.setdefault(group, []).append(pair)

    results = {}
    for group_name in sorted(groups.keys()):
        results[group_name] = analyze_tdr_comparison(groups[group_name], alpha=alpha)
    return results


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def results_to_json(all_results: Dict[str, Any]) -> str:
    """Serialise full results to JSON string."""
    return json.dumps(all_results, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _fmt_p(p: Optional[float]) -> str:
    if p is None:
        return "N/A"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _fmt_ci(ci: list) -> str:
    if ci[0] is None:
        return "N/A"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.1%}"


def _sig_marker(result: Dict) -> str:
    sig = result.get("significant")
    if sig is None:
        return ""
    return " *" if sig else ""


def format_text_summary(all_results: Dict[str, Any]) -> str:
    """Format results as markdown tables suitable for SPEC.md."""
    lines = []
    model = all_results.get("model", "Unknown")
    lines.append(f"### Statistical Analysis — {model}\n")

    # Overall TDR comparison
    overall = all_results["overall_tdr"]
    lines.append("#### Overall TDR: Natural vs TTS\n")
    lines.append("| Metric | Natural | TTS |")
    lines.append("|--------|---------|-----|")
    lines.append(f"| TDR | {_fmt_pct(overall['natural']['tdr'])} | {_fmt_pct(overall['tts']['tdr'])} |")
    lines.append(f"| 95% CI | {_fmt_ci(overall['natural']['ci_95'])} | {_fmt_ci(overall['tts']['ci_95'])} |")
    lines.append(f"| Followed text | {overall['natural']['followed_text']} | {overall['tts']['followed_text']} |")
    lines.append(f"| Followed audio | {overall['natural']['followed_audio']} | {overall['tts']['followed_audio']} |")
    lines.append(f"| Other | {overall['natural']['other']} | {overall['tts']['other']} |")
    lines.append(f"| n (clear) | {overall['natural']['n_clear']} | {overall['tts']['n_clear']} |")
    lines.append("")
    mcn = overall["mcnemar"]
    lines.append(f"- **McNemar test**: {mcn['method']}, p = {_fmt_p(mcn['p_value'])}"
                 f" (b={mcn['b']}, c={mcn['c']}){_sig_marker(overall)}")
    lines.append(f"- **Cohen's h**: {overall['cohens_h']:.3f} ({overall['cohens_h_magnitude']})")
    lines.append(f"- **Alpha** (Bonferroni-adjusted): {overall['alpha']}")
    lines.append("")

    # Audio-only comparison
    ao = all_results["audio_only"]
    lines.append("#### Audio-Only Accuracy: Natural vs TTS\n")
    lines.append(f"- Natural: {_fmt_pct(ao['natural_accuracy'])} ({ao['natural_correct']}/{ao['n_pairs']}), "
                 f"95% CI {_fmt_ci(ao['natural_ci_95'])}")
    lines.append(f"- TTS: {_fmt_pct(ao['tts_accuracy'])} ({ao['tts_correct']}/{ao['n_pairs']}), "
                 f"95% CI {_fmt_ci(ao['tts_ci_95'])}")
    mcn = ao["mcnemar"]
    lines.append(f"- McNemar: {mcn['method']}, p = {_fmt_p(mcn['p_value'])}{_sig_marker(ao)}")
    if ao['cohens_h'] is not None:
        lines.append(f"- Cohen's h: {ao['cohens_h']:.3f} ({ao['cohens_h_magnitude']})")
    lines.append("")

    # TDR by language (natural vs TTS within each language)
    if "tdr_by_language" in all_results:
        lines.append("#### TDR by Language: Natural vs TTS\n")
        lines.append("| Language | Natural TDR | TTS TDR | n (nat) | n (tts) | McNemar p | Cohen's h | Sig |")
        lines.append("|----------|------------|---------|---------|---------|-----------|-----------|-----|")
        for bucket, data in sorted(all_results["tdr_by_language"].items()):
            nat_tdr = _fmt_pct(data["natural"]["tdr"])
            tts_tdr = _fmt_pct(data["tts"]["tdr"])
            p = _fmt_p(data["mcnemar"]["p_value"])
            h = f"{data['cohens_h']:.3f}" if data["cohens_h"] is not None else "N/A"
            sig = "Yes" if data.get("significant") else "No"
            if data.get("significant") is None:
                sig = "N/A"
            lines.append(f"| {bucket} | {nat_tdr} | {tts_tdr} | {data['natural']['n_clear']} | {data['tts']['n_clear']} | {p} | {h} | {sig} |")
        lines.append("")

    # TDR by flip type (natural vs TTS within each type)
    if "tdr_by_flip_type" in all_results:
        lines.append("#### TDR by Flip Type: Natural vs TTS\n")
        lines.append("| Flip Type | Natural TDR | TTS TDR | n (nat) | n (tts) | McNemar p | Cohen's h | Sig |")
        lines.append("|-----------|------------|---------|---------|---------|-----------|-----------|-----|")
        for ft, data in sorted(all_results["tdr_by_flip_type"].items()):
            nat_tdr = _fmt_pct(data["natural"]["tdr"])
            tts_tdr = _fmt_pct(data["tts"]["tdr"])
            p = _fmt_p(data["mcnemar"]["p_value"])
            h = f"{data['cohens_h']:.3f}" if data["cohens_h"] is not None else "N/A"
            sig = "Yes" if data.get("significant") else "No"
            if data.get("significant") is None:
                sig = "N/A"
            lines.append(f"| {ft} | {nat_tdr} | {tts_tdr} | {data['natural']['n_clear']} | {data['tts']['n_clear']} | {p} | {h} | {sig} |")
        lines.append("")

    # Also check backward-compat key
    if "tdr_by_bucket" in all_results and "tdr_by_language" not in all_results:
        lines.append("#### TDR by Language: Natural vs TTS\n")
        lines.append("| Language | Natural TDR | TTS TDR | n (nat) | n (tts) | McNemar p | Cohen's h | Sig |")
        lines.append("|----------|------------|---------|---------|---------|-----------|-----------|-----|")
        for bucket, data in sorted(all_results["tdr_by_bucket"].items()):
            nat_tdr = _fmt_pct(data["natural"]["tdr"])
            tts_tdr = _fmt_pct(data["tts"]["tdr"])
            p = _fmt_p(data["mcnemar"]["p_value"])
            h = f"{data['cohens_h']:.3f}" if data["cohens_h"] is not None else "N/A"
            sig = "Yes" if data.get("significant") else "No"
            if data.get("significant") is None:
                sig = "N/A"
            lines.append(f"| {bucket} | {nat_tdr} | {tts_tdr} | {data['natural']['n_clear']} | {data['tts']['n_clear']} | {p} | {h} | {sig} |")
        lines.append("")

    # Language effect (omnibus) — check both new and old key
    be_key = "language_effect" if "language_effect" in all_results else "bucket_effect"
    if be_key in all_results:
        be = all_results[be_key]
        lines.append("#### Omnibus: TDR × Language Independence (Natural)\n")
        omni = be["omnibus"]
        if omni["method"] != "not_applicable":
            chi2_str = f"{omni['statistic']:.2f}" if omni.get("statistic") is not None else "N/A"
            v_str = f"{omni['cramers_v']:.3f}" if omni.get("cramers_v") is not None else "N/A"
            lines.append(f"- {omni['method']}: χ²={chi2_str}, "
                         f"p = {_fmt_p(omni['p_value'])}, "
                         f"Cramér's V = {v_str}")
        else:
            lines.append(f"- {omni.get('note', 'Not applicable')}")
        lines.append("")

    # Flip type effect (omnibus)
    if "flip_type_effect" in all_results:
        fe = all_results["flip_type_effect"]
        lines.append("#### Omnibus: TDR × Flip Type Independence (Natural)\n")
        omni = fe["omnibus"]
        if omni["method"] != "not_applicable":
            chi2_str = f"{omni['statistic']:.2f}" if omni.get("statistic") is not None else "N/A"
            v_str = f"{omni['cramers_v']:.3f}" if omni.get("cramers_v") is not None else "N/A"
            lines.append(f"- {omni['method']}: χ²={chi2_str}, "
                         f"p = {_fmt_p(omni['p_value'])}, "
                         f"Cramér's V = {v_str}")
        else:
            lines.append(f"- {omni.get('note', 'Not applicable')}")

        if fe.get("pairwise"):
            lines.append("")
            lines.append("Post-hoc pairwise comparisons (Bonferroni-adjusted):\n")
            lines.append("| Comparison | TDR₁ | TDR₂ | z | p | Cohen's h | Sig |")
            lines.append("|------------|------|------|---|---|-----------|-----|")
            for key, pw in sorted(fe["pairwise"].items()):
                if "note" in pw:
                    lines.append(f"| {key} | — | — | — | — | — | {pw['note']} |")
                else:
                    lines.append(
                        f"| {key} | {_fmt_pct(pw['tdr_1'])} | {_fmt_pct(pw['tdr_2'])} "
                        f"| {pw['z_statistic']:.2f} | {_fmt_p(pw['p_value'])} "
                        f"| {pw['cohens_h']:.3f} | {'Yes' if pw['significant'] else 'No'} |"
                    )
        lines.append("")

    # Sensitivity analysis
    if "sensitivity" in all_results:
        sens = all_results["sensitivity"]
        lines.append("#### Sensitivity Analysis (Other → Not-Text)\n")
        lines.append(f"- Natural TDR: {_fmt_pct(sens['natural_tdr'])} "
                     f"(CI {_fmt_ci(sens['natural_ci_95'])})")
        lines.append(f"- TTS TDR: {_fmt_pct(sens['tts_tdr'])} "
                     f"(CI {_fmt_ci(sens['tts_ci_95'])})")
        mcn = sens["mcnemar"]
        lines.append(f"- McNemar: p = {_fmt_p(mcn['p_value'])}{_sig_marker(sens)}")
        if sens['cohens_h'] is not None:
            lines.append(f"- Cohen's h: {sens['cohens_h']:.3f} ({sens['cohens_h_magnitude']})")
        lines.append("")

    # Multiple comparisons note
    lines.append("#### Multiple Comparisons Correction\n")
    lines.append("- 15 primary McNemar tests (1 overall + 8 language + 5 flip type + 1 audio-only)")
    lines.append("- Bonferroni-adjusted α = 0.05 / 15 ≈ 0.0033")
    lines.append("- Omnibus chi-squared tests at α = 0.05")
    lines.append("- 8 languages → 28 pairwise comparisons (gated by omnibus chi-squared)")
    lines.append("- Post-hoc pairwise only if omnibus significant, Bonferroni within family")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level analysis runner
# ---------------------------------------------------------------------------

def run_full_analysis(
    natural_trials: List[Dict],
    tts_trials: List[Dict],
    stimuli_meta: Dict[str, Dict],
    model: str = "qwen2-audio",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Run all statistical analyses.

    Args:
        natural_trials: trial results from natural audio eval
        tts_trials: trial results from TTS eval
        stimuli_meta: stimulus metadata (natural IDs → stim dicts)
        model: model label
        alpha: family-wise alpha (Bonferroni adjustment applied internally)

    Returns:
        Dict with all analysis results.
    """
    # Bonferroni: 8 languages → 28 pairwise + overhead
    # 1 overall + 8 language + 5 flip type + 1 audio-only = 15 primary tests
    n_primary_tests = 15
    adjusted_alpha = alpha / n_primary_tests

    # Pair trials
    conflict_pairs = pair_conflict_trials(natural_trials, tts_trials)
    ao_pairs = pair_audio_only_trials(natural_trials, tts_trials)

    print(f"Paired {len(conflict_pairs)} conflict trial pairs")
    print(f"Paired {len(ao_pairs)} audio-only trial pairs")

    # Overall TDR comparison
    print("\n--- Overall TDR comparison ---")
    overall_tdr = analyze_tdr_comparison(conflict_pairs, alpha=adjusted_alpha)
    print(f"  Natural TDR: {_fmt_pct(overall_tdr['natural']['tdr'])}")
    print(f"  TTS TDR:     {_fmt_pct(overall_tdr['tts']['tdr'])}")
    print(f"  McNemar p:   {_fmt_p(overall_tdr['mcnemar']['p_value'])}")
    print(f"  Cohen's h:   {overall_tdr['cohens_h']:.3f} ({overall_tdr['cohens_h_magnitude']})")

    # Audio-only comparison
    print("\n--- Audio-only accuracy comparison ---")
    ao_result = analyze_audio_only_comparison(ao_pairs, alpha=adjusted_alpha)
    print(f"  Natural: {_fmt_pct(ao_result['natural_accuracy'])}")
    print(f"  TTS:     {_fmt_pct(ao_result['tts_accuracy'])}")
    print(f"  McNemar p: {_fmt_p(ao_result['mcnemar']['p_value'])}")

    # TDR by language (natural vs TTS within each language)
    print("\n--- TDR by language (natural vs TTS) ---")
    tdr_by_language = analyze_tdr_by_group(
        conflict_pairs, stimuli_meta,
        group_fn=lambda meta: _get_language(meta),
        alpha=adjusted_alpha,
    )
    for lang, data in sorted(tdr_by_language.items()):
        print(f"  {lang}: nat={_fmt_pct(data['natural']['tdr'])} "
              f"tts={_fmt_pct(data['tts']['tdr'])} "
              f"p={_fmt_p(data['mcnemar']['p_value'])}")

    # TDR by flip type (natural vs TTS within each type)
    print("\n--- TDR by flip type (natural vs TTS) ---")
    tdr_by_flip = analyze_tdr_by_group(
        conflict_pairs, stimuli_meta,
        group_fn=lambda meta: meta.get("question", {}).get("flip_type", "UNKNOWN"),
        alpha=adjusted_alpha,
    )
    for ft, data in sorted(tdr_by_flip.items()):
        print(f"  {ft}: nat={_fmt_pct(data['natural']['tdr'])} "
              f"tts={_fmt_pct(data['tts']['tdr'])} "
              f"p={_fmt_p(data['mcnemar']['p_value'])}")

    # Omnibus: language effect
    print("\n--- Omnibus: TDR × language independence ---")
    language_effect = analyze_language_effect(conflict_pairs, stimuli_meta)
    omni = language_effect["omnibus"]
    print(f"  {omni['method']}: p={_fmt_p(omni['p_value'])}")

    # Omnibus: flip type effect
    print("\n--- Omnibus: TDR × flip type independence ---")
    flip_effect = analyze_flip_type_effect(conflict_pairs, stimuli_meta)
    omni = flip_effect["omnibus"]
    print(f"  {omni['method']}: p={_fmt_p(omni['p_value'])}")

    # Sensitivity analysis
    print("\n--- Sensitivity analysis (other → not-text) ---")
    sensitivity = analyze_tdr_comparison_sensitivity(conflict_pairs, alpha=adjusted_alpha)
    print(f"  Natural TDR: {_fmt_pct(sensitivity['natural_tdr'])}")
    print(f"  TTS TDR:     {_fmt_pct(sensitivity['tts_tdr'])}")
    print(f"  McNemar p:   {_fmt_p(sensitivity['mcnemar']['p_value'])}")

    return {
        "model": model,
        "alpha_family": alpha,
        "alpha_bonferroni": adjusted_alpha,
        "n_primary_tests": n_primary_tests,
        "overall_tdr": overall_tdr,
        "audio_only": ao_result,
        "tdr_by_language": tdr_by_language,
        "tdr_by_bucket": tdr_by_language,  # backward compat alias
        "tdr_by_flip_type": tdr_by_flip,
        "language_effect": language_effect,
        "bucket_effect": language_effect,  # backward compat alias
        "flip_type_effect": flip_effect,
        "sensitivity": sensitivity,
    }
