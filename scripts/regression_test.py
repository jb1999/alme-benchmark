#!/usr/bin/env python3
"""Compare evaluation results against reference values.

Loads user results and reference metrics, compares TDR overall,
per-language, and per-flip-type within a configurable tolerance.
"""

import argparse
import json
import sys
from pathlib import Path


DEFAULT_TOLERANCE_PP = 2.0  # percentage points


def compare_values(label: str, actual: float, expected: float, tolerance_pp: float) -> bool:
    """Compare two values within tolerance. Returns True if pass."""
    if actual is None or expected is None:
        print(f"  {label}: SKIP (actual={actual}, expected={expected})")
        return True

    diff_pp = abs(actual - expected) * 100
    status = "PASS" if diff_pp <= tolerance_pp else "FAIL"
    print(f"  {label}: {actual:.4f} vs {expected:.4f} "
          f"(diff={diff_pp:.2f}pp) [{status}]")
    return diff_pp <= tolerance_pp


def main():
    parser = argparse.ArgumentParser(
        description="Regression test: compare results against reference"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results metrics JSON from evaluation",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference metrics JSON "
             "(default: data/reference/ultravox_metrics.json)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PP,
        help=f"Tolerance in percentage points (default: {DEFAULT_TOLERANCE_PP})",
    )
    args = parser.parse_args()

    # Load reference
    ref_path = args.reference
    if ref_path is None:
        pkg_dir = Path(__file__).resolve().parent.parent
        ref_path = str(pkg_dir / "data" / "reference" / "ultravox_metrics.json")

    with open(ref_path, "r") as f:
        ref = json.load(f)
    with open(args.results, "r") as f:
        res = json.load(f)

    tolerance = args.tolerance
    all_pass = True

    print(f"Reference: {ref_path}")
    print(f"Results:   {args.results}")
    print(f"Tolerance: {tolerance}pp")
    print()

    # Overall TDR
    print("=== Overall TDR ===")
    if not compare_values("TDR", res.get("tdr"), ref.get("tdr"), tolerance):
        all_pass = False
    print()

    # Per-language TDR
    ref_langs = ref.get("by_language", {})
    res_langs = res.get("by_language", {})
    if ref_langs and res_langs:
        print("=== TDR by Language ===")
        for lang in sorted(set(ref_langs) | set(res_langs)):
            ref_tdr = ref_langs.get(lang, {}).get("tdr")
            res_tdr = res_langs.get(lang, {}).get("tdr")
            if not compare_values(lang, res_tdr, ref_tdr, tolerance):
                all_pass = False
        print()

    # Per-flip-type TDR
    ref_flips = ref.get("by_flip_type", {})
    res_flips = res.get("by_flip_type", {})
    if ref_flips and res_flips:
        print("=== TDR by Flip Type ===")
        for ft in sorted(set(ref_flips) | set(res_flips)):
            ref_tdr = ref_flips.get(ft, {}).get("tdr")
            res_tdr = res_flips.get(ft, {}).get("tdr")
            if not compare_values(ft, res_tdr, ref_tdr, tolerance):
                all_pass = False
        print()

    if all_pass:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
