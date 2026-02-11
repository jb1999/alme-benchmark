#!/usr/bin/env python3
"""Verify that all audio files referenced in stimuli.jsonl exist on disk.

Reads stimuli.jsonl (relative paths) and checks that each audio file
exists under the specified --cv-root directory. Reports per-language
missing counts.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Verify audio files exist for all stimuli"
    )
    parser.add_argument(
        "--stimuli",
        type=str,
        default=None,
        help="Path to stimuli JSONL (default: data/stimuli.jsonl)",
    )
    parser.add_argument(
        "--cv-root",
        type=str,
        default=None,
        help="Root directory for Common Voice corpus. "
             "Also reads ALME_CV_ROOT env var.",
    )
    args = parser.parse_args()

    cv_root = args.cv_root or os.environ.get("ALME_CV_ROOT")
    if not cv_root:
        parser.error(
            "--cv-root is required (or set ALME_CV_ROOT). "
            "Point to the CV 22.0 corpus directory."
        )
    cv_root = Path(cv_root)

    stimuli_path = args.stimuli
    if stimuli_path is None:
        pkg_dir = Path(__file__).resolve().parent.parent
        stimuli_path = str(pkg_dir / "data" / "stimuli.jsonl")

    total = 0
    found = 0
    missing_by_lang = defaultdict(int)
    found_by_lang = defaultdict(int)

    with open(stimuli_path, "r") as f:
        for line in f:
            stim = json.loads(line)
            total += 1
            lang = stim.get("language", "UNKNOWN")
            audio_path = cv_root / stim["audio_path"]

            if audio_path.exists():
                found += 1
                found_by_lang[lang] += 1
            else:
                missing_by_lang[lang] += 1

    missing = total - found

    print(f"Stimuli: {total}")
    print(f"Found:   {found}")
    print(f"Missing: {missing}")
    print()

    if missing_by_lang:
        print("Missing by language:")
        for lang in sorted(missing_by_lang):
            n_miss = missing_by_lang[lang]
            n_found = found_by_lang.get(lang, 0)
            n_total = n_miss + n_found
            print(f"  {lang}: {n_miss}/{n_total} missing")
        print()

    if found_by_lang:
        print("Found by language:")
        for lang in sorted(found_by_lang):
            n_found = found_by_lang[lang]
            n_miss = missing_by_lang.get(lang, 0)
            n_total = n_found + n_miss
            print(f"  {lang}: {n_found}/{n_total}")

    if missing > 0:
        print(f"\nERROR: {missing} audio files missing. "
              "Download the full Common Voice 22.0 corpus.")
        return 1

    print("\nAll audio files found!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
