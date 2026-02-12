#!/usr/bin/env python3
"""Remap stimuli audio paths to TTS audio files.

Reads the standard stimuli JSONL and rewrites each audio_path to point at
the corresponding TTS WAV file, producing a new stimuli file suitable for
evaluation with --cv-root pointed at the TTS audio directory.

TTS files are named: {lang}/tts_{stimulus_id}.wav
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stimuli", required=True,
        help="Input stimuli JSONL (with original CV audio paths)",
    )
    parser.add_argument(
        "--tts-audio-dir", required=True,
        help="Directory containing TTS audio (e.g. data/tts_audio/)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output stimuli JSONL with remapped TTS audio paths",
    )
    args = parser.parse_args()

    written = 0
    skipped = 0

    with open(args.stimuli) as fin, open(args.output, "w") as fout:
        for line in fin:
            stimulus = json.loads(line)
            lang = stimulus["language"].lower()
            sid = stimulus["stimulus_id"]
            tts_path = f"{lang}/tts_{sid}.wav"

            # Check that the TTS file exists
            full_path = os.path.join(args.tts_audio_dir, tts_path)
            if not os.path.exists(full_path):
                skipped += 1
                continue

            stimulus["audio_path"] = tts_path
            fout.write(json.dumps(stimulus, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written: {written}")
    if skipped:
        print(f"Skipped (TTS file not found): {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
