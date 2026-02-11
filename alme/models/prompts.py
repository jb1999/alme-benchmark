"""Shared prompt templates for ALME evaluation.

All model adapters use these templates to ensure consistent prompting
across conditions. Four conditions are defined:

- text_only: Model receives only the transcript.
- audio_only: Model receives only the audio.
- audio_text_aligned: Model receives audio + correct transcript (ceiling).
- audio_text_conflict: Model receives audio + modified transcript (conflict).
"""

import json
from typing import List


def build_user_prompt(
    condition: str,
    text: str,
    question: str,
    choices: List[str],
) -> str:
    """Build the user prompt for a given condition.

    Args:
        condition: One of "text_only", "audio_only",
                   "audio_text_aligned", "audio_text_conflict".
        text: Transcript text (used for text_only, aligned, conflict).
              May be None for audio_only.
        question: The question to answer.
        choices: List of answer choices.

    Returns:
        Formatted prompt string.
    """
    choices_json = json.dumps(choices)
    answer_fmt = (
        "Your answer MUST be exactly one of the CHOICES above, "
        "copied verbatim (same language and script).\n"
        'Output JSON only: {"answer": "<exact choice>", '
        '"confidence": 0.0-1.0, "rationale": "brief"}'
    )

    if condition == "text_only":
        return (
            f'Given this transcript:\n"{text}"\n\n'
            f"QUESTION: {question}\n"
            f"CHOICES: {choices_json}\n\n"
            f"{answer_fmt}"
        )

    if condition == "audio_only":
        return (
            f"Listen to the audio and answer the question.\n\n"
            f"QUESTION: {question}\n"
            f"CHOICES: {choices_json}\n\n"
            f"{answer_fmt}"
        )

    if condition == "audio_text_aligned":
        return (
            f"Here is the audio and its transcript:\n"
            f'"{text}"\n\n'
            f"Answer the question based on the audio and transcript.\n\n"
            f"QUESTION: {question}\n"
            f"CHOICES: {choices_json}\n\n"
            f"{answer_fmt}"
        )

    if condition == "audio_text_conflict":
        return (
            f"INPUTS:\n"
            f"- Audio: [attached]\n"
            f'- Transcript (may contain errors):\n"{text}"\n\n'
            f"IMPORTANT: The transcript may be incorrect. "
            f"Answer based on what you HEAR in the audio.\n\n"
            f"QUESTION: {question}\n"
            f"CHOICES: {choices_json}\n\n"
            f"{answer_fmt}"
        )

    raise ValueError(
        f"Unknown condition: {condition}. Must be one of: "
        "text_only, audio_only, audio_text_aligned, audio_text_conflict"
    )
