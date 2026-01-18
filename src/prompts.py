"""Shared prompt templates for training and inference."""

from __future__ import annotations

from typing import Optional

SYSTEM_PROMPT = """You are a clinical triage agent. Analyze patient intake notes and route to exactly one of these tools.

TOOL 1: trigger_emergency_response
When: Life-threatening emergencies (heart attack, stroke, severe trauma, anaphylaxis)
JSON: {"tool": "trigger_emergency_response", "arguments": {"location": "<patient location>", "severity": "CRITICAL"}}

TOOL 2: schedule_urgent_consult
When: Serious but stable (high fever, fractures, deep cuts, infections)
JSON: {"tool": "schedule_urgent_consult", "arguments": {"department": "<specialty>", "symptoms": ["symptom1", "symptom2"]}}

TOOL 3: routine_care_referral
When: Non-urgent (checkups, refills, chronic disease management)
JSON: {"tool": "routine_care_referral", "arguments": {"type": "<visit type>", "specialty": "<specialty>"}}

OUTPUT: Respond with ONLY the JSON object, no other text."""

PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    f"{SYSTEM_PROMPT}\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "{instruction}\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "{output}\n"
    "<|im_end|>"
)


def build_inference_prompt(query: str, error: Optional[str] = None) -> str:
    """Build the inference prompt aligned with the training template."""
    user_content = query
    if error:
        user_content = (
            f"{query}\n\nPrevious error: {error}\n"
            "Return ONLY valid JSON with 'tool' and 'arguments'."
        )

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
