"""Synthetic data generation using Gemini 1.5 Flash (free tier)."""

import json
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from . import config


# =============================================================================
# Prompt Templates
# =============================================================================
SYSTEM_PROMPT = """You are a medical data generator. Generate realistic patient intake notes and their correct triage classifications.

The 3 tools are:
1. trigger_emergency_response - Life-threatening (heart attack, stroke, trauma, severe bleeding)
   Arguments: {"location": "string", "severity": "CRITICAL"}

2. schedule_urgent_consult - Serious but non-fatal (high fever, fractures, severe pain)
   Arguments: {"department": "string", "symptoms": ["string", ...]}

3. routine_care_referral - Chronic conditions, refills, checkups
   Arguments: {"type": "string", "specialty": "string"}

Output JSON format:
{"query": "patient intake note...", "tool": "tool_name", "arguments": {...}}"""

GENERATION_PROMPTS = [
    # Emergency cases
    "Generate a patient intake note for a cardiac emergency. Include realistic details.",
    "Generate a patient intake note for a severe trauma case from a car accident.",
    "Generate a patient intake note for symptoms indicating a stroke.",
    "Generate a patient intake note for severe allergic reaction (anaphylaxis).",
    "Generate a patient intake note for a patient with severe chest pain and shortness of breath.",
    
    # Urgent cases
    "Generate a patient intake note for a patient with a suspected bone fracture.",
    "Generate a patient intake note for a child with high fever (104°F).",
    "Generate a patient intake note for severe abdominal pain.",
    "Generate a patient intake note for a deep laceration requiring stitches.",
    "Generate a patient intake note for difficulty breathing with wheezing.",
    
    # Routine cases
    "Generate a patient intake note for a diabetes medication refill.",
    "Generate a patient intake note for an annual physical checkup.",
    "Generate a patient intake note for chronic back pain follow-up.",
    "Generate a patient intake note for blood pressure management.",
    "Generate a patient intake note for a routine skin mole check.",
]


def generate_training_data(
    num_examples: int = config.NUM_TRAINING_EXAMPLES,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """
    Generate synthetic training data using Gemini 1.5 Flash.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save JSONL output (default: config.TRAIN_DATA_PATH)
        api_key: Gemini API key (will prompt if not provided)
        
    Returns:
        List of training examples
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Please install google-generativeai: pip install google-generativeai")
    
    if api_key is None:
        import os
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY environment variable or pass api_key argument")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    output_path = output_path or config.TRAIN_DATA_PATH
    examples = []
    
    # Calculate examples per category for balance
    examples_per_prompt = max(1, num_examples // len(GENERATION_PROMPTS))
    
    print(f"Generating {num_examples} training examples...")
    
    for prompt in tqdm(GENERATION_PROMPTS, desc="Categories"):
        for _ in range(examples_per_prompt):
            if len(examples) >= num_examples:
                break
                
            try:
                response = model.generate_content(
                    f"{SYSTEM_PROMPT}\n\n{prompt}",
                    generation_config={
                        "temperature": 0.8,
                        "max_output_tokens": 512,
                        "response_mime_type": "application/json",
                    }
                )
                
                # Parse and validate
                data = json.loads(response.text)
                if all(k in data for k in ["query", "tool", "arguments"]):
                    examples.append(data)
                    
            except (json.JSONDecodeError, Exception) as e:
                print(f"Skipping invalid response: {e}")
                continue
    
    # Save to JSONL
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(examples)} examples to {output_path}")
    return examples


def load_training_data(path: Optional[Path] = None) -> list[dict]:
    """Load training data from JSONL file."""
    path = path or config.TRAIN_DATA_PATH
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples


def format_for_training(example: dict) -> dict:
    """
    Format a single example for SFT training.
    
    Returns dict with 'instruction' and 'output' keys.
    """
    instruction = f"""You are a clinical triage agent. Analyze the patient intake note and route to the appropriate tool.

Patient Note:
{example['query']}

Respond with a JSON object containing 'tool' and 'arguments'."""

    output = json.dumps({
        "tool": example["tool"],
        "arguments": example["arguments"]
    }, indent=2)
    
    return {
        "instruction": instruction,
        "output": output,
    }
