"""Async synthetic data generation using Gemini 1.5 Flash or OpenAI GPT-5.2."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm_asyncio

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


def _extract_json(content: str) -> Optional[dict]:
    """Extract JSON from response content, handling markdown code blocks."""
    if not content:
        return None
    
    # Clean markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1]
    
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return None


async def generate_with_openai_async(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = None,
    reasoning_effort: str = None,
) -> Optional[dict]:
    """
    Generate using OpenAI's responses API with reasoning support.
    
    Uses the new `client.responses.create` API for GPT-5.2.
    """
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        model = model or config.OPENAI_MODEL
        reasoning_effort = reasoning_effort or config.OPENAI_REASONING_EFFORT
        
        full_input = f"{system_prompt}\n\nTask: {prompt}"
        
        response = await client.responses.create(
            model=model,
            input=full_input,
            reasoning={"effort": reasoning_effort}
        )
        
        # Extract text from response
        content = response.output_text if hasattr(response, 'output_text') else str(response)
        return _extract_json(content)
        
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return None


async def generate_with_gemini_async(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = None,
) -> Optional[dict]:
    """Generate using Gemini API (async wrapper)."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model or config.GEMINI_MODEL)
        
        # Gemini SDK is sync, run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model_instance.generate_content(
                f"{system_prompt}\n\n{prompt}",
                generation_config={
                    "temperature": 0.8,
                    "max_output_tokens": 512,
                    "response_mime_type": "application/json",
                }
            )
        )
        return json.loads(response.text)
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None


async def _generate_single(
    prompt: str,
    provider: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> Optional[dict]:
    """Generate a single example with rate limiting."""
    async with semaphore:
        if provider == "openai":
            return await generate_with_openai_async(prompt, SYSTEM_PROMPT, api_key)
        else:
            return await generate_with_gemini_async(prompt, SYSTEM_PROMPT, api_key)


async def generate_training_data_async(
    num_examples: int = config.NUM_TRAINING_EXAMPLES,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    provider: str = config.DATA_GEN_PROVIDER,
    max_concurrent: int = 5,
) -> list[dict]:
    """
    Generate synthetic training data asynchronously.
    
    Args:
        num_examples: Number of examples to generate
        output_path: Path to save JSONL output
        api_key: API key for the chosen provider
        provider: 'gemini' or 'openai'
        max_concurrent: Max concurrent API calls
        
    Returns:
        List of generated examples
    """
    import os
    
    if api_key is None:
        env_key = "OPENAI_API_KEY" if provider == "openai" else "GOOGLE_API_KEY"
        api_key = os.environ.get(env_key)
        if not api_key:
            raise ValueError(f"Set {env_key} environment variable or pass api_key argument")
    
    output_path = output_path or config.TRAIN_DATA_PATH
    
    # Build task list
    examples_per_prompt = max(1, num_examples // len(GENERATION_PROMPTS))
    tasks_prompts = []
    for prompt in GENERATION_PROMPTS:
        for _ in range(examples_per_prompt):
            if len(tasks_prompts) >= num_examples:
                break
            tasks_prompts.append(prompt)
    
    print(f"Generating {len(tasks_prompts)} examples using {provider} (async, {max_concurrent} concurrent)...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [_generate_single(p, provider, api_key, semaphore) for p in tasks_prompts]
    
    results = await tqdm_asyncio.gather(*tasks, desc="Generating")
    
    # Filter valid examples
    examples = [
        r for r in results 
        if r and all(k in r for k in ["query", "tool", "arguments"])
    ]
    
    # Save to JSONL
    with open(output_path, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(examples)} valid examples to {output_path}")
    return examples


def generate_training_data(
    num_examples: int = config.NUM_TRAINING_EXAMPLES,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    provider: str = config.DATA_GEN_PROVIDER,
    max_concurrent: int = 5,
) -> list[dict]:
    """
    Sync wrapper for generate_training_data_async.
    
    For use in notebooks/scripts that don't have an event loop.
    """
    return asyncio.run(
        generate_training_data_async(
            num_examples=num_examples,
            output_path=output_path,
            api_key=api_key,
            provider=provider,
            max_concurrent=max_concurrent,
        )
    )


def load_training_data(path: Optional[Path] = None) -> list[dict]:
    """Load training data from JSONL file."""
    path = path or config.TRAIN_DATA_PATH
    examples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def format_for_training(example: dict) -> dict:
    """Format a single example for SFT training."""
    instruction = f"""You are a clinical triage agent. Analyze the patient intake note and route to the appropriate tool.

Patient Note:
{example['query']}

Respond with a JSON object containing 'tool' and 'arguments'."""

    output = json.dumps({
        "tool": example["tool"],
        "arguments": example["arguments"]
    }, indent=2)
    
    return {"instruction": instruction, "output": output}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic training data.")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="openai", help="Provider")
    parser.add_argument("--api-key", help="API Key")
    parser.add_argument("--num", type=int, default=100, help="Number of examples")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    try:
        generate_training_data(
            num_examples=args.num,
            output_path=args.output,
            api_key=args.api_key,
            provider=args.provider,
            max_concurrent=args.concurrent,
        )
        print("✅ Data generation complete.")
    except Exception as e:
        print(f"❌ Error: {e}")
