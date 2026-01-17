"""Async synthetic data generation using Gemini 1.5 Flash or OpenAI GPT-5.2."""

import asyncio
import json
from pathlib import Path
from typing import Optional

from tqdm.asyncio import tqdm_asyncio

from . import config


# =============================================================================
# Prompt Templates - Optimized for Training Data Quality
# =============================================================================
SYSTEM_PROMPT = """You are a clinical AI data generator creating high-quality training examples for a patient triage system.

TASK: Generate realistic, diverse patient intake notes with correct triage classifications.

STRICT RULES:
1. Output ONLY valid JSON matching the exact schema below
2. Use ONLY these 3 tools - no other tool names are allowed
3. The "query" must be a realistic clinical intake note with varied writing styles
4. Arguments must match the EXACT schema for each tool
5. The "severity" for emergencies must ALWAYS be exactly "CRITICAL"
6. The "symptoms" for urgent cases must ALWAYS be a JSON array of strings

===== TOOL SCHEMAS =====

TOOL 1: trigger_emergency_response
Use when: Life-threatening conditions requiring immediate intervention
Examples: Active MI, stroke (within window), major trauma, anaphylaxis, cardiac arrest, severe hemorrhage, respiratory failure
Arguments: {"location": "<specific patient location>", "severity": "CRITICAL"}

TOOL 2: schedule_urgent_consult
Use when: Serious conditions needing same-day evaluation but patient is stable
Examples: High fever in children, suspected fractures, deep lacerations, acute non-cardiac chest pain, severe infections, kidney stones
Arguments: {"department": "<medical specialty>", "symptoms": ["symptom1", "symptom2", ...]}

TOOL 3: routine_care_referral
Use when: Non-urgent, planned care
Examples: Medication refills, annual physicals, chronic disease follow-ups, preventive screenings, vaccination appointments
Arguments: {"type": "<visit type>", "specialty": "<medical specialty>"}

===== GOLD EXAMPLES =====

{"query": "78yo male found unresponsive at bus stop by bystander, agonal breathing, no pulse detected, CPR initiated by off-duty nurse. EMS en route. Location: Main St bus shelter near City Hall.", "tool": "trigger_emergency_response", "arguments": {"location": "Main St bus shelter near City Hall", "severity": "CRITICAL"}}

{"query": "32yo female, twisted right ankle during recreational soccer 2 hours ago. Reports hearing a 'pop' at time of injury. Significant lateral swelling, ecchymosis developing, unable to bear weight. Pain 7/10, no numbness. Ice applied.", "tool": "schedule_urgent_consult", "arguments": {"department": "Orthopedics", "symptoms": ["ankle swelling", "inability to bear weight", "ecchymosis", "acute pain"]}}

{"query": "Established patient here for quarterly diabetes management. Last A1c 7.1%, down from 7.4%. Home glucose logs show good control. Needs metformin 1000mg refill x90 days. No new symptoms, feet exam due.", "tool": "routine_care_referral", "arguments": {"type": "chronic disease follow-up", "specialty": "Endocrinology"}}

===== OUTPUT FORMAT =====
{"query": "<detailed patient intake note>", "tool": "<exact tool name>", "arguments": {<tool-specific arguments>}}

NOW GENERATE ONE TRAINING EXAMPLE:"""

GENERATION_PROMPTS = [
    # === EMERGENCY CASES (varied presentations) ===
    "Generate intake note: elderly patient with sudden onset stroke symptoms (use FAST criteria). Vary the location.",
    "Generate intake note: young adult in anaphylaxis from unknown allergen at a restaurant.",
    "Generate intake note: construction worker with severe crush injury, written from paramedic perspective.",
    "Generate intake note: pregnant woman with heavy vaginal bleeding and abdominal pain.",
    "Generate intake note: child found unresponsive after near-drowning incident at home pool.",
    "Generate intake note: patient with acute MI symptoms but atypical presentation (diabetic, female, or elderly).",
    "Generate intake note: severe asthma attack with impending respiratory failure.",
    "Generate intake note: gunshot wound victim, bystander report style.",
    "Generate intake note: patient with sudden severe headache 'worst of life' (possible SAH).",
    "Generate intake note: electrocution injury at workplace.",
    
    # === URGENT CASES (varied presentations) ===
    "Generate intake note: toddler with 103°F fever for 24 hours, parent's description style.",
    "Generate intake note: elderly fall with hip pain, unable to walk, nursing home staff report.",
    "Generate intake note: young athlete with possible ACL tear during game.",
    "Generate intake note: adult with severe ear pain and discharge, written informally.",
    "Generate intake note: patient with kidney stone symptoms, writhing in pain.",
    "Generate intake note: deep dog bite on forearm requiring evaluation.",
    "Generate intake note: teenager with severe sore throat, difficulty swallowing (possible peritonsillar abscess).",
    "Generate intake note: adult with sudden vision changes in one eye.",
    "Generate intake note: patient with cellulitis spreading up leg with fever.",
    "Generate intake note: child who swallowed a small battery 1 hour ago.",
    
    # === ROUTINE CASES (varied presentations) ===
    "Generate intake note: middle-aged patient for annual wellness visit, brief note style.",
    "Generate intake note: patient requesting birth control prescription renewal.",
    "Generate intake note: elderly patient for blood pressure medication titration.",
    "Generate intake note: patient scheduling colonoscopy screening at age 50.",
    "Generate intake note: follow-up for well-controlled hypothyroidism.",
    "Generate intake note: patient with stable chronic back pain, needs PT referral.",
    "Generate intake note: child for school sports physical examination.",
    "Generate intake note: patient requesting allergy shot continuation.",
    "Generate intake note: new patient intake for established anxiety disorder, stable on meds.",
    "Generate intake note: patient for routine diabetic eye exam referral.",
    
    # === EDGE CASES (teach model boundaries) ===
    "Generate intake note: chest pain that is clearly musculoskeletal (urgent, not emergency).",
    "Generate intake note: low-grade fever for 5 days with mild symptoms (routine vs urgent boundary).",
    "Generate intake note: headache with some concerning features but stable vitals (urgent consult).",
    "Generate intake note: elderly patient with confusion but no focal deficits, family unsure of timeline.",
    "Generate intake note: abdominal pain that sounds like constipation, not surgical.",
    "Generate intake note: patient with anxiety presenting with chest tightness and palpitations (not cardiac).",
    
    # === DIVERSITY IN STYLE ===
    "Generate intake note: written like a triage nurse's quick assessment.",
    "Generate intake note: written like a detailed EMT handoff report.",
    "Generate intake note: written like a family member describing symptoms over phone.",
    "Generate intake note: written with heavy use of medical abbreviations.",
    "Generate intake note: written in simple layman's terms by patient themselves."
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


VALID_TOOLS = {
    "trigger_emergency_response",
    "schedule_urgent_consult", 
    "routine_care_referral"
}


def _validate_example(data: dict) -> bool:
    """
    Validate that a generated example matches the required schema.
    
    Returns True if valid, False otherwise.
    """
    if not data:
        return False
    
    # Check required top-level keys
    if not all(k in data for k in ["query", "tool", "arguments"]):
        return False
    
    # Validate query is non-empty string
    if not isinstance(data.get("query"), str) or len(data["query"]) < 20:
        return False
    
    # Validate tool name
    tool = data.get("tool")
    if tool not in VALID_TOOLS:
        return False
    
    args = data.get("arguments", {})
    if not isinstance(args, dict):
        return False
    
    # Validate tool-specific arguments
    if tool == "trigger_emergency_response":
        if not isinstance(args.get("location"), str) or not args["location"]:
            return False
        if args.get("severity") != "CRITICAL":
            return False
            
    elif tool == "schedule_urgent_consult":
        if not isinstance(args.get("department"), str) or not args["department"]:
            return False
        symptoms = args.get("symptoms")
        if not isinstance(symptoms, list) or len(symptoms) == 0:
            return False
        if not all(isinstance(s, str) and s for s in symptoms):
            return False
            
    elif tool == "routine_care_referral":
        if not isinstance(args.get("type"), str) or not args["type"]:
            return False
        if not isinstance(args.get("specialty"), str) or not args["specialty"]:
            return False
    
    return True


async def generate_with_openai_async(
    prompt: str,
    system_prompt: str,
    api_key: str,
    model: str = None,
    reasoning_effort: str = None,
) -> Optional[dict]:
    """
    Generate using OpenAI API.
    
    Uses responses.create for GPT-5.x, chat.completions for GPT-4.x.
    """
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=api_key)
        model = model or config.OPENAI_MODEL
        reasoning_effort = reasoning_effort or config.OPENAI_REASONING_EFFORT
        
        full_input = f"{system_prompt}\n\nTask: {prompt}\n\nRespond with valid JSON only."
        
        # GPT-5.x uses responses API, GPT-4.x uses chat completions
        if model.startswith("gpt-5"):
            response = await client.responses.create(
                model=model,
                input=full_input,
                reasoning={"effort": reasoning_effort}
            )
            content = response.output_text if hasattr(response, 'output_text') else str(response)
        else:
            # GPT-4.x uses chat completions with json_object format
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        
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
    
    # Filter and validate examples
    valid_examples = []
    invalid_count = 0
    for r in results:
        if r and _validate_example(r):
            valid_examples.append(r)
        else:
            invalid_count += 1
    
    print(f"Validation: {len(valid_examples)} valid, {invalid_count} rejected")
    
    # Save to JSONL
    with open(output_path, "w") as f:
        for example in valid_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(valid_examples)} examples to {output_path}")
    return valid_examples


def generate_training_data(
    num_examples: int = config.NUM_TRAINING_EXAMPLES,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    provider: str = config.DATA_GEN_PROVIDER,
    max_concurrent: int = 5,
) -> list[dict]:
    """
    Sync wrapper for generate_training_data_async.
    
    Handles running in notebooks (with nest_asyncio) or scripts.
    """
    try:
        return asyncio.run(
            generate_training_data_async(
                num_examples=num_examples,
                output_path=output_path,
                api_key=api_key,
                provider=provider,
                max_concurrent=max_concurrent,
            )
        )
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            # Fallback for Jupyter/Colab
            try:
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(
                    generate_training_data_async(
                        num_examples=num_examples,
                        output_path=output_path,
                        api_key=api_key,
                        provider=provider,
                        max_concurrent=max_concurrent,
                    )
                )
            except ImportError:
                raise RuntimeError(
                    "Running in a notebook/loop? Please run `pip install nest_asyncio` to fix this error."
                ) from e
        raise e


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
    # Instruction is just the query, as the system prompt handles the task definition
    instruction = example["query"]

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
