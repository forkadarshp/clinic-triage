"""Triage agent with self-correction and retry logic."""

import ast
import json
import re
from typing import Optional, Tuple

from . import config
from . import prompts
from .schemas import parse_triage_output, get_mock_response, TriageOutput


class TriageAgent:
    """
    Clinical triage agent with self-correction capabilities.
    
    Handles:
    - JSON extraction from model output
    - Validation via Pydantic schemas
    - Retry logic for invalid outputs
    - Mock execution responses
    """

    _EMERGENCY_KEYWORDS = (
        "heart attack", "cardiac arrest", "no pulse", "unresponsive", "not breathing",
        "stroke", "facial droop", "slurred speech", "severe bleeding", "gunshot",
        "severe trauma", "anaphylaxis", "respiratory failure", "seizure",
        "crushing chest pain", "sudden collapse",
    )
    _URGENT_KEYWORDS = (
        "fracture", "broken", "deep cut", "laceration", "high fever", "104", "103",
        "kidney stone", "severe pain", "cellulitis", "infection", "vision loss",
        "abdominal pain", "vomiting", "swelling",
    )
    _ROUTINE_KEYWORDS = (
        "refill", "checkup", "annual", "physical", "follow-up", "screening",
        "vaccination", "stable", "routine", "lab work", "chronic",
    )
    _SYMPTOM_KEYWORDS = (
        "fever", "pain", "swelling", "vomiting", "nausea", "dizziness",
        "headache", "cough", "shortness of breath", "weakness", "rash",
    )
    
    def __init__(self, model=None, tokenizer=None):
        """
        Initialize the agent.
        
        Args:
            model: Fine-tuned model (will load from checkpoint if None)
            tokenizer: Model tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self._model_loaded = model is not None
    
    def load_model(self, model_path: Optional[str] = None):
        """Load model from checkpoint."""
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            raise ImportError("Please install unsloth")
        
        model_path = model_path or str(config.OUTPUT_DIR / "final_model")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=config.MAX_SEQ_LENGTH,
            load_in_4bit=config.LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)
        self._model_loaded = True
    
    def _build_prompt(self, query: str, error: Optional[str] = None) -> str:
        """Build the inference prompt aligned with the training template."""
        return prompts.build_inference_prompt(query, error=error)

    def _parse_candidate(self, candidate: str) -> Optional[dict]:
        """Parse a JSON-like candidate string into a dict."""
        candidate = candidate.strip()
        if not candidate:
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        try:
            data = ast.literal_eval(candidate)
            if isinstance(data, dict):
                return data
        except (ValueError, SyntaxError):
            pass

        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        if "'" in fixed and '"' not in fixed:
            fixed = fixed.replace("'", '"')

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from model output.
        
        Handles:
        - Raw JSON
        - JSON in code blocks
        - JSON with surrounding text (robust extraction)
        """
        if not text:
            return None

        # 1. Try direct parse
        parsed = self._parse_candidate(text)
        if parsed is not None:
            return parsed
        
        # 2. Extract from code blocks
        code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if code_block_match:
            parsed = self._parse_candidate(code_block_match.group(1))
            if parsed is not None:
                return parsed
        
        # 3. Robust substring search
        # Find all indices of '{' and '}'
        # This is strictly better than regex for nested structures
        brackets_start = [i for i, char in enumerate(text) if char == '{']
        brackets_end = [i for i, char in enumerate(text) if char == '}']
        
        if not brackets_start or not brackets_end:
            return None
            
        # Try to parse from largest possible span to smallest
        # We assume the relevant JSON is likely the largest object or the first one
        for start in brackets_start:
            for end in reversed(brackets_end):
                if end < start:
                    continue
                
                candidate = text[start : end + 1]
                data = self._parse_candidate(candidate)
                if isinstance(data, dict) and "tool" in data:
                    return data
        
        return None

    def _guess_tool_from_text(self, text: str) -> Optional[str]:
        """Guess the tool from raw text using tool names or keywords."""
        if not text:
            return None

        lower = text.lower()
        for tool in config.VALID_TOOLS:
            if tool in lower:
                return tool

        if any(keyword in lower for keyword in self._EMERGENCY_KEYWORDS):
            return config.TOOL_EMERGENCY
        if any(keyword in lower for keyword in self._URGENT_KEYWORDS):
            return config.TOOL_URGENT
        if any(keyword in lower for keyword in self._ROUTINE_KEYWORDS):
            return config.TOOL_ROUTINE

        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract a location hint from the query text."""
        match = re.search(r"(?:location|address)\s*[:\-]\s*([^\n\.]+)", query, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_symptoms(self, query: str) -> list[str]:
        lower = query.lower()
        symptoms = [kw for kw in self._SYMPTOM_KEYWORDS if kw in lower]
        if symptoms:
            return symptoms[:4]
        return ["unspecified symptoms"]

    def _guess_department(self, query: str) -> str:
        lower = query.lower()
        if "fracture" in lower or "broken" in lower or "orthopedic" in lower:
            return "Orthopedics"
        if "pregnan" in lower or "ob/gyn" in lower or "missed period" in lower:
            return "OB/GYN"
        if "kidney" in lower or "flank" in lower:
            return "Urology"
        if "vision" in lower or "eye" in lower:
            return "Ophthalmology"
        if "child" in lower or "pediatric" in lower:
            return "Pediatrics"
        return "Urgent Care"

    def _guess_specialty(self, query: str) -> str:
        lower = query.lower()
        if "diabetes" in lower:
            return "Endocrinology"
        if "blood pressure" in lower or "hypertension" in lower:
            return "Internal Medicine"
        if "skin" in lower or "rash" in lower:
            return "Dermatology"
        if "back pain" in lower:
            return "Physical Therapy"
        return "Family Medicine"

    def _guess_visit_type(self, query: str) -> str:
        lower = query.lower()
        if "refill" in lower:
            return "prescription refill"
        if "annual" in lower or "physical" in lower:
            return "annual physical"
        if "screening" in lower:
            return "screening visit"
        if "follow-up" in lower:
            return "follow-up visit"
        return "routine visit"

    def _coerce_output(self, parsed: dict, query: str) -> Optional[dict]:
        """Fill missing fields to satisfy schema requirements."""
        if not isinstance(parsed, dict):
            return None

        tool = parsed.get("tool")
        if tool not in config.VALID_TOOLS:
            tool = self._guess_tool_from_text(str(tool)) or self._guess_tool_from_text(query)
        if tool not in config.VALID_TOOLS:
            return None

        args = parsed.get("arguments")
        if not isinstance(args, dict):
            args = {}

        if tool == config.TOOL_EMERGENCY:
            location = args.get("location")
            if not isinstance(location, str) or not location.strip():
                location = self._extract_location(query) or "Unknown location"
            return {
                "tool": tool,
                "arguments": {
                    "location": location,
                    "severity": "CRITICAL",
                },
            }

        if tool == config.TOOL_URGENT:
            department = args.get("department")
            if not isinstance(department, str) or not department.strip():
                department = self._guess_department(query)
            symptoms = args.get("symptoms")
            if not isinstance(symptoms, list) or not symptoms:
                symptoms = self._extract_symptoms(query)
            return {
                "tool": tool,
                "arguments": {
                    "department": department,
                    "symptoms": symptoms,
                },
            }

        if tool == config.TOOL_ROUTINE:
            visit_type = args.get("type")
            if not isinstance(visit_type, str) or not visit_type.strip():
                visit_type = self._guess_visit_type(query)
            specialty = args.get("specialty")
            if not isinstance(specialty, str) or not specialty.strip():
                specialty = self._guess_specialty(query)
            return {
                "tool": tool,
                "arguments": {
                    "type": visit_type,
                    "specialty": specialty,
                },
            }

        return None

    def _infer_output_from_text(self, text: str, query: str) -> Optional[dict]:
        tool = self._guess_tool_from_text(text) or self._guess_tool_from_text(query)
        if not tool:
            return None
        return self._coerce_output({"tool": tool, "arguments": {}}, query)
    
    def _generate(self, prompt: str) -> str:
        """Generate model response with optimized inference."""
        import torch
        
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.GENERATION_MAX_TOKENS,
                do_sample=False,  # Greedy decoding for deterministic JSON
                use_cache=True,  # Enable KV cache for faster generation
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def run(self, query: str) -> Tuple[Optional[TriageOutput], str, dict]:
        """
        Run the triage agent with self-correction.
        
        Args:
            query: Patient intake note
            
        Returns:
            Tuple of:
            - Validated TriageOutput (or None if all retries failed)
            - Mock response string
            - Metadata dict with retry info
        """
        metadata = {
            "attempts": 0,
            "errors": [],
            "raw_outputs": [],
        }
        
        error_context = None

        for attempt in range(config.MAX_RETRIES):
            metadata["attempts"] = attempt + 1
            
            try:
                prompt = self._build_prompt(query, error=error_context)

                # Generate response
                raw_output = self._generate(prompt)
                metadata["raw_outputs"].append(raw_output)
                
                # Extract JSON
                parsed = self._extract_json(raw_output)
                if parsed is None:
                    parsed = self._infer_output_from_text(raw_output, query)

                parsed = self._coerce_output(parsed, query)
                if parsed is None:
                    raise ValueError(f"Could not extract valid JSON from: {raw_output[:100]}...")
                
                # Validate with Pydantic
                validated = parse_triage_output(parsed)
                
                # Success - return result
                mock_response = get_mock_response(validated)
                return validated, mock_response, metadata
                
            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {type(e).__name__}: {str(e)}"
                metadata["errors"].append(error_msg)
                
                # Add error context to prompt for retry
                if attempt < config.MAX_RETRIES - 1:
                    error_context = str(e)
        
        # All retries failed
        return None, "[System] ⚠️ Triage failed after multiple attempts.", metadata


def run_triage_agent(query: str, model=None, tokenizer=None) -> dict:
    """
    Convenience function to run triage on a single query.
    
    Args:
        query: Patient intake note
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        
    Returns:
        Dict with 'output', 'response', and 'metadata' keys
    """
    agent = TriageAgent(model=model, tokenizer=tokenizer)
    
    if not agent._model_loaded:
        agent.load_model()
    
    output, response, metadata = agent.run(query)
    
    return {
        "output": output.model_dump() if output else None,
        "response": response,
        "metadata": metadata,
    }
