"""Triage agent with self-correction and retry logic."""

import json
import re
from typing import Optional, Tuple

from . import config
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
    
    def _build_prompt(self, query: str) -> str:
        """Build the inference prompt."""
        return f"""<|im_start|>system
You are a clinical triage agent. Analyze patient intake notes and route to the correct tool.

Available tools:
1. trigger_emergency_response - Life-threatening cases
2. schedule_urgent_consult - Serious but non-fatal
3. routine_care_referral - Chronic conditions, checkups

Respond with ONLY a JSON object containing 'tool' and 'arguments'.
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
    
    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Extract JSON from model output.
        
        Handles:
        - Raw JSON
        - JSON in code blocks
        - JSON with surrounding text
        """
        # Try direct parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON object in text
        patterns = [
            r'```json\s*(.*?)\s*```',  # Code block
            r'```\s*(.*?)\s*```',       # Any code block
            r'\{[^{}]*"tool"[^{}]*\}',  # Simple object with tool
            r'\{.*?\}',                  # Any object
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match.strip())
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _generate(self, prompt: str) -> str:
        """Generate model response."""
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.GENERATION_MAX_TOKENS,
            temperature=config.GENERATION_TEMPERATURE,
            do_sample=True,
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
        
        prompt = self._build_prompt(query)
        
        for attempt in range(config.MAX_RETRIES):
            metadata["attempts"] = attempt + 1
            
            try:
                # Generate response
                raw_output = self._generate(prompt)
                metadata["raw_outputs"].append(raw_output)
                
                # Extract JSON
                parsed = self._extract_json(raw_output)
                if parsed is None:
                    raise ValueError(f"Could not extract JSON from: {raw_output[:100]}...")
                
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
                    prompt += f"\n\nError: {str(e)}. Please output valid JSON with 'tool' and 'arguments'.\n<|im_start|>assistant\n"
        
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
