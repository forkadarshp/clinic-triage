import unittest
from unittest.mock import MagicMock
from src.agent import TriageAgent, run_triage_agent
from src.config import TOOL_EMERGENCY, TOOL_URGENT, TOOL_ROUTINE

class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 2
        
    def __call__(self, text, return_tensors="pt"):
        mock = MagicMock()
        mock.to.return_value = {"input_ids": MagicMock(shape=(1, 10))}
        return mock
    
    def decode(self, token_ids, skip_special_tokens=True):
        return "mock_response"

class MockModel:
    def __init__(self, responses=None):
        self.device = "cpu"
        self.responses = responses or []
        self.call_count = 0
        
    def generate(self, **kwargs):
        # Return next response in queue, or default/last one
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = self.responses[-1] if self.responses else ""
        
        self.call_count += 1
        
        # Mocking the output tensor structure is hard, so we just rely on tokenizer.decode mocking
        # Wait, Agent calls tokenizer.decode on output[0][input_len:]
        # So we need to ensure tokenizer.decode returns what we want based on inputs.
        # But tokenizer.decode is called on the result of model.generate.
        return MagicMock() 

# Start override: TriageAgent._generate is easier to mock than the model itself
# We will mock the _generate method of the agent instance in the tests.

class TestTriageAgent(unittest.TestCase):
    
    def setUp(self):
        self.agent = TriageAgent(model=MagicMock(), tokenizer=MagicMock())
        # We manually set _model_loaded to True since we passed mocks
        self.agent._model_loaded = True

    def test_extract_json_clean(self):
        """Test extraction from clean JSON string."""
        text = '{"tool": "trigger_emergency_response", "arguments": {"location": "home", "severity": "CRITICAL"}}'
        data = self.agent._extract_json(text)
        self.assertEqual(data["tool"], "trigger_emergency_response")
        
    def test_extract_json_markdown(self):
        """Test extraction from markdown block."""
        text = 'Here is the result:\n```json\n{"tool": "routine_care_referral", "arguments": {"type": "refill", "specialty": "PCP"}}\n```'
        data = self.agent._extract_json(text)
        self.assertEqual(data["tool"], "routine_care_referral")

    def test_extract_json_nested(self):
        """Test extraction from text with surrounding noise."""
        text = 'Sure, I can help. {"tool": "schedule_urgent_consult", "arguments": {"department": "ER", "symptoms": ["pain"]}} is the answer.'
        data = self.agent._extract_json(text)
        self.assertEqual(data["tool"], "schedule_urgent_consult")
        
    def test_run_success_first_try(self):
        """Test successful execution on first attempt."""
        valid_json = '{"tool": "trigger_emergency_response", "arguments": {"location": "gym", "severity": "CRITICAL"}}'
        self.agent._generate = MagicMock(return_value=valid_json)
        
        output, response, metadata = self.agent.run("Patient collapsed at gym")
        
        self.assertIsNotNone(output)
        self.assertEqual(output.tool, TOOL_EMERGENCY)
        self.assertEqual(metadata["attempts"], 1)
        self.assertEqual(len(metadata["errors"]), 0)

    def test_run_retry_logic(self):
        """Test retry logic: fail 2 times, succeed on 3rd."""
        # Response 1: Invalid JSON
        r1 = "I think it is emergency" 
        # Response 2: Invalid Schema (wrong tool name)
        r2 = '{"tool": "call_911", "arguments": {"location": "home"}}'
        # Response 3: Success
        r3 = '{"tool": "trigger_emergency_response", "arguments": {"location": "home", "severity": "CRITICAL"}}'
        
        self.agent._generate = MagicMock(side_effect=[r1, r2, r3])
        
        output, response, metadata = self.agent.run("Emergency at home")
        
        self.assertIsNotNone(output)
        self.assertEqual(output.tool, TOOL_EMERGENCY)
        self.assertEqual(metadata["attempts"], 3)
        self.assertEqual(len(metadata["errors"]), 2)
        # Check if error context was added to Prompt (not easily checkable without inspecting arg to _generate calls, but logic assumes it)

    def test_run_fail_all_retries(self):
        """Test heuristic fallback when model output is invalid."""
        bad_resp = "Invalid data"
        self.agent._generate = MagicMock(return_value=bad_resp)
        
        output, response, metadata = self.agent.run("Some query")
        
        self.assertIsNotNone(output)
        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(metadata["attempts"], 1)

    def test_validation_logic(self):
        """Test schema auto-correction fills required fields."""
        # Missing required argument 'location' (severity has default)
        invalid_args = '{"tool": "trigger_emergency_response", "arguments": {"severity": "CRITICAL"}}'
        self.agent._generate = MagicMock(return_value=invalid_args)
        
        output, _, metadata = self.agent.run("Query")
        self.assertIsNotNone(output)
        self.assertEqual(output.tool, TOOL_EMERGENCY)
        self.assertTrue(output.arguments.location)

if __name__ == '__main__':
    unittest.main()
