"""Evaluation module for testing and metrics."""

import json
from pathlib import Path
from typing import Optional

from . import config
from .agent import TriageAgent
from .schemas import ToolName


def load_test_data(path: Optional[Path] = None) -> list[dict]:
    """Load test data from JSONL file."""
    path = path or config.TEST_DATA_PATH
    examples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    return examples


def evaluate(
    agent: TriageAgent,
    test_data: list[dict],
    verbose: bool = True,
) -> dict:
    """
    Evaluate the agent on test data.
    
    Args:
        agent: Initialized TriageAgent
        test_data: List of test examples with 'query' and 'tool' keys
        verbose: Print per-example results
        
    Returns:
        Dict with evaluation metrics
    """
    results = {
        "total": len(test_data),
        "json_valid": 0,
        "routing_correct": 0,
        "details": [],
    }
    
    for i, example in enumerate(test_data):
        query = example["query"]
        expected_tool = example["tool"]
        
        output, response, metadata = agent.run(query)
        
        # Check JSON validity
        json_valid = output is not None
        if json_valid:
            results["json_valid"] += 1
        
        # Check routing accuracy
        predicted_tool = output.tool if output else None
        routing_correct = predicted_tool == expected_tool
        if routing_correct:
            results["routing_correct"] += 1
        
        detail = {
            "index": i + 1,
            "query_preview": query[:80] + "..." if len(query) > 80 else query,
            "expected": expected_tool,
            "predicted": predicted_tool,
            "json_valid": json_valid,
            "routing_correct": routing_correct,
            "attempts": metadata["attempts"],
        }
        results["details"].append(detail)
        
        if verbose:
            status = "✅" if routing_correct else "❌"
            print(f"[{i+1}/{len(test_data)}] {status} Expected: {expected_tool}, Got: {predicted_tool}")
    
    # Calculate percentages
    results["json_validity_pct"] = (results["json_valid"] / results["total"]) * 100
    results["routing_accuracy_pct"] = (results["routing_correct"] / results["total"]) * 100
    
    return results


def print_report(results: dict):
    """Pretty-print evaluation report."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\n📈 Summary ({results['total']} test cases)")
    print("-" * 40)
    print(f"  JSON Validity:    {results['json_valid']}/{results['total']} ({results['json_validity_pct']:.1f}%)")
    print(f"  Routing Accuracy: {results['routing_correct']}/{results['total']} ({results['routing_accuracy_pct']:.1f}%)")
    
    # Per-tool breakdown
    tool_stats = {t.value: {"total": 0, "correct": 0} for t in ToolName}
    for detail in results["details"]:
        expected = detail["expected"]
        if expected in tool_stats:
            tool_stats[expected]["total"] += 1
            if detail["routing_correct"]:
                tool_stats[expected]["correct"] += 1
    
    print(f"\n📋 Per-Tool Breakdown")
    print("-" * 40)
    for tool, stats in tool_stats.items():
        if stats["total"] > 0:
            pct = (stats["correct"] / stats["total"]) * 100
            print(f"  {tool}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")
    
    # Error cases
    errors = [d for d in results["details"] if not d["routing_correct"]]
    if errors:
        print(f"\n⚠️ Misclassified Cases ({len(errors)})")
        print("-" * 40)
        for err in errors:
            print(f"  [{err['index']}] Expected {err['expected']}, got {err['predicted']}")
            print(f"       Query: {err['query_preview']}")
    
    print("\n" + "=" * 60)


def run_evaluation(
    model=None,
    tokenizer=None,
    test_data_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Complete evaluation pipeline.
    
    Args:
        model: Optional pre-loaded model
        tokenizer: Optional pre-loaded tokenizer
        test_data_path: Path to test JSONL
        verbose: Print results
        
    Returns:
        Evaluation results dict
    """
    agent = TriageAgent(model=model, tokenizer=tokenizer)
    
    if not agent._model_loaded:
        print("Loading model from checkpoint...")
        agent.load_model()
    
    print("Loading test data...")
    test_data = load_test_data(test_data_path)
    print(f"Loaded {len(test_data)} test examples")
    
    print("\nRunning evaluation...")
    results = evaluate(agent, test_data, verbose=verbose)
    
    print_report(results)
    
    return results
