# Clinical Triage Router - Take-Home Challenge

## 📧 Original Email from Bharat @ Indhic AI

---

**Subject**: Consultant AI Engineer Role - Technical Challenge

Hi Adarsh,

Thanks for your interest in the Consultant AI Engineer role.

We don't believe in LeetCode-style interviews. Instead, we want to see how you handle the specific engineering challenges: **Small Language Models (SLMs)**, **Agentic workflows**, and **Rigorous Evaluation**.

We have designed a short, high-impact challenge to evaluate your skills in these areas.

---

## 🎯 The Challenge: "The Clinical Triage Router"

### Objective
Build a specialized SLM agent (Model size < 3B) that parses unstructured patient intake notes and routes them to the correct hospital system using structured JSON tool calls.

### The Constraint
This entire task is designed to be runnable on the **Google Colab Free Tier (T4 GPU)**. You are not expected to spend any money on compute or APIs.

---

## 📋 1. The Schema (Ground Truth)

Your agent must classify patient queries into **exactly one** of these 3 tools. Please adhere strictly to this schema so we can evaluate your solution.

### Tool 1: `trigger_emergency_response`
- **Use Case**: Life-threatening (heart attack, stroke, trauma)
- **Arguments**: 
  ```json
  {
    "location": "string",
    "severity": "CRITICAL"
  }
  ```

### Tool 2: `schedule_urgent_consult`
- **Use Case**: Serious but non-fatal (high fever, fractures)
- **Arguments**: 
  ```json
  {
    "department": "string",
    "symptoms": ["string", "string", ...]
  }
  ```

### Tool 3: `routine_care_referral`
- **Use Case**: Chronic conditions, refills, checkups
- **Arguments**: 
  ```json
  {
    "type": "string",
    "specialty": "string"
  }
  ```

---

## 📝 2. The Assignment

Submit a **single Google Colab Notebook** that executes the following 3 phases. The notebook should be **pre-run** so we can see the outputs.

### Phase 1: Synthetic Data & Fine-Tuning
1. **Data Gen**: Generate 50–100 training examples mapping natural language to the JSON schema above
2. **Fine-Tune**: Use Unsloth (or similar) to fine-tune a model ≤ 3B parameters (e.g., `Llama-3.2-1B` or `Qwen-2.5-1.5B`) to output this specific JSON structure
3. **Optimization**: Use 4-bit quantization to ensure it fits in Colab memory

### Phase 2: The Agentic Loop
1. Create a function `run_triage_agent(query)` that inferences your model
2. **Self-Correction**: If the model outputs invalid JSON or hallucinates a tool name, your code must catch the error and retry or handle it gracefully
3. **Mock Execution**: Return a dummy confirmation string (e.g., `[System] Ambulance Dispatched`)

### Phase 3: The Evaluation
1. Create a held-out test set of **10 challenging examples**
2. Run your model against them and print a mini-report:
   - **JSON Validity**: % of parsable outputs
   - **Routing Accuracy**: % of correct tool selections

---

## 🛠️ 3. Resources & "Zero Cost" Guide

### Compute
- **GPU**: Use Google Colab Free Tier (Runtime > Change Runtime Type > T4 GPU)

### Data Generation
- **Recommended**: Gemini 1.5 Flash Free Tier (via Google AI Studio key) OR Hugging Face Inference API

### Evaluation
- Simple Python string matching or heuristic grading is sufficient

---

## 📤 Submission Requirements

### Deliverable
Reply to this email with your **Google Colab Link** (ensure permissions are set to "Anyone with the link can view")

### Deadline
We respect your time. Take **[2-3 Days / the weekend]** to complete this. 

We are looking for **quality of implementation** (clean code, memory management, robust error handling) over speed.

---

## 🎯 Evaluation Criteria (Implied)

Based on the challenge description, they're evaluating:

1. **SLM Expertise**: Can you fine-tune a small model efficiently?
2. **Agentic Workflows**: Self-correction, error handling, retry logic
3. **Rigorous Evaluation**: Clear metrics, honest assessment
4. **Production Quality**: Clean code, memory management, robustness
5. **Zero-Cost Execution**: Works on free tier without external dependencies

---

## ✅ Success Metrics

### Must Have
- ✅ Runs completely on Colab Free Tier (T4 GPU)
- ✅ Model size ≤ 3B parameters
- ✅ Outputs valid JSON for tool calls
- ✅ Handles 3 tools exactly as specified in schema
- ✅ Includes self-correction/error handling
- ✅ Evaluation on 10 test cases with metrics
- ✅ Pre-run notebook with visible outputs

### Should Have
- Clean, commented, production-grade code
- Efficient memory usage (4-bit quantization)
- Robust JSON parsing and validation
- Clear separation of training/validation/test data
- Reproducible results (seed setting)

### Nice to Have
- Creative approach to synthetic data generation
- Advanced error recovery strategies
- Additional evaluation metrics beyond required
- Documentation and explanations in markdown cells
- Modular code structure

---

## 🚩 Red Flags to Avoid

- ❌ Notebook crashes due to memory issues
- ❌ Using paid APIs (OpenAI, Claude, etc.)
- ❌ Hardcoded outputs / faking results
- ❌ Overly complex solutions that don't run reliably
- ❌ Poor error handling (crashes on edge cases)
- ❌ Unclear or messy code
- ❌ Not adhering to the exact JSON schema

---

Looking forward to seeing your code.

Best,  
**Bharat**  
Indhic AI

---

## 📌 Key Takeaways

This challenge tests:
1. **Practical ML engineering** over theoretical knowledge
2. **Resource constraints** (free tier, small models)
3. **Production mindset** (error handling, validation, clean code)
4. **Domain adaptation** (medical use case)
5. **Evaluation rigor** (honest metrics, edge cases)

The emphasis on "quality over speed" suggests they value thoughtful implementation and robust engineering practices.