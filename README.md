# Clinical Triage Router

A specialized SLM agent (≤3B parameters) that parses unstructured patient intake notes and routes them to the correct hospital system using structured JSON tool calls.

## 🎯 Challenge

Fine-tune a small language model to classify patient queries into:
- `trigger_emergency_response` - Life-threatening cases
- `schedule_urgent_consult` - Serious but non-fatal
- `routine_care_referral` - Chronic conditions, checkups

## 🚀 Quick Start

### Google Colab (Recommended)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

### Local Development
```bash
pip install -r requirements.txt
python -c "from src import config, schemas; print('Setup OK')"
```

## 📁 Structure

```
src/
├── config.py          # Hyperparameters, paths
├── schemas.py         # Pydantic tool schemas
├── data_generator.py  # Gemini-powered data gen
├── trainer.py         # Unsloth fine-tuning
├── agent.py           # Triage agent + retry logic
└── evaluator.py       # Test runner + metrics
```

## 📊 Results

| Metric | Score |
|--------|-------|
| JSON Validity | TBD |
| Routing Accuracy | TBD |
| Routing MSE | TBD |

## 🛠️ Tech Stack

- **Model**: Qwen2.5-1.5B (4-bit quantized)
- **Fine-tuning**: Unsloth + LoRA
- **Data Gen**: Gemini 1.5 Flash
- **Validation**: Pydantic v2
