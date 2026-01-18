"""Configuration constants, paths, and hyperparameters."""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_DATA_PATH = DATA_DIR / "train.jsonl"
TEST_DATA_PATH = DATA_DIR / "test.jsonl"

# =============================================================================
# Model Configuration
# =============================================================================
# Unsloth pre-quantized models (4-bit, fits T4 GPU)
MODEL_NAME = "unsloth/Qwen2.5-1.5B-bnb-4bit"
# Alternative: "unsloth/Llama-3.2-1B-bnb-4bit"

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# =============================================================================
# LoRA Configuration
# =============================================================================
LORA_R = 16
LORA_ALPHA = 32  # Match r for function calling tasks (research-backed)
LORA_DROPOUT = 0.0  # Favor memorization for max routing accuracy on small data
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# =============================================================================
# Training Hyperparameters
# =============================================================================
NUM_TRAIN_EPOCHS = 20  # More epochs to fit small dataset well
LEARNING_RATE = 1e-4  # Slightly higher LR to learn routing quickly
BATCH_SIZE = 8  # Moderate batch size for stability
GRADIENT_ACCUMULATION_STEPS = 1  # More updates per epoch on small data
WARMUP_STEPS = 5  # Short warmup for small datasets
MAX_STEPS = -1  # -1 = use epochs
WEIGHT_DECAY = 0.0
SEED = 42

# =============================================================================
# Data Generation
# =============================================================================
NUM_TRAINING_EXAMPLES = 200  # More training data for better accuracy
GEMINI_MODEL = "gemini-1.5-flash"
OPENAI_MODEL = "gpt-4.1"
OPENAI_REASONING_EFFORT = "medium"  # "none", "low", "medium", "high"
DATA_GEN_PROVIDER = "openai"  # "gemini" or "openai"

# =============================================================================
# Agent Configuration
# =============================================================================
MAX_RETRIES = 3
GENERATION_MAX_TOKENS = 256
GENERATION_TEMPERATURE = 0.0  # Greedy decoding for deterministic JSON output

# =============================================================================
# Tool Names (must match exactly)
# =============================================================================
TOOL_EMERGENCY = "trigger_emergency_response"
TOOL_URGENT = "schedule_urgent_consult"
TOOL_ROUTINE = "routine_care_referral"
VALID_TOOLS = [TOOL_EMERGENCY, TOOL_URGENT, TOOL_ROUTINE]
