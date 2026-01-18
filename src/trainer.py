"""Unsloth-based fine-tuning with 4-bit quantization for T4 GPU."""

from pathlib import Path
from typing import Optional

from . import config
from . import prompts
from .data_generator import load_training_data, format_for_training


def load_model_and_tokenizer():
    """
    Load pre-quantized model using Unsloth.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "Please install unsloth: pip install unsloth"
        )
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.MODEL_NAME,
        max_seq_length=config.MAX_SEQ_LENGTH,
        load_in_4bit=config.LOAD_IN_4BIT,
        dtype=None,  # Auto-detect
    )
    
    return model, tokenizer


def prepare_model_for_training(model):
    """
    Add LoRA adapters for efficient fine-tuning.
    
    Args:
        model: Base model from load_model_and_tokenizer
        
    Returns:
        Model with LoRA adapters
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Please install unsloth")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.SEED,
    )
    
    return model


def create_prompt_template(tokenizer):
    """
    Create the prompt template for training.
    
    Returns:
        Formatted prompt template string
    """
    return prompts.PROMPT_TEMPLATE


def prepare_dataset(tokenizer, train_data_path: Optional[Path] = None):
    """
    Prepare the dataset for training.
    
    Args:
        tokenizer: Model tokenizer
        train_data_path: Path to training JSONL file
        
    Returns:
        HuggingFace Dataset ready for training
    """
    from datasets import Dataset
    
    raw_data = load_training_data(train_data_path)

    # Balance classes to improve routing accuracy on small datasets
    from collections import defaultdict
    import random

    by_tool = defaultdict(list)
    for example in raw_data:
        by_tool[example.get("tool")].append(example)

    max_count = max(len(items) for items in by_tool.values()) if by_tool else 0
    balanced = []
    rng = random.Random(config.SEED)
    for items in by_tool.values():
        if not items:
            continue
        balanced.extend(items)
        if len(items) < max_count:
            balanced.extend(rng.choices(items, k=max_count - len(items)))

    rng.shuffle(balanced)
    formatted_data = [format_for_training(ex) for ex in balanced]
    
    prompt_template = create_prompt_template(tokenizer)
    
    def format_sample(sample):
        text = prompt_template.format(
            instruction=sample["instruction"],
            output=sample["output"],
        )
        return {"text": text}
    
    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.map(format_sample)
    
    return dataset


def train(
    model,
    tokenizer,
    dataset,
    output_dir: Optional[Path] = None,
):
    """
    Run fine-tuning using SFTTrainer.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Model tokenizer
        dataset: Prepared dataset
        output_dir: Directory for checkpoints
        
    Returns:
        Trained model
    """
    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
    except ImportError:
        raise ImportError("Please install trl and transformers")
    
    output_dir = output_dir or config.OUTPUT_DIR
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        logging_steps=10,
        save_strategy="epoch",
        seed=config.SEED,
        fp16=True,  # T4 doesn't support bf16, needs Ampere+ GPU
        optim="adamw_8bit",
        max_steps=config.MAX_STEPS,
        dataloader_num_workers=2,  # Parallel data loading
        group_by_length=True,  # Batch similar-length sequences together
        lr_scheduler_type="cosine",  # Smooth learning rate decay
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=False,  # Disable packing - important for structured output learning
    )
    
    # Train
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    
    print(f"Model saved to {output_dir / 'final_model'}")
    
    return model


def run_training_pipeline(train_data_path: Optional[Path] = None):
    """
    Complete training pipeline: load → prepare → train.
    
    Args:
        train_data_path: Path to training data
        
    Returns:
        Tuple of (trained_model, tokenizer)
    """
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("Adding LoRA adapters...")
    model = prepare_model_for_training(model)
    
    print("Preparing dataset...")
    dataset = prepare_dataset(tokenizer, train_data_path)
    print(f"Dataset size: {len(dataset)} examples")
    
    print("Starting training...")
    model = train(model, tokenizer, dataset)
    
    print("Training complete!")
    return model, tokenizer
