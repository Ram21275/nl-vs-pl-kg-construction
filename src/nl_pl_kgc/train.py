from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .data import Example, load_examples
from .experiment import choose_demonstrations
from .metrics import strict_micro_metrics
from .parsing import parse_output
from .prompts import PromptFormat, build_prompt, training_text


def _load_training_stack():
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install with pip install -e '.[train]'."
        ) from exc
    return {
        "torch": torch,
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }


def train_adapter(config: dict[str, Any], prompt_format: PromptFormat) -> Path:
    """Train one paper-style LoRA adapter for natural or code prompts."""
    stack = _load_training_stack()
    torch = stack["torch"]
    dataset_dir = Path(config["dataset_dir"])
    train_examples = load_examples(dataset_dir / "train_triples.json")
    tokenizer = stack["AutoTokenizer"].from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for index, example in enumerate(train_examples):
        demonstrations = choose_demonstrations(
            train_examples, int(config["n_icl_examples"]), int(config["seed"]) + index, example
        )
        rows.append(
            {
                "text": training_text(tokenizer, example, prompt_format, demonstrations),
            }
        )
    dataset = stack["Dataset"].from_list(rows)

    quantization = None
    if config.get("load_in_4bit", True):
        quantization = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )
    model = stack["AutoModelForCausalLM"].from_pretrained(
        config["model_name"],
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if quantization is not None:
        model = stack["prepare_model_for_kbit_training"](model)

    peft_config = stack["LoraConfig"](
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["target_modules"]),
    )
    destination = Path(config["output_dir"]) / prompt_format
    destination.mkdir(parents=True, exist_ok=True)
    args = stack["SFTConfig"](
        output_dir=str(destination),
        max_length=int(config["max_length"]),
        max_steps=int(config["train_steps"]),
        per_device_train_batch_size=int(config["batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=float(config["learning_rate"]),
        warmup_steps=5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=int(config["seed"]),
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit" if quantization is not None else "adamw_torch",
        dataset_text_field="text",
    )
    trainer = stack["SFTTrainer"](
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.model.save_pretrained(destination / "adapter")
    tokenizer.save_pretrained(destination / "adapter")
    return destination / "adapter"


def evaluate_model(config: dict[str, Any], prompt_format: PromptFormat, adapter_path: str | Path):
    """Evaluate an adapter with the paper's strict full-triple micro metric."""
    stack = _load_training_stack()
    torch = stack["torch"]
    from peft import PeftModel

    dataset_dir = Path(config["dataset_dir"])
    train_examples = load_examples(dataset_dir / "train_triples.json")
    test_examples = load_examples(dataset_dir / "test_triples.json")
    limit = int(config.get("eval_limit", 0))
    if limit:
        test_examples = test_examples[:limit]
    demonstrations = choose_demonstrations(
        train_examples, int(config["n_icl_examples"]), int(config["seed"])
    )

    tokenizer = stack["AutoTokenizer"].from_pretrained(adapter_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = stack["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    base = stack["AutoModelForCausalLM"].from_pretrained(
        config["model_name"], quantization_config=quantization, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    predictions = []
    rows = []
    for example in test_examples:
        pair = build_prompt(example.text, prompt_format, demonstrations)
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": pair.system},
                {"role": "user", "content": pair.user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=config["max_length"])
        inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=int(config["max_new_tokens"]),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs["input_ids"].shape[1] :]
        raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed = parse_output(raw, prompt_format)
        predictions.append(parsed)
        rows.append(
            {
                "text": example.text,
                "gold": [list(t) for t in example.triples],
                "raw_output": raw,
                "prediction": [list(t) for t in parsed],
            }
        )
    metrics = strict_micro_metrics([e.triples for e in test_examples], predictions)
    destination = Path(config["output_dir"]) / prompt_format
    (destination / "predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (destination / "metrics.json").write_text(
        json.dumps(metrics.as_dict(), indent=2), encoding="utf-8"
    )
    return metrics
