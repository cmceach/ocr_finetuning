"""Training script for Nemotron Parse finetuning"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import mlflow
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

from ..utils.config_loader import ConfigLoader
from ..data.nemotron_converter import (
    NemotronDataConverter,
    NemotronDataset,
    NemotronDataCollator,
    load_nemotron_dataset,
)
from .nemotron_model import (
    NEMOTRON_PARSE_MODEL_ID,
    load_nemotron_model,
    get_nemotron_processor,
    prepare_model_for_training,
    save_nemotron_model,
    get_quantization_config,
    get_lora_config,
    get_model_memory_footprint,
)


@dataclass
class NemotronTrainingConfig:
    """Configuration for Nemotron Parse finetuning"""

    # Model settings
    model_id: str = NEMOTRON_PARSE_MODEL_ID
    trust_remote_code: bool = True

    # Training mode
    use_lora: bool = True
    use_qlora: bool = False

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list] = None

    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Sequence settings
    max_length: int = 4096

    # Precision settings
    bf16: bool = True
    fp16: bool = False

    # Memory optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # Evaluation settings
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10

    # Early stopping
    early_stopping_patience: int = 3

    # Output settings
    output_dir: str = "./trained_models/nemotron"
    save_total_limit: int = 3
    save_full_model: bool = False

    @classmethod
    def from_config(cls, config: ConfigLoader) -> "NemotronTrainingConfig":
        """Create config from ConfigLoader instance"""
        nemotron_config = config.get("nemotron", {})
        training_config = config.get("training", {})

        return cls(
            model_id=nemotron_config.get("model_id", NEMOTRON_PARSE_MODEL_ID),
            use_lora=nemotron_config.get("use_lora", True),
            use_qlora=nemotron_config.get("use_qlora", False),
            lora_r=nemotron_config.get("lora_r", 16),
            lora_alpha=nemotron_config.get("lora_alpha", 32),
            lora_dropout=nemotron_config.get("lora_dropout", 0.05),
            lora_target_modules=nemotron_config.get("lora_target_modules"),
            learning_rate=nemotron_config.get("learning_rate", 2e-5),
            batch_size=nemotron_config.get("batch_size", 4),
            gradient_accumulation_steps=nemotron_config.get("gradient_accumulation_steps", 4),
            num_epochs=nemotron_config.get("num_epochs", 3),
            warmup_ratio=nemotron_config.get("warmup_ratio", 0.03),
            weight_decay=nemotron_config.get("weight_decay", 0.01),
            max_length=nemotron_config.get("max_length", 4096),
            bf16=nemotron_config.get("bf16", True),
            fp16=nemotron_config.get("fp16", False),
            gradient_checkpointing=nemotron_config.get("gradient_checkpointing", True),
            use_flash_attention=nemotron_config.get("use_flash_attention", True),
            eval_steps=nemotron_config.get("eval_steps", 100),
            save_steps=nemotron_config.get("save_steps", 500),
            logging_steps=nemotron_config.get("logging_steps", 10),
            early_stopping_patience=nemotron_config.get("early_stopping_patience", 3),
            output_dir=config.get("model_storage.local_path", "./trained_models") + "/nemotron",
            save_total_limit=nemotron_config.get("save_total_limit", 3),
            save_full_model=nemotron_config.get("save_full_model", False),
        )


def create_training_arguments(
    training_config: NemotronTrainingConfig,
    output_dir: Optional[str] = None,
) -> Seq2SeqTrainingArguments:
    """
    Create Seq2SeqTrainingArguments from NemotronTrainingConfig.

    Args:
        training_config: Training configuration
        output_dir: Override output directory

    Returns:
        Seq2SeqTrainingArguments instance
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir or training_config.output_dir,
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        bf16=training_config.bf16,
        fp16=training_config.fp16,
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,
        save_strategy="steps",
        save_steps=training_config.save_steps,
        logging_steps=training_config.logging_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=training_config.max_length,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else ["tensorboard"],
        run_name="nemotron-parse-finetuning",
    )


def compute_metrics(eval_preds, processor: Optional[Any] = None) -> Dict[str, float]:
    """
    Compute evaluation metrics for Nemotron Parse.

    Args:
        eval_preds: Evaluation predictions from trainer (predictions, labels)
        processor: Optional processor for decoding (will be set by trainer)

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_preds

    metrics = {}

    if predictions is not None and labels is not None:
        # Decode predictions and labels if processor available
        if processor is not None:
            # Decode predictions (skip padding tokens)
            decoded_preds = processor.batch_decode(
                predictions, skip_special_tokens=True
            )
            decoded_labels = processor.batch_decode(
                labels, skip_special_tokens=True
            )

            # Calculate text similarity metrics
            try:
                from ..evaluation.evaluate_ocr import calculate_text_similarity
            except ImportError:
                # Fallback if evaluation module not available
                def calculate_text_similarity(pred, label):
                    return {"char_accuracy": 0.0, "word_accuracy": 0.0, "exact_match": 0.0}

            char_accuracies = []
            word_accuracies = []
            exact_matches = []

            for pred, label in zip(decoded_preds, decoded_labels):
                # Skip empty predictions/labels
                if not pred.strip() and not label.strip():
                    continue

                similarity = calculate_text_similarity(pred.strip(), label.strip())
                char_accuracies.append(similarity["char_accuracy"])
                word_accuracies.append(similarity["word_accuracy"])
                exact_matches.append(similarity["exact_match"])

            if char_accuracies:
                metrics["eval_char_accuracy"] = sum(char_accuracies) / len(char_accuracies)
                metrics["eval_word_accuracy"] = sum(word_accuracies) / len(word_accuracies)
                metrics["eval_exact_match_rate"] = sum(exact_matches) / len(exact_matches)

        # Token-level metrics
        metrics["eval_num_samples"] = len(predictions)

        # Calculate perplexity if logits available
        # (This would require modifying the trainer to return logits)

    return metrics


def train_nemotron_model(
    config: Optional[ConfigLoader] = None,
    training_config: Optional[NemotronTrainingConfig] = None,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    image_base_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    """
    Train/finetune Nemotron Parse model.

    Args:
        config: Configuration loader instance
        training_config: Training configuration (created from config if None)
        train_data_path: Path to training data (Label Studio JSON or JSONL)
        val_data_path: Optional path to validation data
        image_base_dir: Base directory for resolving image paths
        output_dir: Output directory for saved model
        resume_from_checkpoint: Path to checkpoint to resume from

    Returns:
        Path to saved model
    """
    # Setup configuration
    if config is None:
        config = ConfigLoader()

    if training_config is None:
        training_config = NemotronTrainingConfig.from_config(config)

    if output_dir:
        training_config.output_dir = output_dir

    # Create output directory
    output_path = Path(training_config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup MLflow logging if enabled
    mlflow_config = config.get("logging.mlflow", {})
    if mlflow_config.get("enabled"):
        tracking_uri = mlflow_config.get("tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_config.get("experiment_name", "nemotron-finetuning"))

    print("=" * 60)
    print("Nemotron Parse Finetuning")
    print("=" * 60)

    # Load processor
    print(f"\nLoading processor from {training_config.model_id}...")
    processor = get_nemotron_processor(
        model_id=training_config.model_id,
        trust_remote_code=training_config.trust_remote_code,
    )

    # Setup quantization config for QLoRA
    quantization_config = None
    if training_config.use_qlora:
        print("Using QLoRA (4-bit quantization)")
        quantization_config = get_quantization_config()

    # Load model
    print(f"\nLoading model from {training_config.model_id}...")
    model = load_nemotron_model(
        model_id=training_config.model_id,
        trust_remote_code=training_config.trust_remote_code,
        use_flash_attention=training_config.use_flash_attention,
        quantization_config=quantization_config,
        gradient_checkpointing=training_config.gradient_checkpointing,
    )

    # Prepare model for training (apply LoRA if enabled)
    lora_config = None
    if training_config.use_lora or training_config.use_qlora:
        print("\nApplying LoRA configuration...")
        lora_config = get_lora_config(
            r=training_config.lora_r,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            target_modules=training_config.lora_target_modules,
        )

    model = prepare_model_for_training(
        model=model,
        use_lora=training_config.use_lora or training_config.use_qlora,
        lora_config=lora_config,
        quantization_config=quantization_config,
    )

    # Get memory footprint
    memory_info = get_model_memory_footprint(model)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {memory_info['total_params']:,}")
    print(f"  Trainable parameters: {memory_info['trainable_params']:,}")
    print(f"  Trainable ratio: {memory_info['trainable_ratio']:.2%}")

    # Load datasets
    print(f"\nLoading datasets...")
    if train_data_path is None:
        train_data_path = config.get("data.processed_data_path", "./data/processed_data/nemotron_train.json")

    converter = NemotronDataConverter(image_base_dir=image_base_dir)

    # Load training data
    if train_data_path.endswith(".jsonl"):
        train_samples = converter.convert_from_jsonl(train_data_path)
    else:
        train_samples = converter.convert_from_label_studio(train_data_path)

    # Load or split validation data
    if val_data_path:
        if val_data_path.endswith(".jsonl"):
            val_samples = converter.convert_from_jsonl(val_data_path)
        else:
            val_samples = converter.convert_from_label_studio(val_data_path)
    else:
        # Split training data
        from ..data.nemotron_converter import create_train_val_split
        val_ratio = config.get("data.val_split", 0.1)
        train_samples, val_samples = create_train_val_split(train_samples, val_ratio=val_ratio)

    print(f"  Training samples: {len(train_samples)}")
    print(f"  Validation samples: {len(val_samples)}")

    # Create datasets
    train_dataset = NemotronDataset(
        samples=train_samples,
        processor=processor,
        max_length=training_config.max_length,
    )
    val_dataset = NemotronDataset(
        samples=val_samples,
        processor=processor,
        max_length=training_config.max_length,
    )

    # Create data collator
    data_collator = NemotronDataCollator(
        processor=processor,
        max_length=training_config.max_length,
    )

    # Create training arguments
    training_args = create_training_arguments(
        training_config=training_config,
        output_dir=str(output_path),
    )

    # Create compute_metrics function with processor closure
    def compute_metrics_with_processor(eval_preds):
        return compute_metrics(eval_preds, processor=processor)

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor,
        compute_metrics=compute_metrics_with_processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.early_stopping_patience
            )
        ],
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()

    print("\nTraining completed!")

    # Save final model
    print("\nSaving model...")
    final_model_path = save_nemotron_model(
        model=model,
        output_dir=str(output_path / "final"),
        processor=processor,
        save_full_model=training_config.save_full_model,
    )

    # Save training config
    config_path = output_path / "training_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_id": training_config.model_id,
            "use_lora": training_config.use_lora,
            "use_qlora": training_config.use_qlora,
            "lora_r": training_config.lora_r,
            "lora_alpha": training_config.lora_alpha,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.batch_size,
            "num_epochs": training_config.num_epochs,
            "max_length": training_config.max_length,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
        }, f, indent=2)

    print(f"\nModel saved to: {final_model_path}")
    print(f"Training config saved to: {config_path}")

    return final_model_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Nemotron Parse model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--client-config", type=str, help="Path to client config file")
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data")
    parser.add_argument("--val-data", type=str, help="Path to validation data")
    parser.add_argument("--image-base-dir", type=str, help="Base directory for images")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")

    # Training overrides
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetuning)")
    parser.add_argument("--qlora", action="store_true", help="Use QLoRA (4-bit)")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")

    args = parser.parse_args()

    # Load config
    config = ConfigLoader(args.config, args.client_config)

    # Create training config with overrides
    training_config = NemotronTrainingConfig.from_config(config)

    if args.no_lora:
        training_config.use_lora = False
    if args.qlora:
        training_config.use_qlora = True
        training_config.use_lora = True
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    if args.batch_size:
        training_config.batch_size = args.batch_size
    if args.epochs:
        training_config.num_epochs = args.epochs

    # Train
    train_nemotron_model(
        config=config,
        training_config=training_config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        image_base_dir=args.image_base_dir,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
    )

