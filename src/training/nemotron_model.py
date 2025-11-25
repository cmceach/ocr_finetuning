"""Nemotron Parse model loading and configuration utilities"""

import os
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)

# Optional PEFT imports
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# Model constants
NEMOTRON_PARSE_MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-v1.1"


def get_nemotron_processor(
    model_id: str = NEMOTRON_PARSE_MODEL_ID,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load the Nemotron Parse processor.

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code

    Returns:
        Model processor
    """
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return processor


def get_nemotron_tokenizer(
    model_id: str = NEMOTRON_PARSE_MODEL_ID,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load the Nemotron Parse tokenizer (mBART-based).

    Args:
        model_id: HuggingFace model ID or local path
        trust_remote_code: Whether to trust remote code

    Returns:
        Model tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create quantization config for QLoRA training.

    Args:
        load_in_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype for 4-bit
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        bnb_4bit_use_double_quant: Whether to use double quantization

    Returns:
        BitsAndBytesConfig instance
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype, torch.bfloat16)

    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    bias: str = "none",
    task_type: str = "SEQ_2_SEQ_LM",
) -> "LoraConfig":
    """
    Create LoRA configuration for PEFT training.

    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Bias handling ("none", "all", "lora_only")
        task_type: Task type for PEFT

    Returns:
        LoraConfig instance
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    # Default target modules for Nemotron Parse (mBART decoder + ViT encoder)
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # For attention layers
            "query",
            "key",
            "value",
            "dense",
        ]

    # Map task type string to enum
    task_type_map = {
        "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        "CAUSAL_LM": TaskType.CAUSAL_LM,
        "FEATURE_EXTRACTION": TaskType.FEATURE_EXTRACTION,
    }
    peft_task_type = task_type_map.get(task_type, TaskType.SEQ_2_SEQ_LM)

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=peft_task_type,
    )


def load_nemotron_model(
    model_id: str = NEMOTRON_PARSE_MODEL_ID,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    use_flash_attention: bool = True,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    gradient_checkpointing: bool = False,
) -> Any:
    """
    Load Nemotron Parse model.

    Args:
        model_id: HuggingFace model ID or local path
        device_map: Device mapping strategy
        torch_dtype: Model dtype (None for auto)
        trust_remote_code: Whether to trust remote code
        use_flash_attention: Whether to use flash attention 2
        quantization_config: Optional quantization config for QLoRA
        gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        Loaded model
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    attn_implementation = "flash_attention_2" if use_flash_attention else "eager"

    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": attn_implementation,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # Load the model
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        **model_kwargs,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    return model


def prepare_model_for_training(
    model: Any,
    use_lora: bool = True,
    lora_config: Optional["LoraConfig"] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
) -> Any:
    """
    Prepare model for training with optional LoRA.

    Args:
        model: Base model
        use_lora: Whether to apply LoRA
        lora_config: LoRA configuration (created if None and use_lora=True)
        quantization_config: Quantization config (for QLoRA preparation)

    Returns:
        Training-ready model
    """
    if not use_lora:
        # Full finetuning - just enable training
        model.train()
        return model

    if not PEFT_AVAILABLE:
        raise ImportError("PEFT is required for LoRA. Install with: pip install peft")

    # Prepare for k-bit training if quantized
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Create default LoRA config if not provided
    if lora_config is None:
        lora_config = get_lora_config()

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def save_nemotron_model(
    model: Any,
    output_dir: str,
    processor: Optional[Any] = None,
    save_full_model: bool = False,
) -> str:
    """
    Save finetuned Nemotron model.

    Args:
        model: Finetuned model (base or PEFT)
        output_dir: Output directory
        processor: Optional processor to save
        save_full_model: Whether to merge and save full model (for PEFT)

    Returns:
        Path to saved model
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if model is a PEFT model
    is_peft_model = PEFT_AVAILABLE and isinstance(model, PeftModel)

    if is_peft_model:
        if save_full_model:
            # Merge LoRA weights and save full model
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(output_path / "merged")
            save_path = str(output_path / "merged")
        else:
            # Save only adapter weights
            model.save_pretrained(output_path / "adapter")
            save_path = str(output_path / "adapter")
    else:
        # Save full model
        model.save_pretrained(output_path)
        save_path = str(output_path)

    # Save processor if provided
    if processor is not None:
        processor.save_pretrained(output_path)

    return save_path


def load_finetuned_nemotron(
    base_model_id: str = NEMOTRON_PARSE_MODEL_ID,
    adapter_path: Optional[str] = None,
    merged_model_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load a finetuned Nemotron model (either adapter or merged).

    Args:
        base_model_id: Base model ID (for adapter loading)
        adapter_path: Path to LoRA adapter weights
        merged_model_path: Path to merged model
        device_map: Device mapping strategy
        torch_dtype: Model dtype
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, processor)
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    processor = get_nemotron_processor(
        merged_model_path or base_model_id,
        trust_remote_code=trust_remote_code,
    )

    if merged_model_path:
        # Load merged model directly
        model = AutoModelForVision2Seq.from_pretrained(
            merged_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
    elif adapter_path:
        # Load base model and apply adapter
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required to load adapter. Install with: pip install peft")

        base_model = load_nemotron_model(
            model_id=base_model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        # Load base model
        model = load_nemotron_model(
            model_id=base_model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    return model, processor


def get_model_memory_footprint(model: Any) -> Dict[str, Any]:
    """
    Get model memory footprint information.

    Args:
        model: Model to analyze

    Returns:
        Dictionary with memory information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory in bytes (assuming float32 for simplicity)
    param_bytes = total_params * 4  # float32
    trainable_bytes = trainable_params * 4

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "estimated_total_memory_gb": param_bytes / (1024 ** 3),
        "estimated_trainable_memory_gb": trainable_bytes / (1024 ** 3),
    }

