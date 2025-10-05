"""
Model and tokenizer loading with quantization and LoRA setup.
"""
import logging
import torch
from typing import Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import setup_chat_format


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Loads and configures models for GRPO training.
    """
    
    def __init__(self, config):
        """
        Initialize model loader.
        
        Args:
            config: ModelConfig object
        """
        self.config = config
    
    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """
        Create BitsAndBytes quantization config.
        
        Returns:
            BitsAndBytesConfig for 4-bit quantization
        """
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(
                torch, 
                self.config.bnb_4bit_compute_dtype
            )
        )
    
    def _get_lora_config(self) -> LoraConfig:
        """
        Create LoRA config.
        
        Returns:
            LoraConfig for PEFT
        """
        return LoraConfig(
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            r=self.config.lora_r,
            bias=self.config.lora_bias,
            target_modules=self.config.lora_target_modules,
            task_type=self.config.lora_task_type,
        )
    
    def _check_flash_attention(self) -> str:
        """
        Check if Flash Attention is available.
        
        Returns:
            Attention implementation to use
        """
        if not self.config.use_flash_attention:
            return "sdpa"
        
        try:
            import flash_attn
            logger.info("✓ Flash Attention 2 available")
            return "flash_attention_2"
        except ImportError:
            logger.warning("⚠ Flash Attention 2 not available, using SDPA")
            return "sdpa"
    
    def load_model_and_tokenizer(
        self,
        apply_chat_format: bool = True
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with quantization.
        
        Args:
            apply_chat_format: Whether to setup chat format
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Get configs
        bnb_config = self._get_bnb_config()
        attn_implementation = self._check_flash_attention()
        
        # Determine dtype
        torch_dtype = getattr(torch, self.config.torch_dtype)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            device_map=self.config.device_map,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        
        logger.info("✓ Model loaded")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=True,
        )
        
        # Configure tokenizer
        tokenizer.padding_side = self.config.padding_side
        tokenizer.model_max_length = self.config.model_max_length
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token = eos_token")
        
        logger.info("✓ Tokenizer loaded")
        
        # Apply chat format if requested
        if apply_chat_format:
            model, tokenizer = setup_chat_format(model, tokenizer)
            logger.info("✓ Chat format applied")
        
        # Suppress warnings
        if hasattr(model, 'warnings_issued'):
            model.warnings_issued["estimate_tokens"] = True
        
        return model, tokenizer
    
    def apply_lora(
        self,
        model: AutoModelForCausalLM
    ) -> AutoModelForCausalLM:
        """
        Apply LoRA adapters to model.
        
        Args:
            model: Base model
        
        Returns:
            Model with LoRA adapters
        """
        logger.info("Applying LoRA adapters...")
        
        peft_config = self._get_lora_config()
        model = get_peft_model(model, peft_config)
        
        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(
            f"Trainable params: {trainable_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )
        
        return model
    
    def get_lora_config(self) -> LoraConfig:
        """
        Get LoRA configuration.
        
        Returns:
            LoraConfig object
        """
        return self._get_lora_config()


def load_base_model_for_merge(
    model_id: str,
    config
) -> AutoModelForCausalLM:
    """
    Load base model for merging with LoRA adapters.
    
    Args:
        model_id: Model identifier
        config: ModelConfig
    
    Returns:
        Base model without LoRA
    """
    logger.info(f"Loading base model for merge: {model_id}")
    
    # Get quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype)
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=getattr(torch, config.torch_dtype),
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    
    logger.info("✓ Base model loaded for merge")
    return model


def print_model_info(model: AutoModelForCausalLM):
    """
    Print model information.
    
    Args:
        model: Model to inspect
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    print("\n" + "=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print(f"Model dtype: {model.dtype}")
    print(f"Device: {model.device}")
    print("=" * 50 + "\n")
