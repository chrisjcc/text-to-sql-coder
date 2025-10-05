"""
GRPO trainer setup and execution.
"""
import logging
import os
from typing import Optional
import verifiers as vf
from transformers import AutoTokenizer
from peft import PeftModel

from ..models.model_loader import load_base_model_for_merge
from .environment import SQLEnvironment


logger = logging.getLogger(__name__)


class GRPOTrainerWrapper:
    """
    Wrapper for verifiers GRPOTrainer with lifecycle management.
    """
    
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        environment: vf.Environment,
        grpo_config,
        peft_config=None
    ):
        """
        Initialize GRPO trainer wrapper.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            environment: Verifiers environment
            grpo_config: GRPOConfig object
            peft_config: Optional PEFT config
        """
        self.model = model
        self.tokenizer = tokenizer
        self.environment = environment
        self.grpo_config = grpo_config
        self.peft_config = peft_config
        self.trainer: Optional[vf.GRPOTrainer] = None
    
    def create_trainer(self) -> vf.GRPOTrainer:
        """
        Create GRPOTrainer instance.
        
        Returns:
            Configured GRPOTrainer
        """
        logger.info("Creating GRPOTrainer...")
        
        # Create training arguments using grpo_defaults
        args = vf.grpo_defaults(
            run_name=self.grpo_config.run_name,
            output_dir=self.grpo_config.output_dir,
            num_train_epochs=self.grpo_config.num_train_epochs,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            learning_rate=self.grpo_config.learning_rate,
            max_seq_len=self.grpo_config.max_seq_len,
            max_prompt_length=self.grpo_config.max_prompt_length,
            max_tokens=self.grpo_config.max_tokens,
            num_generations=self.grpo_config.num_generations,
            beta=self.grpo_config.beta,
            optim=self.grpo_config.optim,
            max_grad_norm=self.grpo_config.max_grad_norm,
            warmup_ratio=self.grpo_config.warmup_ratio,
            lr_scheduler_type=self.grpo_config.lr_scheduler_type,
            bf16=self.grpo_config.bf16,
            tf32=self.grpo_config.tf32,
            gradient_checkpointing=self.grpo_config.gradient_checkpointing,
            save_strategy=self.grpo_config.save_strategy,
            logging_steps=self.grpo_config.logging_steps,
            push_to_hub=self.grpo_config.push_to_hub,
            report_to=self.grpo_config.report_to,
            sync_ref_model=self.grpo_config.sync_ref_model,
        )
        
        # Add ref_model_sync_steps if specified
        if self.grpo_config.ref_model_sync_steps:
            args.ref_model_sync_steps = self.grpo_config.ref_model_sync_steps
        
        # Create trainer
        self.trainer = vf.GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,  # Note: processing_class, not tokenizer
            env=self.environment,
            args=args,
            peft_config=self.peft_config,
        )
        
        logger.info("✓ GRPOTrainer created")
        return self.trainer
    
    def train(self, resume_from_checkpoint: bool = True):
        """
        Execute training.
        
        Args:
            resume_from_checkpoint: Whether to resume from checkpoint
        """
        if self.trainer is None:
            self.create_trainer()
        
        logger.info("Starting GRPO training...")
        logger.info(f"Output directory: {self.grpo_config.output_dir}")
        logger.info(f"Training epochs: {self.grpo_config.num_train_epochs}")
        logger.info(f"Batch size: {self.grpo_config.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {self.grpo_config.gradient_accumulation_steps}")
        logger.info(f"Num generations: {self.grpo_config.num_generations}")
        
        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info("✓ Training completed")
    
    def save_model(self):
        """Save trained model adapters."""
        if self.trainer is None:
            logger.error("Trainer not created")
            return
        
        logger.info(f"Saving model to {self.grpo_config.output_dir}...")
        self.trainer.save_model()
        logger.info("✓ Model saved")
    
    def merge_and_save(self, model_config):
        """
        Merge LoRA adapters with base model and save.
        
        Args:
            model_config: ModelConfig object
        """
        import torch
        import gc
        
        logger.info("Merging LoRA adapters with base model...")
        
        output_dir = self.grpo_config.output_dir
        merged_dir = os.path.join(output_dir, "merged")
        
        # Free memory
        logger.info("Freeing GPU memory...")
        del self.model
        del self.trainer
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load base model
        logger.info("Loading base model...")
        base_model = load_base_model_for_merge(
            model_config.model_id,
            model_config
        )
        
        # Load tokenizer from trained model
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        
        # Apply chat format
        from trl import setup_chat_format
        base_model, tokenizer = setup_chat_format(base_model, tokenizer)
        
        # Load PEFT adapter
        logger.info("Loading PEFT adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            output_dir,
            torch_dtype=torch.bfloat16,
        )
        
        # Merge
        logger.info("Merging adapters...")
        merged_model = peft_model.merge_and_unload()
        
        # Save
        logger.info(f"Saving merged model to {merged_dir}...")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        
        logger.info("✓ Merged model saved")
        
        # Cleanup
        del base_model, peft_model, merged_model
        torch.cuda.empty_cache()
        gc.collect()


def create_trainer(
    model,
    tokenizer: AutoTokenizer,
    environment: vf.Environment,
    grpo_config,
    peft_config=None
) -> GRPOTrainerWrapper:
    """
    Convenience function to create trainer.
    
    Args:
        model: Model to train
        tokenizer: Tokenizer
        environment: Training environment
        grpo_config: GRPOConfig
        peft_config: Optional PEFT config
    
    Returns:
        Configured trainer wrapper
    """
    wrapper = GRPOTrainerWrapper(
        model=model,
        tokenizer=tokenizer,
        environment=environment,
        grpo_config=grpo_config,
        peft_config=peft_config
    )
    
    wrapper.create_trainer()
    return wrapper
