"""
Main training script for SQL GRPO training.

Usage:
    python scripts/train.py

Environment variables (.env):
    WANDB_API_KEY: Weights & Biases API key
    HF_KEY: Hugging Face API token
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import Config
from src.data.dataset_loader import SQLDatasetLoader, DatasetPreprocessor
from src.models.model_loader import ModelLoader, print_model_info
from src.training.environment import create_environment
from src.training.trainer import create_trainer
from src.utils.hub_utils import upload_to_hub


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_credentials():
    """Load and setup API credentials."""
    load_dotenv()
    
    wandb_key = os.getenv("WANDB_API_KEY")
    hf_key = os.getenv("HF_KEY")
    
    if not wandb_key or not hf_key:
        raise ValueError(
            "Missing API keys. Please set WANDB_API_KEY and HF_KEY in .env file"
        )
    
    # Login to services
    wandb.login(key=wandb_key)
    login(token=hf_key)
    
    logger.info("✓ Credentials setup complete")
    return hf_key


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("SQL GRPO TRAINING PIPELINE")
    logger.info("="*60)
    
    # 1. Setup credentials
    logger.info("\n[1/8] Setting up credentials...")
    hf_key = setup_credentials()
    
    # 2. Load configuration
    logger.info("\n[2/8] Loading configuration...")
    config = Config()
    config.validate()
    logger.info(config)
    
    # 3. Load dataset
    logger.info("\n[3/8] Loading dataset...")
    dataset_loader = SQLDatasetLoader(config.data)
    dataset_dict = dataset_loader.load()
    
    # Print sample
    dataset_loader.print_sample('train', 0)
    
    # Preprocess for GRPO
    preprocessor = DatasetPreprocessor(config.environment.system_prompt)
    train_dataset = preprocessor.preprocess_for_grpo(dataset_dict['train'])
    test_dataset = preprocessor.preprocess_for_grpo(dataset_dict['test'])
    
    # Validate
    if not preprocessor.validate_dataset(train_dataset):
        raise ValueError("Dataset validation failed")
    
    # 4. Load model and tokenizer
    logger.info("\n[4/8] Loading model and tokenizer...")
    model_loader = ModelLoader(config.model)
    model, tokenizer = model_loader.load_model_and_tokenizer(
        apply_chat_format=True
    )
    
    # Print model info
    print_model_info(model)
    
    # 5. Create environment
    logger.info("\n[5/8] Creating GRPO environment...")
    environment = create_environment(
        dataset=train_dataset,
        env_config=config.environment,
        reward_config=config.reward,
        reward_type="combined",  # Use combined reward
        use_multi_turn=False,
    )
    
    # Test environment
    logger.info("\nTesting environment...")
    from src.training.environment import SQLEnvironment
    env_factory = SQLEnvironment(config.environment, config.reward)
    env_factory.environment = environment
    env_factory.test_environment(num_samples=2)
    
    # 6. Create trainer
    logger.info("\n[6/8] Creating GRPO trainer...")
    peft_config = model_loader.get_lora_config()
    trainer_wrapper = create_trainer(
        model=model,
        tokenizer=tokenizer,
        environment=environment,
        grpo_config=config.grpo,
        peft_config=peft_config
    )
    
    # 7. Train
    logger.info("\n[7/8] Starting training...")
    logger.info("="*60)
    
    try:
        trainer_wrapper.train(resume_from_checkpoint=True)
        trainer_wrapper.save_model()
        
        logger.info("\n✓ Training completed successfully")
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        trainer_wrapper.save_model()
        logger.info("✓ Model saved")
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        raise
    
    # 8. Merge and upload (optional)
    logger.info("\n[8/8] Post-training steps...")
    
    # Merge adapters
    try:
        logger.info("Merging LoRA adapters...")
        trainer_wrapper.merge_and_save(config.model)
    except Exception as e:
        logger.warning(f"Failed to merge adapters: {e}")
    
    # Upload to hub
    if config.grpo.push_to_hub:
        try:
            logger.info("Uploading to Hugging Face Hub...")
            upload_to_hub(
                model_dir=config.grpo.output_dir,
                hub_config=config.hub,
                hf_token=hf_key,
                adapter_only=True,  # Upload adapter only (lighter)
                commit_message="Initial GRPO-trained SQL model upload"
            )
        except Exception as e:
            logger.warning(f"Failed to upload to hub: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING PIPELINE COMPLETED")
    logger.info("="*60)
    logger.info(f"\nModel saved to: {config.grpo.output_dir}")
    logger.info(f"Merged model at: {config.grpo.output_dir}/merged")
    if config.grpo.push_to_hub:
        logger.info(f"Hub URL: https://huggingface.co/{config.hub.username}/{config.hub.repo_name}")


if __name__ == "__main__":
    main()
