"""
Test configuration for dry-run training with minimal resources.
Use this for validation and smoke testing.
"""
from dataclasses import dataclass
from .training_config import (
    ModelConfig, DataConfig, GRPOConfig, RewardConfig, 
    EnvironmentConfig, HubConfig, Config
)


@dataclass
class TestModelConfig(ModelConfig):
    """Minimal model config for testing."""
    # Use smaller model for testing (if available)
    model_id: str = "microsoft/DialoGPT-small"  # Much smaller than Llama
    # Reduce memory usage
    lora_r: int = 16  # Much smaller LoRA rank
    lora_alpha: int = 32
    model_max_length: int = 512  # Shorter sequences


@dataclass
class TestDataConfig(DataConfig):
    """Minimal data config for testing."""
    train_size: int = 20  # Very small for testing
    test_size: int = 4    # Minimal test set
    

@dataclass
class TestGRPOConfig(GRPOConfig):
    """Minimal GRPO config for testing."""
    output_dir: str = "test-sql-grpo-model"
    run_name: str = "sql-grpo-test"
    
    # Minimal training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # GRPO specific - minimal
    max_seq_len: int = 512
    max_prompt_length: int = 256
    max_tokens: int = 256
    num_generations: int = 2  # Minimum for GRPO
    
    # Disable expensive features for testing
    push_to_hub: bool = False
    report_to: str = "none"  # Disable wandb for testing
    gradient_checkpointing: bool = False  # Simpler for testing


@dataclass
class TestRewardConfig(RewardConfig):
    """Test reward config with shorter timeouts."""
    execution_timeout: float = 2.0  # Shorter timeout for testing
    max_result_rows: int = 100      # Fewer rows for testing


@dataclass
class TestHubConfig(HubConfig):
    """Test hub config - don't upload."""
    repo_name: str = "test-sql-adapter"
    private: bool = True  # Keep test models private


class TestConfig:
    """Test configuration combining all minimal configs."""
    
    def __init__(self):
        self.model = TestModelConfig()
        self.data = TestDataConfig()
        self.grpo = TestGRPOConfig()
        self.reward = TestRewardConfig()
        self.environment = EnvironmentConfig()  # Keep default
        self.hub = TestHubConfig()
    
    def validate(self):
        """Validate test configuration."""
        assert self.grpo.max_seq_len <= self.model.model_max_length
        assert self.grpo.num_generations >= 2
        assert self.data.train_size > 0
        assert self.data.test_size > 0
        
    def __repr__(self):
        return f"""TestConfig:
  Model: {self.model.model_id}
  Training samples: {self.data.train_size}
  Test samples: {self.data.test_size}
  Epochs: {self.grpo.num_train_epochs}
  Batch size: {self.grpo.per_device_train_batch_size}
  Num generations: {self.grpo.num_generations}
  Output dir: {self.grpo.output_dir}
"""