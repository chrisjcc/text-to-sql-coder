"""
Training configuration for SQL GRPO training.
Centralizes all hyperparameters and settings.
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    model_id: str = "meta-llama/Meta-Llama-3.1-8B"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    
    # Quantization config
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    
    # LoRA config
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_r: int = 256
    lora_bias: str = "none"
    lora_target_modules: str = "all-linear"
    lora_task_type: str = "CAUSAL_LM"
    
    # Tokenizer config
    padding_side: str = "right"
    model_max_length: int = 2048


@dataclass
class DataConfig:
    """Data configuration parameters."""
    
    dataset_name: str = "b-mc2/sql-create-context"
    dataset_split: str = "train"
    train_size: int = 10_000
    test_size: int = 2_500
    shuffle: bool = True
    seed: int = 42
    
    # Cache
    cache_dir: Optional[str] = None


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    
    # Basic training
    output_dir: str = "code-llama-3-1-8b-text-to-sql-grpo"
    run_name: str = "sql-grpo-training"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    
    # GRPO specific
    max_seq_len: int = 2048
    max_prompt_length: int = 1024
    max_tokens: int = 1024
    num_generations: int = 4  # G in GRPO paper - samples per prompt
    beta: float = 0.1  # KL divergence coefficient
    
    # Optimization
    optim: str = "adamw_torch_fused"
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant"
    
    # Precision
    bf16: bool = True
    tf32: bool = True
    
    # Checkpointing
    gradient_checkpointing: bool = True
    save_strategy: str = "epoch"
    logging_steps: int = 10
    
    # Hub & reporting
    push_to_hub: bool = True
    report_to: str = "wandb"
    
    # Reference model (optional)
    sync_ref_model: bool = False
    ref_model_sync_steps: Optional[int] = None


@dataclass
class RewardConfig:
    """Reward function configuration."""
    
    # Reward weights
    syntax_weight: float = 0.2
    execution_weight: float = 0.6
    semantic_weight: float = 0.2
    
    # Reward values
    correct_reward: float = 1.0
    syntax_valid_reward: float = 0.3
    partial_match_reward: float = 0.5
    syntax_error_penalty: float = -0.5
    execution_error_penalty: float = -0.8
    wrong_result_penalty: float = -0.3
    
    # SQL execution
    execution_timeout: float = 5.0  # seconds
    max_result_rows: int = 1000
    use_test_database: bool = True
    
    # Validation
    normalize_sql: bool = True
    case_sensitive: bool = False


@dataclass
class EnvironmentConfig:
    """Verifiers environment configuration."""
    
    system_prompt: str = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.

Instructions:
1. Analyze the schema carefully
2. Generate syntactically correct SQL
3. Ensure your query answers the user's question
4. Use proper SQL formatting

SCHEMA:
{schema}"""
    
    # Environment type
    use_single_turn: bool = True
    max_turns: int = 1  # For multi-turn if needed
    
    # Response format (optional XML parsing)
    use_xml_format: bool = False
    xml_tags: List[str] = field(default_factory=lambda: ["think", "answer"])


@dataclass
class HubConfig:
    """Hugging Face Hub configuration."""
    
    repo_name: str = "code-llama-3.1-8b-sql-adapter"
    username: str = "chrisjcc"
    private: bool = False
    
    model_card_template: str = """---
tags:
- transformers
- finetuned
- grpo
- text-to-sql
- reinforcement-learning
license: llama3.1
---

# SQL Code Generator - GRPO Trained Model

This model was trained using Group Relative Policy Optimization (GRPO) on the b-mc2/sql-create-context dataset.

## Model Details
- Base Model: Meta-Llama-3.1-8B
- Training Method: GRPO (Reinforcement Learning)
- Task: Natural Language to SQL Translation
- Training Dataset: SQL-Create-Context

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{username}/{repo_name}")
tokenizer = AutoTokenizer.from_pretrained("{username}/{repo_name}")

prompt = "Given schema: ... Generate SQL for: Show all customers"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Training Details
- Trained with GRPO using verifiers library
- Reward function based on SQL execution correctness
- LoRA adapters with r=256, alpha=128

## Citation
If you use this model, please cite:
```bibtex
@misc{{sql-grpo-model,
  author = {{{username}}},
  title = {{SQL Code Generator - GRPO Trained}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{username}/{repo_name}}}}}
}}
```
"""


class Config:
    """Main configuration class combining all configs."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.grpo = GRPOConfig()
        self.reward = RewardConfig()
        self.environment = EnvironmentConfig()
        self.hub = HubConfig()
    
    def validate(self):
        """Validate configuration settings."""
        assert self.grpo.max_seq_len <= self.model.model_max_length, \
            "max_seq_len cannot exceed model_max_length"
        assert self.grpo.num_generations >= 2, \
            "num_generations should be at least 2 for GRPO"
        assert 0 <= self.grpo.beta <= 1, \
            "beta should be between 0 and 1"
        assert self.data.train_size > 0, \
            "train_size must be positive"
        
    def __repr__(self):
        return f"""Config:
  Model: {self.model.model_id}
  Output: {self.grpo.output_dir}
  Training samples: {self.data.train_size}
  Epochs: {self.grpo.num_train_epochs}
  Batch size: {self.grpo.per_device_train_batch_size}
  Learning rate: {self.grpo.learning_rate}
  Num generations: {self.grpo.num_generations}
"""
