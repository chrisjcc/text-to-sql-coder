# SQL GRPO Training - Detailed Usage Guide

This guide provides step-by-step instructions for using the SQL GRPO training framework.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Configuration](#configuration)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Customization](#customization)
6. [Troubleshooting](#troubleshooting)

## Installation & Setup

### Prerequisites

- Python 3.11 or 3.12
- CUDA-compatible GPU (16GB+ VRAM recommended)
- 50GB free disk space

### Step 1: Clone and Install

```bash
# Clone repository
git clone <repository-url>
cd sql_grpo_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install package in editable mode
pip install -e .
```

### Step 2: Install Flash Attention (Optional but Recommended)

For ~30% faster training:

```bash
pip install flash-attn --no-build-isolation
```

### Step 3: Setup API Keys

```bash
# Copy example env file
cp .env.example .env

# Edit .env with your keys
nano .env  # or use your preferred editor
```

Add your keys:
```bash
WANDB_API_KEY=your_wandb_key
HF_KEY=your_huggingface_token
```

Get keys from:
- Weights & Biases: https://wandb.ai/settings
- Hugging Face: https://huggingface.co/settings/tokens

### Step 4: Verify Installation

```bash
# Run tests
python -m pytest tests/test_rewards.py -v

# Test reward functions
python -c "
from config.training_config import Config
from src.rewards.sql_reward import create_sql_reward

config = Config()
reward = create_sql_reward(config.reward)
print('✓ Reward function working')
"
```

## Configuration

All configuration is in `config/training_config.py`. Here's how to customize it:

### Basic Configuration

```python
from config.training_config import Config

config = Config()

# Change model
config.model.model_id = "meta-llama/Meta-Llama-3.1-8B"

# Adjust training
config.grpo.num_train_epochs = 3
config.grpo.learning_rate = 2e-4
config.grpo.num_generations = 4  # Samples per prompt

# Adjust dataset
config.data.train_size = 10_000
config.data.test_size = 2_500

# Save changes
# (Edit the file directly or create a new config file)
```

### Memory Optimization

For GPUs with limited memory:

```python
# Reduce batch size
config.grpo.per_device_train_batch_size = 1

# Increase gradient accumulation
config.grpo.gradient_accumulation_steps = 16

# Reduce sequence length
config.grpo.max_seq_len = 1536

# Enable gradient checkpointing (already default)
config.grpo.gradient_checkpointing = True
```

### Reward Function Tuning

```python
# Adjust reward weights
config.reward.syntax_weight = 0.2
config.reward.execution_weight = 0.6  # Most important
config.reward.semantic_weight = 0.2

# Adjust reward values
config.reward.correct_reward = 1.0
config.reward.execution_error_penalty = -0.8
```

## Training

### Basic Training

```bash
python scripts/train.py
```

This will:
1. Load the SQL-Create-Context dataset
2. Setup GRPO environment with reward functions
3. Train for specified epochs
4. Save LoRA adapters
5. Merge adapters with base model
6. Upload to Hugging Face Hub (if enabled)

### Monitor Training

View training progress in Weights & Biases:

```bash
# Training metrics will be at:
# https://wandb.ai/<your-username>/sql-grpo-training
```

Key metrics to watch:
- `reward/mean`: Should increase over training
- `reward/std`: Should remain > 0 (diversity)
- `kl_divergence`: Should stay < 0.5
- `loss`: Should decrease

### Resume Training

Training automatically resumes from checkpoints:

```bash
# Just run the same command
python scripts/train.py

# Or explicitly disable resume
# (modify trainer.train(resume_from_checkpoint=False))
```

### Training with Custom Dataset

1. **Prepare your dataset** in HuggingFace format:

```python
from datasets import Dataset

data = {
    "question": ["What is...", "How to..."],
    "context": ["CREATE TABLE...", "CREATE TABLE..."],
    "answer": ["SELECT...", "SELECT..."]
}

dataset = Dataset.from_dict(data)
dataset.push_to_hub("username/my-sql-dataset")
```

2. **Update config**:

```python
config.data.dataset_name = "username/my-sql-dataset"
```

3. **Run training**:

```bash
python scripts/train.py
```

## Evaluation

### Evaluate Trained Model

```bash
python scripts/evaluate.py \
    --model_path code-llama-3-1-8b-text-to-sql-grpo/merged \
    --num_samples 100 \
    --num_generations 1 \
    --show_samples 5
```

Options:
- `--model_path`: Path to trained model
- `--num_samples`: Number of test samples
- `--num_generations`: SQL queries per sample
- `--show_samples`: Sample results to display

### Evaluate on Custom Data

```python
from scripts.evaluate import ModelEvaluator
from config.training_config import Config
from datasets import load_dataset

config = Config()
evaluator = ModelEvaluator("path/to/model", config)
evaluator.load_model()
evaluator.load_reward_function()

# Load your test data
test_data = load_dataset("your/test-dataset")

# Evaluate
results = evaluator.evaluate_dataset(test_data, num_samples=50)
evaluator.print_results(results)
```

### Interactive Testing

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "code-llama-3-1-8b-text-to-sql-grpo/merged",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(
    "code-llama-3-1-8b-text-to-sql-grpo/merged"
)

# Generate SQL
messages = [
    {"role": "system", "content": "You are a SQL generator..."},
    {"role": "user", "content": "Get all users older than 25"}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256)
sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(sql)
```

## Customization

### Custom Reward Function

Create a new reward function:

```python
# src/rewards/my_reward.py
from src.rewards.base_reward import BaseReward, RewardResult

class MyCustomReward(BaseReward):
    def compute(self, prompt, completion, answer, **kwargs):
        # Your custom logic
        score = self._compute_score(completion, answer)
        
        return RewardResult(
            score=score,
            details={"custom_metric": score}
        )
    
    def _compute_score(self, completion, answer):
        # Example: reward query optimization
        if "INDEX" in completion.upper():
            return 1.0
        return 0.5
```

Use in training:

```python
# In environment.py
from src.rewards.my_reward import MyCustomReward

reward = MyCustomReward(config.reward)
rubric = vf.Rubric(reward)
```

### Custom System Prompt

Edit in `config/training_config.py`:

```python
class EnvironmentConfig:
    system_prompt: str = """You are an expert SQL developer.
    
Generate optimized SQL queries that:
1. Use proper indexing
2. Avoid N+1 queries
3. Include proper JOINs

SCHEMA:
{schema}"""
```

### Multi-Turn Environment (Advanced)

Implement iterative refinement:

```python
# src/training/multi_turn_sql_env.py
import verifiers as vf
from typing import Tuple

class SQLRefinementEnv(vf.MultiTurnEnv):
    def __init__(self, dataset, rubric, max_turns=3):
        super().__init__(max_turns=max_turns, dataset=dataset, rubric=rubric)
    
    async def is_completed(self, messages, state, **kwargs):
        # Check if SQL is correct or max turns reached
        if await super().is_completed(messages, state, **kwargs):
            return True
        
        # Check if last SQL was correct
        last_reward = state.get("last_reward", 0)
        return last_reward > 0.9
    
    async def env_response(self, messages, state, **kwargs):
        # Execute SQL and provide feedback
        last_sql = messages[-1]["content"]
        
        # Get execution results
        reward = self.rubric(prompt="", completion=last_sql, answer="", **kwargs)
        
        # Generate feedback
        if reward > 0.9:
            feedback = "SQL is correct!"
        else:
            feedback = "SQL has issues. Try again with corrections."
        
        state["last_reward"] = reward
        
        return (
            [{"role": "user", "content": feedback}],
            state
        )
```

### Different Base Models

Train with other models:

```python
# Mistral 7B
config.model.model_id = "mistralai/Mistral-7B-v0.1"

# CodeLlama
config.model.model_id = "codellama/CodeLlama-7b-hf"

# Phi-3
config.model.model_id = "microsoft/Phi-3-mini-4k-instruct"
```

## Troubleshooting

### Out of Memory Errors

```bash
# Error: CUDA out of memory
```

**Solutions:**

1. Reduce batch size:
```python
config.grpo.per_device_train_batch_size = 1
config.grpo.gradient_accumulation_steps = 16
```

2. Reduce sequence length:
```python
config.grpo.max_seq_len = 1024
config.grpo.max_prompt_length = 512
```

3. Enable gradient checkpointing (default):
```python
config.grpo.gradient_checkpointing = True
```

4. Use smaller model:
```python
config.model.model_id = "meta-llama/Meta-Llama-3.1-8B"  # Instead of 70B
```

### Slow Training

```bash
# Training is very slow
```

**Solutions:**

1. Install Flash Attention:
```bash
pip install flash-attn --no-build-isolation
```

2. Use multiple GPUs:
```bash
# Training automatically uses all available GPUs
# Verify with:
nvidia-smi
```

3. Reduce generations per prompt:
```python
config.grpo.num_generations = 2  # Instead of 4
```

### Low Rewards

```bash
# Rewards are consistently negative or low
```

**Solutions:**

1. Check SQL execution:
```python
from src.rewards.sql_executor import SQLExecutor

executor = SQLExecutor()
result = executor.execute(
    "SELECT * FROM users",
    "CREATE TABLE users (id INT, name TEXT)"
)
print(result)
```

2. Test reward function:
```python
from src.rewards.sql_reward import create_sql_reward

reward = create_sql_reward(config.reward)
score = reward(
    prompt="test",
    completion="SELECT * FROM users",
    answer="SELECT * FROM users",
    context="CREATE TABLE users (id INT)"
)
print(f"Score: {score}")
```

3. Adjust reward weights:
```python
# Focus more on syntax if execution is failing
config.reward.syntax_weight = 0.5
config.reward.execution_weight = 0.3
config.reward.semantic_weight = 0.2
```

### Import Errors

```bash
# ModuleNotFoundError: No module named 'src'
```

**Solution:**

```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or install package
pip install -e .
```

### WandB Connection Issues

```bash
# wandb: ERROR Unable to connect
```

**Solution:**

```bash
# Login again
wandb login

# Or disable wandb
# In config:
config.grpo.report_to = "none"
```

### HuggingFace Upload Fails

```bash
# Error: Authentication token not found
```

**Solution:**

```bash
# Login to HuggingFace
huggingface-cli login

# Or set token in .env
HF_KEY=your_token_here
```

## Advanced Topics

### Distributed Training

For multi-node training:

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch scripts/train.py
```

### Custom Dataset Format

If your dataset has different columns:

```python
# In dataset_loader.py
def convert_custom_format(sample):
    return {
        'prompt': sample['my_question_field'],
        'answer': sample['my_sql_field'],
        'context': sample['my_schema_field'],
    }

dataset = dataset.map(convert_custom_format)
```

### Export to ONNX

For production deployment:

```python
from transformers import AutoModelForCausalLM
from optimum.onnxruntime import ORTModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("path/to/model")

# Export
ort_model = ORTModelForCausalLM.from_pretrained(
    "path/to/model",
    export=True
)

ort_model.save_pretrained("path/to/onnx")
```

## Best Practices

1. **Start small**: Train on 1000 samples first to verify everything works
2. **Monitor early**: Check first epoch metrics before running full training
3. **Test rewards**: Validate reward functions on ground truth before training
4. **Save checkpoints**: Keep intermediate checkpoints in case of issues
5. **Version control**: Track config changes and experiments
6. **Document changes**: Keep notes on what works and what doesn't

## Getting Help

- **Issues**: Open a GitHub issue
- **Questions**: Check existing issues or start a discussion
- **Contributing**: See CONTRIBUTING.md

## References

- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
