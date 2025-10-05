# SQL Text-to-Code Fine-Tune Training

A modular, production-ready implementation for training Large Language Models to translate natural language to SQL using **Group Relative Policy Optimization (GRPO)** with the verifiers library.

## 🎯 Overview

This project trains LLMs using reinforcement learning to generate SQL queries from natural language questions. Unlike traditional supervised fine-tuning, GRPO uses verifiable rewards based on SQL execution correctness to optimize the model.

### Key Features

- ✅ **GRPO Training**: Uses verifiers library for RL-based training
- ✅ **Modular Architecture**: Clean separation of concerns
- ✅ **Multiple Reward Types**: Syntax, execution, and semantic rewards
- ✅ **LoRA/QLoRA**: Memory-efficient training with 4-bit quantization
- ✅ **Safe SQL Execution**: Sandboxed validation environment
- ✅ **Hub Integration**: Automatic upload to Hugging Face

## 📁 Project Structure

```
sql_grpo_training/
├── config/
│   └── training_config.py      # All configuration parameters
├── src/
│   ├── data/
│   │   └── dataset_loader.py   # Dataset loading and preprocessing
│   ├── rewards/
│   │   ├── base_reward.py      # Abstract reward interface
│   │   ├── sql_executor.py     # Safe SQL execution
│   │   └── sql_reward.py       # SQL-specific rewards
│   ├── models/
│   │   └── model_loader.py     # Model loading with quantization
│   ├── training/
│   │   ├── environment.py      # Verifiers environment setup
│   │   └── trainer.py          # GRPO trainer wrapper
│   └── utils/
│       ├── logging_utils.py
│       └── hub_utils.py         # HF Hub upload
├── scripts/
│   ├── train.py                # Main training script
│   └── evaluate.py             # Evaluation script
├── tests/
│   ├── test_rewards.py
│   └── test_data.py
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo>
cd sql_grpo_training

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Flash Attention for better performance
pip install flash-attn --no-build-isolation
```

### 2. Setup Credentials

Create a `.env` file with your API keys:

```bash
cp .env.example .env
# Edit .env and add your keys
```

Required keys:
- `WANDB_API_KEY`: For experiment tracking
- `HF_KEY`: For Hugging Face Hub access

### 3. Run Training

```bash
python scripts/train.py
```

The script will:
1. Load the SQL-Create-Context dataset
2. Setup the GRPO environment with reward functions
3. Train the model with RL
4. Save LoRA adapters and merged model
5. Upload to Hugging Face Hub (optional)

## ⚙️ Configuration

All configuration is centralized in `config/training_config.py`. Key parameters:

### Model Configuration
```python
model_id = "meta-llama/Meta-Llama-3.1-8B"  # Base model
lora_r = 256                                # LoRA rank
lora_alpha = 128                            # LoRA alpha
```

### Training Configuration
```python
num_train_epochs = 3
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
num_generations = 4  # Samples per prompt for GRPO
beta = 0.1           # KL divergence coefficient
```

### Reward Configuration
```python
syntax_weight = 0.2      # Weight for syntax validation
execution_weight = 0.6   # Weight for execution correctness
semantic_weight = 0.2    # Weight for semantic similarity
```

## 🎯 Reward Functions

The training uses a composite reward function with three components:

### 1. Syntax Reward
- Validates SQL syntax using sqlparse
- Fast but doesn't guarantee correctness
- Score: -0.5 (invalid) to 1.0 (valid + high quality)

### 2. Execution Reward (Primary)
- Executes generated SQL in sandboxed SQLite environment
- Compares results with reference query
- Score: -0.8 (execution error) to 1.0 (exact match)

### 3. Semantic Reward
- Normalizes and compares SQL structure
- Token-level similarity using Jaccard index
- Score: -0.2 (very different) to 0.8 (semantically similar)

### Combined Reward
Default weights: 20% syntax + 60% execution + 20% semantic

## 📊 Monitoring

Training metrics are logged to Weights & Biases:

```bash
wandb login
# Run training
# View at: https://wandb.ai/<username>/<project>
```

Key metrics:
- `reward/mean`: Average reward across generations
- `reward/std`: Reward standard deviation (diversity)
- `kl_divergence`: KL divergence from reference model
- `loss`: GRPO training loss

## 🧪 Testing Reward Functions

Test reward functions before training:

```python
from config.training_config import Config
from src.rewards.sql_reward import create_sql_reward

config = Config()
reward_fn = create_sql_reward(config.reward, reward_type="combined")

# Test on example
result = reward_fn.compute(
    prompt="Schema: CREATE TABLE users (id INT, name TEXT);\n\nQuestion: Get all users",
    completion="SELECT * FROM users",
    answer="SELECT * FROM users",
    context="CREATE TABLE users (id INT, name TEXT);"
)

print(f"Score: {result.score}")
print(f"Details: {result.details}")
```

## 🔧 Advanced Usage

### Custom Reward Functions

Create custom rewards by extending `BaseReward`:

```python
from src.rewards.base_reward import BaseReward, RewardResult

class MyCustomReward(BaseReward):
    def compute(self, prompt, completion, answer, **kwargs):
        # Your reward logic
        score = compute_my_score(completion, answer)
        
        return RewardResult(
            score=score,
            details={"my_metric": score}
        )
```

### Using Different Models

Edit `config/training_config.py`:

```python
model_id = "mistralai/Mistral-7B-v0.1"  # Or any other model
```

### Multi-GPU Training

GRPO is optimized for 2+ GPUs:

```bash
# Automatic with accelerate
accelerate config  # Configure multi-GPU setup
python scripts/train.py
```

### Training with Different Datasets

Modify `dataset_loader.py` to load your custom dataset:

```python
dataset = load_dataset("your-username/your-sql-dataset")
```

Required columns: `question`, `context` (schema), `answer` (SQL)

## 📈 Expected Results

With default settings on Llama-3.1-8B:

- **Training time**: ~6-8 hours on 2x A100 (40GB)
- **Memory**: ~24GB per GPU with 4-bit quantization
- **Final reward**: ~0.6-0.8 (depending on dataset difficulty)
- **Execution accuracy**: ~70-80% on test set

## 🐛 Troubleshooting

### Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Enable `gradient_checkpointing=True`
- Use smaller model or reduce `max_seq_len`

### Slow Training
- Install Flash Attention: `pip install flash-attn --no-build-isolation`
- Increase `per_device_train_batch_size` if memory allows
- Use multiple GPUs

### Low Rewards
- Check SQL execution errors in logs
- Verify schema extraction is working correctly
- Test reward functions on ground truth samples
- Adjust reward weights in config

### Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

## 📚 Key Dependencies

- **verifiers** (>=0.1.0): RL training library
- **transformers** (>=4.40.0): Model loading
- **peft** (>=0.10.0): LoRA adapters
- **trl** (>=0.8.0): Chat format setup
- **bitsandbytes** (>=0.42.0): Quantization
- **sqlparse** (>=0.4.4): SQL parsing
- **wandb** (>=0.16.0): Experiment tracking

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. Multi-turn SQL refinement environment
2. Additional reward functions (query optimization, security)
3. Support for other SQL dialects (PostgreSQL, MySQL)
4. Evaluation scripts with standard benchmarks
5. SFT warmup training option

## 📖 References

- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [SQL-Create-Context Dataset](https://huggingface.co/datasets/b-mc2/sql-create-context)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Verifiers library by PrimeIntellect-ai
- SQL-Create-Context dataset by b-mc2
- Meta's Llama models
- Hugging Face ecosystem

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [chrisjcc.physics@gmail.com].

---

**Note**: This is a research/educational project. Always validate SQL outputs before executing on production databases.
