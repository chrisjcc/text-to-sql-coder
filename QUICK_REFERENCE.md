# SQL Fine-Tune Training - Quick Reference

## Installation

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation  # Optional
cp .env.example .env  # Add your API keys
```

## Training

```bash
python scripts/train.py
```

## Evaluation

```bash
python scripts/evaluate.py --model_path <path> --num_samples 100
```

## Configuration Cheat Sheet

### File: `config/training_config.py`

```python
# Model
model_id = "meta-llama/Meta-Llama-3.1-8B"
lora_r = 256
lora_alpha = 128

# Data
train_size = 10_000
test_size = 2_500
