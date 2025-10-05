"""
Evaluation script for trained SQL model.

Usage:
    python scripts/evaluate.py --model_path <path> --num_samples 100
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.training_config import Config
from src.rewards.sql_reward import create_sql_reward
from src.data.dataset_loader import convert_to_verifiers_format


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate trained SQL generation model."""
    
    def __init__(
        self,
        model_path: str,
        config: Config
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            config: Configuration object
        """
        self.model_path = model_path
        self.config = config
        self.model = None
        self.tokenizer = None
        self.reward_fn = None
    
    def load_model(self):
        """Load trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        
        logger.info("✓ Model loaded")
    
    def load_reward_function(self):
        """Load reward function for evaluation."""
        self.reward_fn = create_sql_reward(
            self.config.reward,
            reward_type="combined"
        )
    
    def generate_sql(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        num_return_sequences: int = 1
    ) -> list[str]:
        """
        Generate SQL from prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_return_sequences: Number of sequences to generate
        
        Returns:
            List of generated SQL queries
        """
        # Format prompt as chat message
        messages = [
            {"role": "system", "content": self.config.environment.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.model_max_length
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated = []
        for output in outputs:
            # Remove prompt from output
            output = output[inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated.append(text)
        
        return generated
    
    def evaluate_sample(
        self,
        sample: dict,
        num_generations: int = 1
    ) -> dict:
        """
        Evaluate model on a single sample.
        
        Args:
            sample: Dataset sample
            num_generations: Number of SQL queries to generate
        
        Returns:
            Evaluation results
        """
        prompt = sample['prompt']
        answer = sample['answer']
        context = sample.get('context', '')
        
        # Generate SQL
        generated_sqls = self.generate_sql(
            prompt,
            num_return_sequences=num_generations
        )
        
        # Evaluate each generation
        rewards = []
        for gen_sql in generated_sqls:
            reward = self.reward_fn(
                prompt=prompt,
                completion=gen_sql,
                answer=answer,
                context=context
            )
            rewards.append(reward)
        
        return {
            "prompt": prompt[:100] + "...",
            "answer": answer,
            "generated": generated_sqls,
            "rewards": rewards,
            "best_reward": max(rewards),
            "mean_reward": sum(rewards) / len(rewards),
        }
    
    def evaluate_dataset(
        self,
        dataset,
        num_samples: int = 100,
        num_generations: int = 1
    ) -> dict:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Evaluation dataset
            num_samples: Number of samples to evaluate
            num_generations: Number of generations per sample
        
        Returns:
            Aggregated evaluation results
        """
        logger.info(f"Evaluating on {num_samples} samples...")
        
        results = []
        num_samples = min(num_samples, len(dataset))
        
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[i]
            result = self.evaluate_sample(sample, num_generations)
            results.append(result)
        
        # Aggregate results
        best_rewards = [r['best_reward'] for r in results]
        mean_rewards = [r['mean_reward'] for r in results]
        
        aggregated = {
            "num_samples": num_samples,
            "num_generations_per_sample": num_generations,
            "best_reward_mean": sum(best_rewards) / len(best_rewards),
            "best_reward_std": torch.tensor(best_rewards).std().item(),
            "mean_reward_mean": sum(mean_rewards) / len(mean_rewards),
            "mean_reward_std": torch.tensor(mean_rewards).std().item(),
            "success_rate": sum(1 for r in best_rewards if r > 0.5) / len(best_rewards),
            "perfect_rate": sum(1 for r in best_rewards if r > 0.95) / len(best_rewards),
            "detailed_results": results,
        }
        
        return aggregated
    
    def print_results(self, results: dict, show_samples: int = 5):
        """
        Print evaluation results.
        
        Args:
            results: Evaluation results
            show_samples: Number of sample results to display
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Samples evaluated: {results['num_samples']}")
        print(f"Generations per sample: {results['num_generations_per_sample']}")
        print(f"\nBest Reward Mean: {results['best_reward_mean']:.3f} ± {results['best_reward_std']:.3f}")
        print(f"Mean Reward Mean: {results['mean_reward_mean']:.3f} ± {results['mean_reward_std']:.3f}")
        print(f"Success Rate (>0.5): {results['success_rate']*100:.1f}%")
        print(f"Perfect Rate (>0.95): {results['perfect_rate']*100:.1f}%")
        
        print(f"\n{'='*60}")
        print(f"SAMPLE RESULTS (showing {show_samples})")
        print("="*60)
        
        for i, result in enumerate(results['detailed_results'][:show_samples]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Prompt: {result['prompt']}")
            print(f"Ground Truth: {result['answer']}")
            print(f"Generated: {result['generated'][0]}")
            print(f"Reward: {result['rewards'][0]:.3f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained SQL model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="code-llama-3-1-8b-text-to-sql-grpo/merged",
        help="Path to trained model"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of SQL queries to generate per sample"
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=5,
        help="Number of sample results to display"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path, config)
    evaluator.load_model()
    evaluator.load_reward_function()
    
    # Load test dataset
    logger.info("Loading test dataset...")
    dataset = load_dataset(
        config.data.dataset_name,
        split="train"
    )
    
    # Select test subset
    test_start = config.data.train_size
    test_end = test_start + config.data.test_size
    dataset = dataset.select(range(test_start, test_end))
    
    # Convert to verifiers format
    dataset = dataset.map(convert_to_verifiers_format, batched=False)
    
    # Evaluate
    results = evaluator.evaluate_dataset(
        dataset,
        num_samples=args.num_samples,
        num_generations=args.num_generations
    )
    
    # Print results
    evaluator.print_results(results, show_samples=args.show_samples)
    
    # Save results
    import json
    output_path = Path(args.model_path) / "evaluation_results.json"
    with open(output_path, "w") as f:
        # Remove detailed results for JSON
        results_to_save = {k: v for k, v in results.items() if k != "detailed_results"}
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
