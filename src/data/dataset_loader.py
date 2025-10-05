"""
Dataset loading and preprocessing for SQL GRPO training.
"""
import logging
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset, DatasetDict


logger = logging.getLogger(__name__)


class SQLDatasetLoader:
    """
    Loads and preprocesses SQL datasets for GRPO training.
    """
    
    def __init__(self, config):
        """
        Initialize dataset loader.
        
        Args:
            config: DataConfig object
        """
        self.config = config
        self.dataset: Optional[DatasetDict] = None
    
    def load(self) -> DatasetDict:
        """
        Load dataset from HuggingFace Hub.
        
        Returns:
            DatasetDict with train and test splits
        """
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=self.config.cache_dir,
        )
        
        # Shuffle if requested
        if self.config.shuffle:
            dataset = dataset.shuffle(seed=self.config.seed)
        
        # Select subset
        total_size = self.config.train_size + self.config.test_size
        if total_size < len(dataset):
            dataset = dataset.select(range(total_size))
            logger.info(f"Selected {total_size} samples from dataset")
        
        # Split into train/test
        test_size = self.config.test_size / (self.config.train_size + self.config.test_size)
        dataset_dict = dataset.train_test_split(
            test_size=test_size,
            seed=self.config.seed
        )
        
        logger.info(f"Dataset loaded: {len(dataset_dict['train'])} train, "
                   f"{len(dataset_dict['test'])} test samples")
        
        self.dataset = dataset_dict
        return dataset_dict
    
    def get_train_dataset(self) -> Dataset:
        """Get training dataset."""
        if self.dataset is None:
            self.load()
        return self.dataset['train']
    
    def get_test_dataset(self) -> Dataset:
        """Get test dataset."""
        if self.dataset is None:
            self.load()
        return self.dataset['test']
    
    def get_sample(self, split: str = 'train', index: int = 0) -> Dict[str, Any]:
        """
        Get a single sample from dataset.
        
        Args:
            split: 'train' or 'test'
            index: Sample index
        
        Returns:
            Dataset sample
        """
        if self.dataset is None:
            self.load()
        return self.dataset[split][index]
    
    def print_sample(self, split: str = 'train', index: int = 0):
        """Print a dataset sample for inspection."""
        sample = self.get_sample(split, index)
        print(f"\n=== Dataset Sample ({split}[{index}]) ===")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 200:
                value = value[:200] + "..."
            print(f"{key}: {value}")
        print("=" * 50)


def convert_to_verifiers_format(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert dataset sample to verifiers format.
    
    Expected input format (b-mc2/sql-create-context):
    {
        "question": "What is the average...",
        "context": "CREATE TABLE...",
        "answer": "SELECT AVG(...)..."
    }
    
    Output format for verifiers SingleTurnEnv:
    {
        "prompt": "Question with schema context",
        "answer": "SQL query",
        "context": "Original schema (for reward function)",
    }
    
    Args:
        sample: Original dataset sample
    
    Returns:
        Converted sample
    """
    question = sample.get('question', '')
    context = sample.get('context', '')
    answer = sample.get('answer', '')
    
    # Create prompt with schema
    prompt = f"""Question: {question}

Schema:
{context}

Generate the SQL query to answer this question."""
    
    return {
        'prompt': prompt,
        'answer': answer,
        'context': context,  # Keep for reward function
        'original_question': question,  # Keep for reference
    }


def create_conversation_format(
    sample: Dict[str, Any],
    system_prompt_template: str
) -> Dict[str, Any]:
    """
    Convert to OpenAI conversation format (for SFT warmup).
    
    Args:
        sample: Dataset sample
        system_prompt_template: System prompt with {schema} placeholder
    
    Returns:
        Sample with messages field
    """
    question = sample.get('question', '')
    context = sample.get('context', '')
    answer = sample.get('answer', '')
    
    # Format system prompt with schema
    system_prompt = system_prompt_template.format(schema=context)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    return {
        "messages": messages,
        **sample  # Keep original fields
    }


class DatasetPreprocessor:
    """
    Preprocesses datasets for GRPO training.
    """
    
    def __init__(self, system_prompt_template: str):
        """
        Initialize preprocessor.
        
        Args:
            system_prompt_template: System prompt template
        """
        self.system_prompt_template = system_prompt_template
    
    def preprocess_for_grpo(
        self,
        dataset: Dataset,
        batch_size: int = 1000
    ) -> Dataset:
        """
        Preprocess dataset for GRPO training.
        
        Args:
            dataset: Input dataset
            batch_size: Batch size for mapping
        
        Returns:
            Preprocessed dataset
        """
        logger.info("Preprocessing dataset for GRPO training...")
        
        # Convert to verifiers format
        dataset = dataset.map(
            convert_to_verifiers_format,
            batched=False,
            desc="Converting to verifiers format"
        )
        
        logger.info(f"Preprocessed {len(dataset)} samples")
        return dataset
    
    def preprocess_for_sft(
        self,
        dataset: Dataset,
        tokenizer,
        batch_size: int = 1000
    ) -> Dataset:
        """
        Preprocess dataset for SFT warmup training.
        
        Args:
            dataset: Input dataset
            tokenizer: Tokenizer with chat template
            batch_size: Batch size for mapping
        
        Returns:
            Preprocessed dataset with 'text' field
        """
        logger.info("Preprocessing dataset for SFT training...")
        
        # Convert to conversation format
        def convert_fn(sample):
            return create_conversation_format(
                sample, 
                self.system_prompt_template
            )
        
        dataset = dataset.map(
            convert_fn,
            batched=False,
            desc="Converting to conversation format"
        )
        
        # Apply chat template
        def format_for_training(example):
            formatted_text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": formatted_text}
        
        dataset = dataset.map(
            format_for_training,
            batched=False,
            desc="Applying chat template"
        )
        
        logger.info(f"Preprocessed {len(dataset)} samples for SFT")
        return dataset
    
    def validate_dataset(self, dataset: Dataset) -> bool:
        """
        Validate dataset has required fields.
        
        Args:
            dataset: Dataset to validate
        
        Returns:
            True if valid
        """
        required_fields = {'prompt', 'answer'}
        
        if not all(field in dataset.column_names for field in required_fields):
            missing = required_fields - set(dataset.column_names)
            logger.error(f"Missing required fields: {missing}")
            return False
        
        # Check sample
        sample = dataset[0]
        if not sample['prompt'] or not sample['answer']:
            logger.error("Empty prompt or answer in first sample")
            return False
        
        logger.info("Dataset validation passed")
        return True
