"""
Verifiers environment setup for SQL GRPO training.
"""
import logging
from typing import Optional
import verifiers as vf
from datasets import Dataset

from ..rewards.sql_reward import create_sql_reward


logger = logging.getLogger(__name__)


class SQLEnvironment:
    """
    Factory for creating SQL training environments.
    """
    
    def __init__(
        self,
        env_config,
        reward_config
    ):
        """
        Initialize environment factory.
        
        Args:
            env_config: EnvironmentConfig object
            reward_config: RewardConfig object
        """
        self.env_config = env_config
        self.reward_config = reward_config
        self.environment: Optional[vf.Environment] = None
    
    def create_single_turn_env(
        self,
        dataset: Dataset,
        reward_type: str = "combined"
    ) -> vf.SingleTurnEnv:
        """
        Create SingleTurnEnv for SQL generation.
        
        Args:
            dataset: Training dataset with 'prompt' and 'answer' columns
            reward_type: Type of reward function to use
        
        Returns:
            Configured SingleTurnEnv
        """
        logger.info(f"Creating SingleTurnEnv with {reward_type} reward")
        
        # Create reward function
        reward_fn = create_sql_reward(self.reward_config, reward_type)
        
        # Create rubric
        if self.env_config.use_xml_format:
            # Use XML parser for structured output
            parser = vf.XMLParser(self.env_config.xml_tags)
            rubric = vf.Rubric(
                reward_fn,
                parser.get_format_reward_func(),
                weights=[0.9, 0.1]  # 90% correctness, 10% format
            )
            system_prompt = f"""{self.env_config.system_prompt}

Output your response in the following format:
<think>Your reasoning process here</think>
<answer>Your SQL query here</answer>"""
        else:
            # Simple reward without format requirements
            rubric = vf.Rubric(reward_fn)
            system_prompt = self.env_config.system_prompt
        
        # Create environment
        env = vf.SingleTurnEnv(
            dataset=dataset,
            rubric=rubric,
            system_prompt=system_prompt,
        )
        
        self.environment = env
        logger.info("✓ Environment created")
        
        return env
    
    def create_multi_turn_env(
        self,
        dataset: Dataset,
        reward_type: str = "combined",
        max_turns: int = 3
    ) -> vf.MultiTurnEnv:
        """
        Create MultiTurnEnv for iterative SQL refinement.
        
        This allows the model to receive feedback and refine queries.
        
        Args:
            dataset: Training dataset
            reward_type: Type of reward function
            max_turns: Maximum conversation turns
        
        Returns:
            Configured MultiTurnEnv
        """
        logger.info(f"Creating MultiTurnEnv with {reward_type} reward")
        
        # Create reward function
        reward_fn = create_sql_reward(self.reward_config, reward_type)
        rubric = vf.Rubric(reward_fn)
        
        # For multi-turn, we could implement a custom environment
        # that provides feedback after each SQL attempt
        # This is more advanced and requires custom implementation
        
        raise NotImplementedError(
            "MultiTurnEnv for SQL refinement not yet implemented. "
            "Use SingleTurnEnv for now."
        )
    
    def validate_environment(self) -> bool:
        """
        Validate environment is properly configured.
        
        Returns:
            True if valid
        """
        if self.environment is None:
            logger.error("Environment not created")
            return False
        
        # Check dataset
        dataset = self.environment.get_dataset()
        if dataset is None or len(dataset) == 0:
            logger.error("Environment has no dataset")
            return False
        
        # Check required columns
        required_cols = {'prompt', 'answer'}
        if not required_cols.issubset(dataset.column_names):
            logger.error(f"Dataset missing columns: {required_cols - set(dataset.column_names)}")
            return False
        
        logger.info("✓ Environment validation passed")
        return True
    
    def test_environment(
        self,
        num_samples: int = 1,
        client = None
    ):
        """
        Test environment with sample rollouts.
        
        Args:
            num_samples: Number of samples to test
            client: OpenAI client for generation (optional)
        """
        if self.environment is None:
            logger.error("Environment not created")
            return
        
        logger.info(f"Testing environment with {num_samples} samples...")
        
        dataset = self.environment.get_dataset()
        
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\n{'='*60}")
            print(f"TEST SAMPLE {i+1}")
            print(f"{'='*60}")
            print(f"\nPrompt:\n{sample['prompt'][:300]}...")
            print(f"\nGround Truth:\n{sample['answer']}")
            
            if client:
                # Test with actual generation
                try:
                    # This would require vLLM or OpenAI client setup
                    # For now, just test reward function
                    pass
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(f"Generation test failed: {e}")
            
            # Test reward function with ground truth
            reward = self.environment.rubric.reward_functions[0](
                prompt=sample['prompt'],
                completion=sample['answer'],
                answer=sample['answer'],
                context=sample.get('context', '')
            )
            
            print(f"\nReward for ground truth: {reward:.3f}")
            print(f"{'='*60}\n")


def create_environment(
    dataset: Dataset,
    env_config,
    reward_config,
    reward_type: str = "combined",
    use_multi_turn: bool = False
) -> vf.Environment:
    """
    Convenience function to create environment.
    
    Args:
        dataset: Training dataset
        env_config: EnvironmentConfig
        reward_config: RewardConfig
        reward_type: Type of reward ("syntax", "execution", "semantic", "combined")
        use_multi_turn: Whether to use multi-turn environment
    
    Returns:
        Configured environment
    """
    factory = SQLEnvironment(env_config, reward_config)
    
    if use_multi_turn:
        env = factory.create_multi_turn_env(
            dataset,
            reward_type=reward_type,
            max_turns=env_config.max_turns
        )
    else:
        env = factory.create_single_turn_env(
            dataset,
            reward_type=reward_type
        )
    
    # Validate
    if not factory.validate_environment():
        raise ValueError("Environment validation failed")
    
    return env
