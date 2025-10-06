"""
Base reward function interface.
Defines the contract for all reward functions.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RewardResult:
    """Structure for reward function results."""
    
    score: float  # Final reward score
    details: Dict[str, Any]  # Breakdown of reward components
    metadata: Optional[Dict[str, Any]] = None  # Additional info
    
    def __post_init__(self):
        """Validate reward score is in valid range."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Reward score {self.score} must be in [-1, 1]")


class BaseReward(ABC):
    """
    Abstract base class for reward functions.
    
    All reward functions should inherit from this class and implement
    the compute() method.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize reward function.
        
        Args:
            config: Configuration object (e.g., RewardConfig)
        """
        self.config = config
        self._call_count = 0
        self._total_reward = 0.0
    
    @abstractmethod
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute reward for a given completion.
        
        Args:
            prompt: The input prompt/question
            completion: Model's generated completion
            answer: Ground truth answer
            **kwargs: Additional context (e.g., schema, metadata)
        
        Returns:
            RewardResult with score and details
        """
        pass
    
    def __call__(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> float:
        """
        Make reward function callable. Returns scalar for verifiers.
        
        This is the interface expected by verifiers library.
        """
        try:
            result = self.compute(prompt, completion, answer, **kwargs)
            self._call_count += 1
            self._total_reward += result.score
            return result.score
        except Exception as e:  # pylint: disable=broad-except
            # Log error but don't crash training
            import logging
            logger = logging.getLogger(self.__class__.__name__)
            logger.error(f"Error in reward computation: {e}", exc_info=True)
            return -1.0  # Maximum penalty for errors
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics about reward function usage."""
        return {
            "call_count": self._call_count,
            "total_reward": self._total_reward,
            "average_reward": (
                self._total_reward / self._call_count 
                if self._call_count > 0 else 0.0
            ),
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self._call_count = 0
        self._total_reward = 0.0


class CompositeReward(BaseReward):
    """
    Combines multiple reward functions with weights.
    
    Example:
        syntax_reward = SyntaxReward()
        execution_reward = ExecutionReward()
        composite = CompositeReward(
            rewards=[syntax_reward, execution_reward],
            weights=[0.3, 0.7]
        )
    """
    
    def __init__(
        self,
        rewards: List[BaseReward],
        weights: Optional[List[float]] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize composite reward.
        
        Args:
            rewards: List of reward functions
            weights: List of weights (must sum to 1.0)
            config: Optional configuration
        """
        super().__init__(config)
        
        if not rewards:
            raise ValueError("Must provide at least one reward function")
        
        if weights is None:
            # Equal weights
            weights = [1.0 / len(rewards)] * len(rewards)
        
        if len(rewards) != len(weights):
            raise ValueError("Number of rewards must match number of weights")
        
        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        
        self.rewards = rewards
        self.weights = weights
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute weighted combination of all rewards.
        
        Returns:
            RewardResult with combined score and individual components
        """
        results = []
        details = {}
        
        for i, (reward_fn, weight) in enumerate(zip(self.rewards, self.weights)):
            result = reward_fn.compute(prompt, completion, answer, **kwargs)
            results.append(result)
            details[f"reward_{i}_{reward_fn.__class__.__name__}"] = {
                "score": result.score,
                "weight": weight,
                "weighted_score": result.score * weight,
                "details": result.details,
            }
        
        # Compute weighted sum
        final_score = sum(
            result.score * weight 
            for result, weight in zip(results, self.weights)
        )
        
        return RewardResult(
            score=final_score,
            details=details,
            metadata={"num_components": len(results)}
        )


class CachedReward(BaseReward):
    """
    Wrapper that caches reward computations.
    Useful for expensive reward functions.
    """
    
    def __init__(
        self,
        reward_fn: BaseReward,
        cache_size: int = 10000
    ):
        """
        Initialize cached reward.
        
        Args:
            reward_fn: Underlying reward function
            cache_size: Maximum cache size
        """
        super().__init__()
        self.reward_fn = reward_fn
        self.cache: Dict[tuple, RewardResult] = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _make_key(
        self,
        prompt: str,
        completion: str,
        answer: str
    ) -> tuple:
        """Create cache key from inputs."""
        return (hash(prompt), hash(completion), hash(answer))
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """Compute reward with caching."""
        key = self._make_key(prompt, completion, answer)
        
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        self.cache_misses += 1
        result = self.reward_fn.compute(prompt, completion, answer, **kwargs)
        
        # Add to cache if not full
        if len(self.cache) < self.cache_size:
            self.cache[key] = result
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
