# ============================================
# config/__init__.py
# ============================================
"""Configuration module for SQL GRPO training."""

from text_to_sql_coder.training_config import (
    Config,
    ModelConfig,
    DataConfig,
    GRPOConfig,
    RewardConfig,
    EnvironmentConfig,
    HubConfig,
)

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "GRPOConfig",
    "RewardConfig",
    "EnvironmentConfig",
    "HubConfig",
]

# ============================================
# src/__init__.py
# ============================================
"""SQL GRPO Training Package."""

__version__ = "0.1.0"
__author__ = "Your Name"

# ============================================
# src/data/__init__.py
# ============================================
"""Data loading and preprocessing module."""

from .dataset_loader import (
    SQLDatasetLoader,
    DatasetPreprocessor,
    convert_to_verifiers_format,
    create_conversation_format,
)

__all__ = [
    "SQLDatasetLoader",
    "DatasetPreprocessor",
    "convert_to_verifiers_format",
    "create_conversation_format",
]

# ============================================
# src/rewards/__init__.py
# ============================================
"""Reward functions module."""

from .base_reward import (
    BaseReward,
    RewardResult,
    CompositeReward,
    CachedReward,
)
from .sql_executor import (
    SQLExecutor,
    ExecutionResult,
    SchemaExtractor,
)
from .sql_reward import (
    SyntaxReward,
    ExecutionReward,
    SemanticReward,
    SQLReward,
    create_sql_reward,
)

__all__ = [
    # Base classes
    "BaseReward",
    "RewardResult",
    "CompositeReward",
    "CachedReward",
    # Executor
    "SQLExecutor",
    "ExecutionResult",
    "SchemaExtractor",
    # SQL rewards
    "SyntaxReward",
    "ExecutionReward",
    "SemanticReward",
    "SQLReward",
    "create_sql_reward",
]

# ============================================
# src/models/__init__.py
# ============================================
"""Model loading module."""

from .model_loader import (
    ModelLoader,
    load_base_model_for_merge,
    print_model_info,
)

__all__ = [
    "ModelLoader",
    "load_base_model_for_merge",
    "print_model_info",
]

# ============================================
# src/training/__init__.py
# ============================================
"""Training module."""

from .environment import (
    SQLEnvironment,
    create_environment,
)
from .trainer import (
    GRPOTrainerWrapper,
    create_trainer,
)

__all__ = [
    "SQLEnvironment",
    "create_environment",
    "GRPOTrainerWrapper",
    "create_trainer",
]

# ============================================
# src/utils/__init__.py
# ============================================
"""Utilities module."""

from .hub_utils import (
    HubUploader,
    upload_to_hub,
)

__all__ = [
    "HubUploader",
    "upload_to_hub",
]

# ============================================
# tests/__init__.py
# ============================================
"""Tests module."""

# Empty init for tests
