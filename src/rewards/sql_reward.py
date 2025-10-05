"""
SQL-specific reward functions for GRPO training.
Combines syntax validation, execution correctness, and semantic similarity.
"""
import re
from typing import Optional
import sqlparse

from .base_reward import BaseReward, RewardResult, CompositeReward
from .sql_executor import SQLExecutor, SchemaExtractor


class SyntaxReward(BaseReward):
    """
    Reward based on SQL syntax validity.
    Fast but doesn't guarantee correctness.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.executor = SQLExecutor()
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute syntax-based reward.
        
        Returns:
            RewardResult with syntax validity score
        """
        # Extract SQL from completion (handle XML tags if present)
        sql = self._extract_sql(completion)
        
        # Validate syntax
        is_valid, error = self.executor.validate_syntax(sql)
        
        if is_valid:
            # Additional checks for SQL quality
            quality_score = self._assess_quality(sql)
            final_score = 0.5 + (0.5 * quality_score)  # 0.5 to 1.0
            
            return RewardResult(
                score=final_score,
                details={
                    "syntax_valid": True,
                    "quality_score": quality_score,
                    "extracted_sql": sql[:100],  # First 100 chars
                }
            )
        else:
            return RewardResult(
                score=-0.5,
                details={
                    "syntax_valid": False,
                    "error": error,
                    "extracted_sql": sql[:100],
                }
            )
    
    def _extract_sql(self, completion: str) -> str:
        """Extract SQL from completion text."""
        # Remove XML tags if present
        sql = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL)
        sql = re.sub(r'<answer>(.*?)</answer>', r'\1', sql, flags=re.DOTALL)
        
        # Clean up
        sql = sql.strip()
        
        # If still has markdown code blocks, extract
        if '```sql' in sql.lower():
            match = re.search(r'```sql\s+(.*?)\s+```', sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1)
        elif '```' in sql:
            match = re.search(r'```\s+(.*?)\s+```', sql, re.DOTALL)
            if match:
                sql = match.group(1)
        
        return sql.strip()
    
    def _assess_quality(self, sql: str) -> float:
        """
        Assess SQL quality (formatting, structure).
        
        Returns:
            Quality score between 0 and 1
        """
        score = 1.0
        
        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return 0.0
            
            statement = parsed[0]
            tokens = [t for t in statement.tokens if not t.is_whitespace]
            
            # Check for basic structure
            if len(tokens) < 3:
                score -= 0.3
            
            # Check for proper capitalization (optional)
            sql_upper = sql.upper()
            keywords = ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 'ORDER BY']
            keyword_count = sum(1 for kw in keywords if kw in sql_upper)
            
            if keyword_count > 0:
                score += 0.1 * min(keyword_count, 3)
            
        except Exception:
            score = 0.5
        
        return max(0.0, min(1.0, score))


class ExecutionReward(BaseReward):
    """
    Reward based on SQL execution correctness.
    Compares generated SQL output with reference output.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        timeout = config.execution_timeout if config else 5.0
        max_rows = config.max_result_rows if config else 1000
        self.executor = SQLExecutor(timeout=timeout, max_rows=max_rows)
        self.schema_extractor = SchemaExtractor()
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute execution-based reward.
        
        Returns:
            RewardResult with execution correctness score
        """
        # Extract SQL from completion
        generated_sql = self._extract_sql(completion)
        reference_sql = answer.strip()
        
        # Extract schema from prompt/context
        schema = self._extract_schema(prompt, kwargs)
        
        if not schema:
            return RewardResult(
                score=-1.0,
                details={
                    "error": "No schema found",
                    "execution_success": False,
                }
            )
        
        # Execute both queries
        gen_result = self.executor.execute(generated_sql, schema)
        ref_result = self.executor.execute(reference_sql, schema)
        
        # Compare results
        if not gen_result.success:
            penalty = -0.8 if "timeout" in gen_result.error.lower() else -0.6
            return RewardResult(
                score=penalty,
                details={
                    "execution_success": False,
                    "error": gen_result.error,
                    "reference_success": ref_result.success,
                }
            )
        
        if not ref_result.success:
            # Reference failed - give partial credit if generated executes
            return RewardResult(
                score=0.3,
                details={
                    "execution_success": True,
                    "reference_success": False,
                    "note": "Reference query failed",
                }
            )
        
        # Compare results
        exact_match, similarity = self.executor.compare_results(
            gen_result, ref_result, order_matters=False
        )
        
        if exact_match:
            score = 1.0
        else:
            # Partial credit based on similarity
            score = similarity * 0.7 - 0.3  # Scale to [-0.3, 0.7]
        
        return RewardResult(
            score=score,
            details={
                "execution_success": True,
                "reference_success": True,
                "exact_match": exact_match,
                "similarity": similarity,
                "generated_rows": gen_result.row_count,
                "reference_rows": ref_result.row_count,
                "execution_time": gen_result.execution_time,
            }
        )
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL from text."""
        # Reuse syntax reward extraction
        syntax_reward = SyntaxReward()
        return syntax_reward._extract_sql(text)
    
    def _extract_schema(self, prompt: str, kwargs: dict) -> Optional[str]:
        """Extract schema from prompt or kwargs."""
        # Check kwargs first
        if 'schema' in kwargs:
            return kwargs['schema']
        
        if 'context' in kwargs:
            return self.schema_extractor.extract_from_context(kwargs['context'])
        
        # Try to extract from prompt
        if 'SCHEMA:' in prompt or 'CREATE TABLE' in prompt.upper():
            return self.schema_extractor.extract_from_context(prompt)
        
        return None


class SemanticReward(BaseReward):
    """
    Reward based on semantic similarity of SQL queries.
    Normalizes and compares query structure.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute semantic similarity reward.
        
        Returns:
            RewardResult with semantic similarity score
        """
        # Extract and normalize SQL
        generated_sql = self._extract_sql(completion)
        reference_sql = answer.strip()
        
        # Normalize both queries
        gen_normalized = self._normalize_sql(generated_sql)
        ref_normalized = self._normalize_sql(reference_sql)
        
        # Compare normalized queries
        if gen_normalized == ref_normalized:
            score = 1.0
            similarity = 1.0
        else:
            # Compute token-level similarity
            similarity = self._compute_token_similarity(
                gen_normalized, ref_normalized
            )
            score = similarity * 0.8 - 0.2  # Scale to [-0.2, 0.8]
        
        return RewardResult(
            score=score,
            details={
                "exact_match": gen_normalized == ref_normalized,
                "token_similarity": similarity,
                "generated_normalized": gen_normalized[:100],
                "reference_normalized": ref_normalized[:100],
            }
        )
    
    def _extract_sql(self, text: str) -> str:
        """Extract SQL from text."""
        syntax_reward = SyntaxReward()
        return syntax_reward._extract_sql(text)
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison."""
        try:
            # Parse and format
            formatted = sqlparse.format(
                sql,
                strip_comments=True,
                reindent=False,
                keyword_case='upper',
                identifier_case='lower'
            )
            
            # Remove extra whitespace
            normalized = ' '.join(formatted.split())
            return normalized
            
        except Exception:
            # Fallback to simple normalization
            return ' '.join(sql.upper().split())
    
    def _compute_token_similarity(
        self, 
        sql1: str, 
        sql2: str
    ) -> float:
        """
        Compute token-level similarity between queries.
        
        Uses Jaccard similarity on tokens.
        """
        tokens1 = set(sql1.split())
        tokens2 = set(sql2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0


class SQLReward(CompositeReward):
    """
    Combined SQL reward function.
    
    Combines syntax, execution, and semantic rewards with configurable weights.
    """
    
    def __init__(self, config=None):
        """
        Initialize SQL reward with components.
        
        Args:
            config: RewardConfig with weights and settings
        """
        # Create component rewards
        syntax_reward = SyntaxReward(config)
        execution_reward = ExecutionReward(config)
        semantic_reward = SemanticReward(config)
        
        # Get weights from config or use defaults
        if config:
            weights = [
                config.syntax_weight,
                config.execution_weight,
                config.semantic_weight,
            ]
        else:
            weights = [0.2, 0.6, 0.2]  # Default weights
        
        # Initialize composite reward
        super().__init__(
            rewards=[syntax_reward, execution_reward, semantic_reward],
            weights=weights,
            config=config
        )
    
    def compute(
        self,
        prompt: str,
        completion: str,
        answer: str,
        **kwargs
    ) -> RewardResult:
        """
        Compute combined SQL reward.
        
        Returns:
            RewardResult with combined score and component breakdown
        """
        result = super().compute(prompt, completion, answer, **kwargs)
        
        # Add SQL-specific metadata
        result.metadata = result.metadata or {}
        result.metadata.update({
            "reward_type": "sql_combined",
            "components": ["syntax", "execution", "semantic"],
            "weights": self.weights,
        })
        
        return result


# Factory function for easy instantiation
def create_sql_reward(config=None, reward_type: str = "combined") -> BaseReward:
    """
    Factory function to create SQL reward functions.
    
    Args:
        config: RewardConfig object
        reward_type: Type of reward ("syntax", "execution", "semantic", "combined")
    
    Returns:
        Appropriate reward function
    """
    reward_map = {
        "syntax": SyntaxReward,
        "execution": ExecutionReward,
        "semantic": SemanticReward,
        "combined": SQLReward,
    }
    
    if reward_type not in reward_map:
        raise ValueError(
            f"Unknown reward_type: {reward_type}. "
            f"Choose from {list(reward_map.keys())}"
        )
    
    return reward_map[reward_type](config)
