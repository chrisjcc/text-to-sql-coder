"""
Safe SQL execution environment for reward validation.
Provides sandboxed SQL execution with timeout and resource limits.
"""
import sqlite3
import time
import hashlib
import sys
import threading
from typing import List, Tuple, Optional, Any, Dict
from contextlib import contextmanager
from dataclasses import dataclass
import signal


class TimeoutException(Exception):
    """Raised when SQL execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("SQL execution timed out")


@dataclass
class ExecutionResult:
    """Result of SQL execution."""
    
    success: bool
    results: Optional[List[Tuple]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    row_count: int = 0
    
    def __eq__(self, other):
        """Compare execution results."""
        if not isinstance(other, ExecutionResult):
            return False
        if not self.success or not other.success:
            return False
        return self.results == other.results


class SQLExecutor:
    """
    Safe SQL executor with sandboxing and timeout.
    
    Uses SQLite in-memory database for safe execution.
    """
    
    def __init__(
        self,
        timeout: float = 5.0,
        max_result_rows: int = 1000
    ):
        """
        Initialize SQL executor.
        
        Args:
            timeout: Maximum execution time in seconds
            max_result_rows: Maximum number of rows to return
        """
        self.timeout = timeout
        self.max_result_rows = max_result_rows
        self._cache: Dict[str, Any] = {}
    
    def _create_database(
        self,
        schema: str
    ) -> sqlite3.Connection:
        """
        Create in-memory database with schema.
        
        Args:
            schema: SQL schema statements
        
        Returns:
            SQLite connection
        """
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row  # Enable column access
        
        try:
            # Execute schema
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
            return conn
        except Exception as e:
            conn.close()
            raise ValueError(f"Failed to create schema: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """
        Normalize SQL query for comparison.
        
        Args:
            query: SQL query string
        
        Returns:
            Normalized query
        """
        # Remove comments, extra whitespace
        import sqlparse
        
        # Parse and format
        parsed = sqlparse.parse(query)
        if not parsed:
            return query.strip()
        
        formatted = sqlparse.format(
            query,
            strip_comments=True,
            reindent=False,
            keyword_case='upper'
        )
        
        # Remove extra whitespace
        return ' '.join(formatted.split())
    
    @contextmanager
    def _timeout_context(self):
        """Context manager for query timeout with cross-platform support."""
        # Check if SIGALRM is available (Unix-like systems)
        if hasattr(signal, 'SIGALRM') and sys.platform != 'win32':
            # Unix: Use signal-based timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout))
            try:
                yield
            finally:
                signal.alarm(0)
        else:
            # Windows: Use threading-based timeout (less precise but portable)
            result = {'exception': None}
            
            def target():
                try:
                    pass  # The actual execution happens in the caller
                except Exception as e:
                    result['exception'] = e
            
            # Note: For Windows, we'll implement timeout differently in execute()
            # This context manager becomes a no-op placeholder
            yield
    
    def _execute_with_thread_timeout(self, cursor, query):
        """Execute query with threading-based timeout (Windows compatible)."""
        result = {'data': None, 'exception': None}
        
        def target():
            try:
                cursor.execute(query)
                result['data'] = cursor.fetchmany(self.max_result_rows)
            except Exception as e:
                result['exception'] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            # Timeout occurred - we can't cleanly kill the thread, but we can return timeout error
            raise TimeoutException("Query execution timed out (Windows)")
        
        if result['exception']:
            raise result['exception']
        
        return result['data']
    
    def execute(
        self,
        query: str,
        schema: str,
        use_timeout: bool = True
    ) -> ExecutionResult:
        """
        Execute SQL query safely.
        
        Args:
            query: SQL query to execute
            schema: Database schema
            use_timeout: Whether to enforce timeout
        
        Returns:
            ExecutionResult with results or error
        """
        # Create cache key
        cache_key = hashlib.md5(
            f"{query}:{schema}".encode()
        ).hexdigest()
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        conn = None
        start_time = time.time()
        
        try:
            # Create database with schema
            conn = self._create_database(schema)
            cursor = conn.cursor()
            
            # Execute query with optional timeout
            if use_timeout:
                if hasattr(signal, 'SIGALRM') and sys.platform != 'win32':
                    # Unix: Use signal-based timeout
                    with self._timeout_context():
                        cursor.execute(query)
                        results = cursor.fetchmany(self.max_result_rows)
                else:
                    # Windows: Use threading-based timeout
                    results = self._execute_with_thread_timeout(cursor, query)
            else:
                cursor.execute(query)
                results = cursor.fetchmany(self.max_result_rows)
            
            # Convert Row objects to tuples
            results = [tuple(row) for row in results]
            
            execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                results=results,
                execution_time=execution_time,
                row_count=len(results)
            )
            
            # Cache result
            self._cache[cache_key] = result
            
            return result
            
        except TimeoutException:
            return ExecutionResult(
                success=False,
                error="Query execution timed out",
                execution_time=self.timeout
            )
        except sqlite3.Error as e:
            return ExecutionResult(
                success=False,
                error=f"SQL error: {str(e)}",
                execution_time=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Unexpected error: {str(e)}",
                execution_time=time.time() - start_time
            )
        finally:
            if conn:
                conn.close()
    
    def compare_results(
        self,
        result1: ExecutionResult,
        result2: ExecutionResult,
        order_matters: bool = False
    ) -> Tuple[bool, float]:
        """
        Compare two execution results.
        
        Args:
            result1: First execution result
            result2: Second execution result
            order_matters: Whether row order should match
        
        Returns:
            Tuple of (exact_match, similarity_score)
        """
        # Both must succeed
        if not result1.success or not result2.success:
            return False, 0.0
        
        # Compare row counts
        if result1.row_count != result2.row_count:
            # Partial credit based on overlap
            min_count = min(result1.row_count, result2.row_count)
            max_count = max(result1.row_count, result2.row_count)
            size_similarity = min_count / max_count if max_count > 0 else 0.0
            return False, size_similarity * 0.3
        
        # Compare results
        if order_matters:
            exact_match = result1.results == result2.results
            similarity = 1.0 if exact_match else 0.0
        else:
            # Convert to sets for unordered comparison
            set1 = set(result1.results)
            set2 = set(result2.results)
            exact_match = set1 == set2
            
            # Jaccard similarity for partial credit
            if not exact_match:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 1.0
        
        return exact_match, similarity
    
    def validate_syntax(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL syntax without execution.
        
        Args:
            query: SQL query
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            import sqlparse
            
            # Parse query
            parsed = sqlparse.parse(query)
            
            if not parsed:
                return False, "Empty or invalid query"
            
            # Check for basic SQL structure
            statement = parsed[0]
            if not statement.tokens:
                return False, "No tokens found"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def clear_cache(self):
        """Clear execution result cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached results."""
        return len(self._cache)


class SchemaExtractor:
    """
    Extracts and validates database schemas.
    """
    
    @staticmethod
    def extract_from_context(context: str) -> str:
        """
        Extract schema from context string.
        
        Args:
            context: Full context including schema
        
        Returns:
            Extracted schema SQL
        """
        # Common patterns for schema extraction
        if "CREATE TABLE" in context.upper():
            # Schema is likely complete
            return context
        
        # Try to find schema markers
        lines = context.split('\n')
        schema_lines = []
        in_schema = False
        
        for line in lines:
            upper_line = line.upper()
            if "CREATE" in upper_line or "INSERT" in upper_line:
                in_schema = True
            if in_schema:
                schema_lines.append(line)
        
        return '\n'.join(schema_lines)
    
    @staticmethod
    def validate_schema(schema: str) -> Tuple[bool, Optional[str]]:
        """
        Validate schema can be executed.
        
        Args:
            schema: SQL schema statements
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            conn = sqlite3.connect(':memory:')
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.close()
            return True, None
        except Exception as e:
            return False, str(e)
