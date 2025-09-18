# src/cement_ai_platform/utils/retry_decorator.py
import time
import functools
import logging
from typing import Tuple, Callable, Any

logger = logging.getLogger(__name__)

def retry(
    exceptions: Tuple[Exception, ...] = (Exception,),
    total_tries: int = 5,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
):
    """
    Retry decorator with exponential backoff for handling transient failures.
    
    Args:
        exceptions (Tuple[Exception, ...]): Exception types to retry on
        total_tries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        backoff_factor (float): Multiplier for delay after each retry
        max_delay (float): Maximum delay between retries in seconds
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            
            for attempt in range(1, total_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == total_tries:
                        logger.error(
                            f"All {total_tries} attempts failed for {func.__name__}: {e}"
                        )
                        raise e
                    
                    logger.warning(
                        f"Attempt {attempt}/{total_tries} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            
            # This should never be reached, but just in case
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def retry_gcp_operation(
    total_tries: int = 3,
    initial_delay: float = 2.0
):
    """
    Specialized retry decorator for Google Cloud Platform operations.
    
    Args:
        total_tries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        
    Returns:
        Callable: Decorated function with GCP-specific retry logic
    """
    from google.api_core import exceptions as gcp_exceptions
    
    return retry(
        exceptions=(
            gcp_exceptions.ServiceUnavailable,
            gcp_exceptions.DeadlineExceeded,
            gcp_exceptions.InternalServerError,
            gcp_exceptions.TooManyRequests,
            ConnectionError,
            TimeoutError
        ),
        total_tries=total_tries,
        initial_delay=initial_delay,
        backoff_factor=2.0,
        max_delay=30.0
    )

def retry_network_operation(
    total_tries: int = 3,
    initial_delay: float = 1.0
):
    """
    Specialized retry decorator for network operations.
    
    Args:
        total_tries (int): Maximum number of retry attempts
        initial_delay (float): Initial delay between retries in seconds
        
    Returns:
        Callable: Decorated function with network-specific retry logic
    """
    return retry(
        exceptions=(
            ConnectionError,
            TimeoutError,
            OSError,
            Exception  # Catch-all for network issues
        ),
        total_tries=total_tries,
        initial_delay=initial_delay,
        backoff_factor=1.5,
        max_delay=10.0
    )
