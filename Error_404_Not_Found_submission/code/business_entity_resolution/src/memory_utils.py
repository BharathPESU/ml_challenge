import os
import psutil
import logging

logger = logging.getLogger(__name__)

def get_memory_usage_mb() -> float:
    """Returns the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def report_memory(context: str = ""):
    """Logs the current memory usage."""
    mem_mb = get_memory_usage_mb()
    prefix = f"[{context}] " if context else ""
    logger.info(f"{prefix}Memory usage: {mem_mb:.1f} MB")
