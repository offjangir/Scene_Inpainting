"""
Environment setup utilities for cache directories and paths.
"""
import os
from typing import Optional


def setup_cache_directories(
    cache_base: Optional[str] = None,
    use_scratch: bool = True,
    user: Optional[str] = None
):
    """
    Set up HuggingFace and PyTorch cache directories.
    
    Args:
        cache_base: Base directory for caches. If None, uses environment variables or defaults.
        use_scratch: If True and cache_base is None, use /scratch/{user}/hf_cache
        user: Username for scratch directory. If None, uses $USER environment variable.
    """
    if cache_base is None:
        if use_scratch:
            user = user or os.environ.get("USER", "user")
            cache_base = f"/scratch/{user}/hf_cache"
        else:
            cache_base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    # Set cache directories
    os.environ["HF_HOME"] = cache_base
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_base
    os.environ["TRANSFORMERS_CACHE"] = cache_base
    os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(cache_base), "torch_cache")
    
    return cache_base

