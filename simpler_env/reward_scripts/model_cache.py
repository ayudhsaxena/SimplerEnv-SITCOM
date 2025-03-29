"""
Shared model caching system for large models used across different scripts.
"""

import gc
import torch

# Caches for different model types
_VIT_MODEL_CACHE = {}
_AGENT_CACHE = {}
_GROUNDED_SAM2_CACHE = {}

def get_sam2_from_cache(
    sam2_checkpoint,
    sam2_model_config,
    grounding_model,
    box_threshold,
    text_threshold,
    output_dir=None,
    object_name='target',
):
    """
    Get a GroundedSAM2 instance from cache or create a new one if not found.
    
    Args:
        sam2_checkpoint: Path to the SAM2 model checkpoint
        sam2_model_config: SAM2 model configuration file
        grounding_model: Hugging Face model ID for Grounding DINO
        box_threshold: Confidence threshold for bounding boxes
        text_threshold: Confidence threshold for text prompts
        output_dir: Directory to save results (optional)
        
    Returns:
        GroundedSAM2 instance
    """
    from .grounded_sam_2 import GroundedSAM2
    
    # Create a cache key based on the model parameters
    cache_key = (
        f"{sam2_checkpoint}_{sam2_model_config}_{grounding_model}_{box_threshold}_{text_threshold}_{object_name}"
    )
    
    # Check if we have this model in the cache
    if cache_key in _GROUNDED_SAM2_CACHE:
        return _GROUNDED_SAM2_CACHE[cache_key]
    
    # Model not in cache, create a new one
    print(f"Loading new GroundedSAM2 model")
    grounded_sam2 = GroundedSAM2(
        sam2_checkpoint=sam2_checkpoint,
        sam2_model_config=sam2_model_config,
        grounding_model=grounding_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_dir=output_dir
    )
    
    # Cache the model for future use
    _GROUNDED_SAM2_CACHE[cache_key] = grounded_sam2
    
    return grounded_sam2

def clear_sam2_cache():
    """Clear the GroundedSAM2 model cache."""
    global _GROUNDED_SAM2_CACHE
    _GROUNDED_SAM2_CACHE.clear()

def clear_vit_cache():
    """Clear the ViT model cache."""
    global _VIT_MODEL_CACHE
    _VIT_MODEL_CACHE.clear()
    
def clear_all_caches():
    """Clear all model caches to free up memory."""
    clear_sam2_cache()
    clear_vit_cache()
    
    global _AGENT_CACHE
    _AGENT_CACHE.clear()
    
    # Run garbage collection
    gc.collect()
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("All model caches have been cleared")