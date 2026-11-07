
def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, memory_budget_bytes: int, token_ids: list, token_scores: list, eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with memory budget and eviction policy.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension per attention head
        dtype_bytes: Bytes per element (e.g., 2 for FP16)
        memory_budget_bytes: Total KV cache memory budget in bytes
        token_ids: List of token IDs arriving sequentially
        token_scores: List of importance scores per token
        eviction_policy: 'fifo' or 'score'
        num_protected: Number of initial tokens protected from eviction
    
    Returns:
        Dictionary with bytes_per_token, max_tokens, final_cache,
        evicted_tokens, and num_evictions
    """
    bytes_per_token = num_layers * num_heads * head_dim * 2 * dtype_bytes
    max_token = memory_budget_bytes // bytes_per_token
    # score_tensor = torch.tensor(token_scores)
    evicted_tokens = []
    num_evictions = 0
    
      #evict token
    if len(token_ids) <= max_token:
        final_cache = token_ids
    else:
        if num_protected == len(token_ids):
                final_cache = token_ids
        else: 
            num_evictions = len(token_ids) - max_token
            if eviction_policy == "fifo":
                final_cache = token_ids[:num_protected] + token_ids[num_protected+num_evictions:]
                evicted_tokens = token_ids[num_protected: num_protected + num_evictions]
            elif eviction_policy == "score":
                evicted_tokens = []
                final_cache = token_ids
                for i in range(num_evictions):
                    indices = sorted(
                        range(num_protected, len(token_scores[:max_token])),
                        key=lambda i: token_scores[i]
                    )[0]

                    evicted_tokens.append(final_cache.pop(indices))
                    token_scores.pop(indices)
                

        

    return{
        "bytes_per_token":bytes_per_token,
        "max_tokens": max_token,
        "final_cache":final_cache,
        "evicted_tokens":evicted_tokens,
        "num_evictions":num_evictions
    }