import numpy as np

def sliding_kv_cache_update(cache_k, cache_v, current_len, new_k, new_v, window_size):
    """
    Update a pre-allocated sliding-window KV cache and produce the causal mask
    for the new queries against the updated cache.

    Args:
        cache_k, cache_v: nested lists / arrays of shape
            (batch, num_heads, window_size, head_dim).
        current_len: int, number of valid entries currently in the cache (<= window_size).
        new_k, new_v: nested lists / arrays of shape
            (batch, num_heads, new_len, head_dim).
        window_size: int, maximum cache length.

    Returns:
        dict with keys 'cache_k', 'cache_v', 'current_len', 'mask'.
    """
    #list(batch, num_heads, window_size, head_dim)
    #list(batch, num_heads, len, head_idm)
    cache_k = np.array(cache_k, dtype=float, copy=True)
    cache_v = np.array(cache_v, dtype=float, copy=True)
    new_k = np.array(new_k)
    new_v = np.array(new_v)
    new_len = new_k.shape[2]
    updated_len = min(current_len + new_len, window_size)

    valid_len = current_len

    for i in range(new_len):
        if valid_len < window_size:
            cache_k[:,:, valid_len, :] = new_k[:,:,i,:]
            cache_v[:,:, valid_len,:] = new_v[:,:,i,:]
            valid_len += 1

        else:
            #pop 
            cache_k[:,:, :-1, :] = cache_k[:,:,1:,:].copy()
            cache_v[:,:, :-1, :] = cache_v[:,:,1:,:].copy()

            #insert
            cache_k[:,:,-1,:] = new_k[:,:,i,:]
            cache_v[:,:,-1,:] = new_v[:,:,i,:]

    
    mask = []
    for i in range(new_len):
        current_position = max(updated_len - new_len + i , 0)
        row = []

        for j in range(updated_len):
            if j <= current_position:
                row.append(1)
            else:
                row.append(0)

        mask.append(row)

    return {
        "cache_k": cache_k.tolist(),
        "cache_v": cache_v.tolist(),
        "current_len": updated_len,
        "mask": mask
    }







