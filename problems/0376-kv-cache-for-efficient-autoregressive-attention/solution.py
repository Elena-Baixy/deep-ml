import numpy as np

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    k = x_new @ W_K
    q = x_new @ W_Q
    v = x_new @ W_V
    #(,d_k)

    if cache is None:
        K = k.reshape(1,-1)
        V = v.reshape(1,-1)
    else:
        K = np.vstack([cache[0], k.reshape(1,-1)])
        V = np.vstack([cache[1], v.reshape(1,-1)])

    wei = q @ K.T
    wei = wei / np.sqrt(W_K.shape[1])
    
    wei_shifted = wei - max(wei)
    exp_score = np.exp(wei_shifted) 
    wei = exp_score / np.sum(exp_score)

    output = wei @ V

    return (output, (K,V))

    


