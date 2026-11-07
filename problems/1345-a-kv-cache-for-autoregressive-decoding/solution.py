import torch
import torch.nn as nn


class KVCache:
    def __init__(self):
        # TODO: start empty
        self.k = None
        self.v = None
        self.cache = (self.k, self.v)

    def append(self, k, v):
        # TODO: concatenate k, v onto the cache along the sequence dim (dim=2)
        # TODO: return the full (k, v) accumulated so far
        if self.k is None:
            self.k = k
            self.v = v
        else: 
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return (self.k, self.v)
        

    def __len__(self):
        # TODO: number of cached positions, 0 when empty
        if self.k is None and self.v is None:
            return 0
        else:
            return self.k.shape[2]

    def reset(self):
        # TODO: drop everything
        self.k = None
        self.v = None
        self.cache = (self.k, self.v)



def attend_with_cache(q, k_new, v_new, cache):
    # TODO: append the new keys/values, then attend q over the full cache
    # TODO: softmax(q @ k^T / sqrt(head_dim)) @ v
    k_all, v_all = cache.append(k_new, v_new)

    wei = q @ k_all.transpose(-2, -1)
    wei = wei / (q.shape[-1] ** 0.5)

    wei = nn.functional.softmax(wei, dim =-1)

    output = wei @ v_all

    return output