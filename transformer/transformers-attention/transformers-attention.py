import torch
import math
def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    scores = Q @ K.transpose(-2, -1)
    d_k = Q.shape[-1]
    scores_scaled = scores / math.sqrt(d_k)
    weights = torch.softmax(scores_scaled, dim=-1)
    output = weights @ V
    return output