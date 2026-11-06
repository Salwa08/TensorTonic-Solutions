
import numpy as np

def apply_causal_mask(scores: list, mask_value: float = -1e9) -> np.ndarray:
    """
    Returns a causally masked NumPy array matching the shape of scores.
    """
    scores = np.asarray(scores, dtype=float)

    # Create a mask for future positions
    mask = np.triu(
        np.ones(scores.shape[-2:], dtype=bool),
        k=1
    )

    # Replace future-position scores
    result = np.where(mask, mask_value, scores)

    return result
