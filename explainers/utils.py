import numpy as np

def compute_log_odds(data, label_to_explain):
    """
    Computes log(p / (1 - p)) for the target class, where p is the softmax probability 
    of the given class.

    Parameters:
    - data: dict, must contain 'scores' as a list of logits (list of lists or 2D array).
    - label_to_explain: int, index of the class to compute the log-odds for.

    Returns:
    - log_odds: np.ndarray, log-odds values for the target class across all samples.
    """
    def softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability trick
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    logits = np.array(data)  # Convert to NumPy array
    probs = softmax(logits)  # Apply softmax
    log_odds = np.log(probs / (1 - probs))
    log_odds = log_odds[:, label_to_explain]

    return log_odds
