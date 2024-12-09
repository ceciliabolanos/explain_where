import numpy as np
import re

def process_importance_values(values, segment_size=500, step_size=250):
    """
    Process importance values using a Bartlett window approach for smoother transitions.
    
    Args:
        values: numpy array of importance values for each segment
        segment_size: size of each segment in samples (default: 50)
        step_size: number of samples between segment starts (default: 10)
    
    Returns:
        tuple: (processed_importance, timeline)
        - processed_importance: numpy array of smoothed importance values
        - timeline: numpy array of corresponding time points
    """
    # Calculate total duration
    num_segments = len(values)
    total_duration = (num_segments - 1) * step_size + segment_size
    timeline = np.arange(0, total_duration) / 1000
    
    # Create matrix for accumulating contributions
    accumulated_importance = np.zeros(total_duration)
    
    # For each segment, distribute its importance across its duration using Bartlett window
    for i, importance in enumerate(values):
        start_idx = i * step_size
        end_idx = start_idx + segment_size
        accumulated_importance[start_idx:end_idx] += importance 

    # Calculate overlap count for normalization
    overlap_count = np.zeros(total_duration)
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_size
        overlap_count[start_idx:end_idx] += 1
    
    # Avoid division by zero and normalize
    overlap_count = np.maximum(overlap_count, 1)
    processed_importance = accumulated_importance / overlap_count
    
    return processed_importance, timeline
    

def get_patterns():
    return [re.compile(rf'ft_\w+\.json$')]