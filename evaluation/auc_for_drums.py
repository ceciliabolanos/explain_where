import numpy as np
from sklearn.metrics import roc_curve, auc
import argparse
import os
import re
import json
from tqdm import tqdm
import pandas as pd

def time_in_segmentation(segment_start, list_of_tuples, granularidad_ms, final):
    """
    Determines whether a segment is in the valid region, discard region, or outside any marker.
    
    Parameters:
        segment_start (float): Start time of the segment.
        list_of_tuples (list of tuples): Each tuple is (tuple_start, tuple_end) representing a ground-truth marker.
        granularidad_ms (float): Duration of the segment in milliseconds.
        final (float): Duration (in seconds) from tuple_start defining the valid portion.
        
    Returns:
        int: 
            1  if the segment overlaps the valid region [tuple_start, tuple_start+final],
           -1  if it overlaps the discard region [tuple_start+final, tuple_end],
            0  if it does not belong to any ground-truth marker.
    """
    interval_start = segment_start
    interval_end = segment_start + granularidad_ms / 1000.0

    for tuple_start, tuple_end in list_of_tuples:
        valid_tuple_end = tuple_start + final
        if tuple_end < valid_tuple_end:
            valid_tuple_end = tuple_end
        # Check if there is any overlap with the valid region
        intersection_valid_start = max(interval_start, tuple_start)
        intersection_valid_end = min(interval_end, valid_tuple_end)
        if intersection_valid_end > intersection_valid_start:
            return 1
        # Check if there is any overlap with the discard region
        intersection_discard_start = max(interval_start, valid_tuple_end)
        intersection_discard_end = min(interval_end, tuple_end)
        if intersection_discard_end > intersection_discard_start:
            return -1  # Discard this segment
    
    return 0

def generate_sequence(length):
    """
    Generates a list of segment start times.
    
    For this example, we use a constant step of 0.1 seconds.
    """
    return [i * 0.1 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms, final):
    """
    Creates a segmentation vector (labels for each segment).
    
    Parameters:
        times_gt (list): List of ground truth tuples.
        times (list): List of segment start times.
        granularidad_ms (float): Duration of the segment in ms.
        final (float): Duration in seconds from tuple_start defining valid region.
    
    Returns:
        numpy.array: An array where each entry is:
                     1  (if the segment is valid),
                    -1  (if the segment is in the discard region),
                     0  (if the segment is outside any marker).
    """
    real_vector = np.zeros(len(times))
    for i, t in enumerate(times):
        real_vector[i] = time_in_segmentation(t, times_gt, granularidad_ms, final)
    return real_vector

def process_audio_file(data, method, final):
    """
    Processes an audio file: it retrieves the model scores, creates the segmentation
    vector, filters out discard segments, and then computes the ROC curve and AUC.
    
    Parameters:
        data (dict): Contains metadata and importance scores.
        method (str): Which model’s importance scores to use.
        final (float): Duration (in seconds) from tuple_start defining the valid region.
    
    Returns:
        dict or None: A dictionary with the results, or None if ROC cannot be computed.
    """
    # Extract metadata
    segment_length = data["metadata"]['segment_length']  # assumed in ms
    filename = data["metadata"]['filename']
    overlap = 0  # adjust if necessary
    granularidad_ms = segment_length - overlap
    times_gt = data['metadata']["true_markers"]

    # Retrieve model importance scores based on the chosen method.
    if method == 'tree_importance':
        values = data['importance_scores']['random_forest_tree_importance']['values']
    elif method == 'linear_regression':
        values = data['importance_scores']['linear_regression']['values']['coefficients']
    elif method == 'kernel_shap':
        values = data['importance_scores']['kernel_shap']['values']['coefficients']
    elif method == 'linear_regression_noreg_noweights':
        values = data['importance_scores']['linear_regression_nocon']['values']['coefficients']
    elif method == 'kernel_shap_sumcons':
        values = data['importance_scores']['importances_kernelshap_analyzer_1constraint']['values']['coefficients']    

    # Generate segment start times.
    times = generate_sequence(len(values))
    
    # Create the segmentation vector (labels: 1, -1, or 0)
    segmentation_labels = create_segmentation_vector(times_gt, times, granularidad_ms, final)

    # Filter out segments that are in the discard region (i.e., with label -1).
    valid_mask = segmentation_labels != -1
    filtered_segmentation = segmentation_labels[valid_mask]
    filtered_values = np.array(values)[valid_mask]

    # Reasoning:
    # We remove segments from the discard region so that only segments that are either
    # valid (1) or negative (0) are used to compute the ROC curve and AUC.
    if not (np.any(filtered_segmentation == 1) and np.any(filtered_segmentation == 0)):
        print(f"Warning: Not enough class diversity in file {filename} for ROC computation.")
        return None

    # Compute ranking scores.
    # Here we rank the importance scores: higher scores receive higher ranks.
    sorted_indices = np.argsort(filtered_values)[::-1]  
    ranking_scores = np.zeros_like(filtered_values, dtype=float)
    ranking_scores[sorted_indices] = np.linspace(1, 0, len(filtered_values))

    # Compute ROC curve and AUC using the filtered values.
    fpr, tpr, thresholds = roc_curve(filtered_segmentation, ranking_scores)
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    # Calculate scores
    order_segmentation_values = {
        'real_order': filtered_segmentation, # Original order of segmentation values (0s and 1s)
        'model_order': filtered_values, # Segmentation values sorted by importance
        'roc_auc': roc_auc,
    }

    return {
        'filename': filename,
        'event_label': data['metadata']['id_explained'],
        'actual_score': data['metadata']['true_score'],
        **order_segmentation_values,
        'true_markers': times_gt,
    }

def get(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset, final):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}. Retrying {file_path}")
           
            result = process_audio_file(data, method, final)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_test/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'order_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_{final}.tsv'), sep='\t', index=False)


def get_with_name(method: str, base_path: str, dataset, name, final):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft1_.*_{name}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}. Retrying {file_path}")
           
            result = process_audio_file(data, method, final)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'order_{method}_{name}_{final}.tsv'), sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results1',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    names = ["zeros", "noise"] 

    for name in names:
        for final in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
                get_with_name(method, args.base_path, 'drums', name, final)


if __name__ == '__main__':
    main()

    