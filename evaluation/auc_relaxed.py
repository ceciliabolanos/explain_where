import numpy as np
from sklearn.metrics import roc_curve, auc
import argparse
import os
import re
import json
from tqdm import tqdm
import pandas as pd

def time_in_segmentation(max_index, list_of_tuples, granularidad_ms, interesection_duration_threshold):
    interval_start = max_index
    interval_end = max_index + granularidad_ms/1000
    
    for tuple_start, tuple_end in list_of_tuples:

        intersection_start = max(interval_start, tuple_start)
        intersection_end = min(interval_end, tuple_end)
        
        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            if intersection_duration > interesection_duration_threshold:
                return 1
            elif intersection_duration > 0:
                return -1
    return 0

def generate_sequence(length):
    """
    Generates a list of segment start times.
    
    For this example, we use a constant step of 0.1 seconds.
    """
    return [i * 0.1 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms, intersection):
    """
    Creates a segmentation vector (labels for each segment).
    
    Parameters:
        times_gt (list): List of ground truth tuples.
        times (list): List of segment start times.
        granularidad_ms (float): Duration of the segment in ms.
        intersection (float): Duration in seconds from tuple_start defining valid region.
    
    Returns:
        numpy.array: An array where each entry is:
                     1  (if the segment is valid),
                    -1  (if the segment is in the discard region),
                     0  (if the segment is outside any marker).
    """
    real_vector = np.zeros(len(times))
    for i, t in enumerate(times):
        real_vector[i] = time_in_segmentation(t, times_gt, granularidad_ms, intersection)
    return real_vector

def process_audio_file(data, method, intersection):
    """
    Processes an audio file: it retrieves the model scores, creates the segmentation
    vector, filters out discard segments, and then computes the ROC curve and AUC.
    
    Parameters:
        data (dict): Contains metadata and importance scores.
        method (str): Which model’s importance scores to use.
        intersection (float): Duration (in seconds) from tuple_start defining the valid region.
    
    Returns:
        dict or None: A dictionary with the results, or None if ROC cannot be computed.
    """
    # Extract metadata
    segment_length = data["metadata"]['segment_length']  # assumed in ms
    filename = data["metadata"]['filename']
    overlap = 0  # adjust if necessary
    granularidad_ms = segment_length - overlap
    times_gt = data['metadata']["true_markers"]

    if method == 'RF':
        values = data['importance_scores']['RF']['values']
    elif method == 'SHAP':
        values = data['importance_scores']['SHAP']['values']
    elif method == 'LR':
        values = data['importance_scores']['LR']['values']
    
    times = generate_sequence(len(values))
    
    # Create the segmentation vector (labels: 1, -1, or 0)
    segmentation_labels = create_segmentation_vector(times_gt, times, granularidad_ms, intersection)

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

def get(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset, intersection):
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
           
            result = process_audio_file(data, method, intersection)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'auc_relaxed_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_{intersection}.tsv'), sep='\t', index=False)


def get_audioset(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset, intersection, id):
    results = []
    selected_files = pd.read_csv('/home/ec2-user/explain_where/datasets/audioset/audioset.csv')

    for i in tqdm(range(len(selected_files))):
        id1 = int(selected_files.loc[i]['event_label'])
        filename = selected_files.loc[i]['filename']
        file_path = f'{base_path}/explanations_{dataset}/{filename}/ast/ft_{id1}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.json'
        if os.path.exists(file_path):
            if id1 == id:
                try:
                    with open(file_path, "r") as file:
                        data = json.load(file)
                    result = process_audio_file(data, method, intersection)
                except json.JSONDecodeError as e:
                        print(f"JSON error: {e}. Skipping {file_path}")
                        result = None  # Append None in case of JSON error
                results.append(result)
    
    if id == 0:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_speech/')
    if id == 137:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_music/')
    if id == 74:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_dog/')
    os.makedirs(output_dir, exist_ok=True)

    results = [r for r in results if r is not None]    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'auc_relaxed_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_{intersection}.tsv'), sep='\t', index=False)


def get_with_name_audioset(method: str, name, base_path: str, dataset, intersection, id):
    results = []
    selected_files = pd.read_csv('/home/ec2-user/explain_where/datasets/audioset/audioset.csv')
    for i in tqdm(range(len(selected_files))):
        id1 = int(selected_files.loc[i]['event_label'])
        filename = selected_files.loc[i]['filename']
        file_path = f'{base_path}/explanations_{dataset}/{filename}/ast/ft_{id}_{name}.json'
        if os.path.exists(file_path):
            if id1 == id:
                try:
                    with open(file_path, "r") as file:
                        data = json.load(file)
                    result = process_audio_file(data, method, intersection)
                except json.JSONDecodeError as e:
                        print(f"JSON error: {e}. Skipping {file_path}")
                        result = None  # Append None in case of JSON error
                results.append(result)
           
    if id == 0:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_speech/')
    if id == 137:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_music/')
    if id == 74:
        output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}_dog/')
    os.makedirs(output_dir, exist_ok=True)

    results = [r for r in results if r is not None]
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'auc_relaxed_{method}_{name}_{intersection}.tsv'), sep='\t', index=False)

                   
def get_with_name(method: str, name, base_path: str, dataset, intersection):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_{name}\.json$')
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            try:
                with open(file_path, "r") as file:
                    data = json.load(file)
                result = process_audio_file(data, method, intersection)
            except json.JSONDecodeError as e:
                    print(f"JSON error: {e}. Skipping {file_path}")
                    result = None  # Append None in case of JSON error
            results.append(result)
       
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'auc_relaxed_{method}_{name}_{intersection}.tsv'), sep='\t', index=False)
                   
def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    names = ["zeros", "noise"]

    # # datasets = ['cough', 'drums', 'kws']
    datasets =  ['nocough']
    # for dataset in datasets:    
    #     for name in names:
    #         for method in ['SHAP', 'LR', 'RF']:        
    #             get_with_name(method, name, args.base_path, dataset, 0.09)
                
    for dataset in datasets:
        for mask_type in ['zeros', 'noise']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['SHAP', 'LR', 'RF']:
                        get(method, mask_percentage, window_size, mask_type, 'euclidean', args.base_path, dataset, 0.09)

    # for dataset in ['audioset']:    
    #     for id in [0, 137, 74]:
    #         for name in names:
    #             for method in ['SHAP', 'LR', 'RF']:
    #                 get_with_name_audioset(method, name, args.base_path, dataset, 0.09, id)
                    

    # for id in [0, 137, 74]:
    #     for mask_type in ['zeros', 'noise']:
    #         for mask_percentage in [0.2, 0.3, 0.4]:
    #             for window_size in [1, 3, 5]:
    #                 for method in ['SHAP', 'LR', 'RF']:
    #                     get_audioset(method, mask_percentage, window_size, mask_type, 'euclidean', args.base_path, 'audioset', 0.09, id)
    
    
if __name__ == '__main__':
    main()

    