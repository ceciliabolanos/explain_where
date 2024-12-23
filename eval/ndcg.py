import json
import numpy as np
from tqdm import tqdm
import re
import os
import pandas as pd
import argparse


def time_in_segmentation(max_index, list_of_tuples, granularidad_ms):
    interval_start = max_index
    interval_end = max_index + granularidad_ms/1000
    
    for tuple_start, tuple_end in list_of_tuples:
        # Calculate intersection
        intersection_start = max(interval_start, tuple_start)
        intersection_end = min(interval_end, tuple_end)
        
        # If there is an intersection, check its duration
        if intersection_end > intersection_start:
            intersection_duration = intersection_end - intersection_start
            if intersection_duration >= 0.1:
                return 1 # capaz intesection deberia ser el resultado
                # return add percentage of intersection to the result
    return 0

def generate_sequence(length):
    return [i * 0.250 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms): 
    # Initialize result vectors
    real_vector = np.zeros(len(times))
    
    for i in range(len(real_vector)):
        real_vector[i] = time_in_segmentation(times[i], times_gt, granularidad_ms)
    
    return real_vector

def calculate_ndcg(predictions, k=None):
    l = int(sum(predictions))  # number of ones in predictions
    # Calculate DCG
    dcg = 0
    for i in range(min(int(k), len(predictions))):
        if predictions[i]==1:
            dcg += predictions[i] / np.log2(i + 2)
    
    # Calculate IDCG
    idcg = 0
    for i in range(l):
        idcg += 1 / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0

def process_audio_file(data, method):
    id_explained = data['metadata']['id_explained']
    segment_length = data["metadata"]['segment_length']
    filename = data["metadata"]['filename']
    overlap = 0 #data["metadata"]['overlap']
    granularidad_ms = segment_length - overlap
    times_gt = data['metadata']["true_markers"]
    
    # Get importance scores
    if method == 'tree_importance':
        values = data['importance_scores']['random_forest'][method]['values']
    elif method == 'shap':
        values = data['importance_scores']['random_forest'][method]['values']
    elif method == 'naive':
        values = data['importance_scores'][method]['values']
    elif method == 'linear_regression':
        values = data['importance_scores'][method]['masked']['values']['coefficients']
   
    times = generate_sequence(len(values))
    
    times_segmentation = create_segmentation_vector(times_gt, times, granularidad_ms)

    importance_time_pairs = list(zip(values, times, times_segmentation))

    if method == 'naive':
        sorted_pairs_descending = sorted(importance_time_pairs, key=lambda x: x[0], reverse=True)
    else:
        sorted_pairs_descending = sorted(importance_time_pairs, key=lambda x: abs(x[0]), reverse=True)

    sorted_importances_d, sorted_times_d, sorted_times_segmentation = zip(*sorted_pairs_descending)

    random_k_scores = []
    random_total_scores = []
    
    for _ in range(5):
        randomized_vector = np.random.permutation(sorted_times_segmentation)
        random_k_scores.append(
            calculate_ndcg(randomized_vector, k=sum(sorted_times_segmentation))
        )
        random_total_scores.append(
            calculate_ndcg(randomized_vector, k=len(sorted_times_segmentation))
        )

    # Calculate average random scores
    random_k = sum(random_k_scores) / len(random_k_scores)
    random_total = sum(random_total_scores) / len(random_total_scores)

    # Calculate scores
    ndcg_values = {
        'ndcg_k': calculate_ndcg(sorted_times_segmentation, k=sum(sorted_times_segmentation)),
        'ndcg_k_random': random_k,
        'ndcg_total': calculate_ndcg(sorted_times_segmentation, k=len(sorted_times_segmentation)),
        'ndcg_total_random': random_total,
        'k': sum(sorted_times_segmentation),
        'random_k_scores': random_k_scores,
        'random_total_scores': random_total_scores
    }


    return {
        'filename': filename,
        'event_label': data['metadata']['label_explained'],
        'actual_score': data['metadata']['true_score'],
        **ndcg_values
    }

def get(method: str, base_path: str, mask_percentage, window_size):
    results = []
    pattern = re.compile(rf'ft_.*_p{mask_percentage}_m{window_size}\.json$')
    for root, _, files in tqdm(os.walk(os.path.join(base_path, 'audioset_audios_eval'))):
        json_files = [f for f in files if pattern.match(f)]
        
        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)

            result = process_audio_file(data, method)
            results.append(result)
        
    # Save results
    output_dir = os.path.join(base_path, 'audioset_evaluation/ndcg')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'ndcg_{method}_p{mask_percentage}_w{window_size}.tsv'), sep='\t', index=False)

def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/cbolanos/experiments',
                      help='Base path for AudioSet experiments')

    args = parser.parse_args()
    for method in ['tree_importance', 'shap', 'naive', 'linear_regression']:
        for mask_percentage in [0.1, 0.15, 0.2, 0.3, 0.4]:
            for window_size in [2, 3, 4, 5, 6]:
                get(method, args.base_path, mask_percentage, window_size)

if __name__ == '__main__':
    main()

    
    
    
    