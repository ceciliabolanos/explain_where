import json
import numpy as np
from tqdm import tqdm
import re
import os
import pandas as pd
import argparse
from sklearn.metrics import roc_curve, auc

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
    return 0

def generate_sequence(length):
    return [i * 0.1 for i in range(length)]

def create_segmentation_vector(times_gt, times, granularidad_ms, intersection): 
    real_vector = np.zeros(len(times))

    for i in range(len(real_vector)):
        real_vector[i] = time_in_segmentation(times[i], times_gt, granularidad_ms, intersection)
    
    return real_vector


def process_audio_file(data, method, intersection):
    granularidad_ms = data["metadata"]['segment_length']
    filename = data["metadata"]['filename']
    times_gt = data['metadata']["true_markers"]
    
    # Get importance scores
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
    times = generate_sequence(len(values))
    
    times_segmentation = create_segmentation_vector(times_gt, times, granularidad_ms, intersection)

    sorted_indices = np.argsort(values)[::-1]  
    ranking_scores = np.zeros_like(values)
    ranking_scores[sorted_indices] = np.linspace(1, 0, len(values))
    
    fpr, tpr, thresholds = roc_curve(np.array(times_segmentation), ranking_scores)
    
    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    if sum(times_segmentation) == 0:
        print(filename)
    
    order_segmentation_values = {
        'real_order': times_segmentation, 
        'model_order': values,
        'roc_auc': roc_auc,
        'k': sum(times_segmentation), 
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
    pred_df.to_csv(os.path.join(output_dir, f'auc_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}_{intersection}.tsv'), sep='\t', index=False)


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
            except json.JSONDecodeError as e:
                print(f"JSON error: {e}. Retrying {file_path}")
           
            result = process_audio_file(data, method, intersection)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'auc_{method}_{name}_{intersection}.tsv'), sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results1',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    names = ["0.2-1", "0.2-3", "0.2-5",
            "0.3-1", "0.3-3", "0.3-5",
            "0.4-1","0.4-3","0.4-5", 
             "zeros", "noise", "stat", "all"]

    ############## audioset
    # for function in ['euclidean']:
    #     for mask_type in ['zeros', 'stat', 'noise']:
    #         for mask_percentage in [0.2, 0.3, 0.4]:
    #             for window_size in [1, 3, 5]:
    #                 for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
    #                     get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'audioset', 0.05)
    # for name in names:
    #     for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
    #         get_with_name(method, name, args.base_path, 'audioset', 0.05)

    ############## drums 
    for function in ['euclidean']:
        for mask_type in ['zeros', 'stat', 'noise']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
                        get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'drums', 0.05)
    for name in names:
        for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
            get_with_name(method, name, args.base_path, 'drums', 0.05)
   
    ############## kws
    for function in ['euclidean']:
        for mask_type in ['noise', 'stat', 'zeros']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
                        get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'kws',  0.05)

    # for name in names:
    #     for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
    #         get_with_name(method, name, args.base_path, 'kws', 0.05)


    ############## cough 
    # for function in ['euclidean']:
    #     for mask_type in ['zeros', 'stat', 'noise']:
    #         for mask_percentage in [0.2, 0.3, 0.4]:
    #             for window_size in [1, 3, 5]:
    #                 for method in ['tree_importance', 'linear_regression', 'kernel_shap', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
    #                     get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'cough', 0.05)
    # for name in names:
    #     for method in ['tree_importance', 'linear_regression', 'kernel_shap']:
    #         get_with_name(method, name, args.base_path, 'cough')




    # for function in ['euclidean']:
    #     for mask_type in ['zeros']:
    #         for mask_percentage in [0.1, 0.2, 0.3, 0.4]:
    #             for window_size in [1, 2, 3, 4, 5]:
    #                 for method in ['tree_importance', 'linear_regression', 'kernel_shap']:
    #                     get(method, mask_percentage, window_size, mask_type, function, args.base_path, 'audioset')

if __name__ == '__main__':
    main()

    