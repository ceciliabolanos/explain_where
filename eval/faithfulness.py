import json
from tqdm import tqdm
from utils import process_importance_values
import os
import pandas as pd
import argparse
import re
import numpy as np
from models.ast import ASTModel
from models.drums import DrumsModel
from models.kws import KWSModel

def get_audio_base(wav_data, masking_type, segment_length, sr=16000, overlap=0):
    if masking_type not in {"zeros", "noise", "stat", "all"}:
        raise ValueError("Invalid masking_type. Choose from 'zeros', 'noise', 'stat', or 'all'.")
    
    overlap_samples = int(sr * (overlap / 1000))
    segment_length = int(sr * (segment_length / 1000))
    step_samples = segment_length - overlap_samples
    
    masked_audio = np.copy(wav_data)
    
    current_pos = 0
    while current_pos < len(wav_data):
        end = min(current_pos + segment_length, len(wav_data))
        
        if masking_type == "zeros":
            masked_audio[current_pos:end] = 0
        
        elif masking_type == "noise":
            noise_std = np.random.uniform(0.01, 0.1)
            masked_audio[current_pos:end] = np.random.normal(np.mean(wav_data), noise_std, end - current_pos)
        
        elif masking_type == "stat":
            fill_value = np.mean(wav_data[current_pos:end])
            masked_audio[current_pos:end] = fill_value
        
        elif masking_type == "all":
            random_mask_type = np.random.choice(["zeros", "noise", "stat"])
            if random_mask_type == "zeros":
                masked_audio[current_pos:end] = 0
            elif random_mask_type == "noise":
                noise_std = np.random.uniform(0.01, 0.1)
                masked_audio[current_pos:end] = np.random.normal(np.mean(wav_data), noise_std, end - current_pos)
            elif random_mask_type == "stat":
                fill_value = np.mean(wav_data[current_pos:end])
                masked_audio[current_pos:end] = fill_value
        
        current_pos += step_samples
    
    return masked_audio

def process_audio_file(data, dataset, method, masking_type):
    filename = data["metadata"]['filename']
    id_explained = data['metadata']['id_explained']
    segment_length = data["metadata"]['segment_length']
    overlap = 0 
    granularidad_ms = segment_length - overlap

    if dataset == 'audioset':
       model = ASTModel(filename, id_explained) 
    if dataset == 'drums':
       model = DrumsModel(filename, id_explained) 
       filename = os.path.basename(filename)
    if dataset == 'kws':
       model = KWSModel(filename, id_explained) 
       filename = os.path.basename(filename)
    
    wav_data, real_score_id = model.process_input()     
    predict_fn = model.get_predict_fn()   

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
    
    importance_values, times = process_importance_values(values, segment_size=segment_length, step_size=granularidad_ms)

    # Process importance scores
    importance_time_pairs = list(zip(importance_values, times))

    sorted_pairs_descending = sorted(importance_time_pairs, key=lambda x: x[0], reverse=True)

    sorted_importances_d, sorted_times_d = zip(*sorted_pairs_descending)

    # Calculate scores
    audio_baseline = get_audio_base(wav_data, masking_type, segment_length)
    
    audio_sacando_topk = wav_data.copy()
    audio_solo_topk = audio_baseline.copy()
    
    score_curves = {
        'score_curve_sacando_topk': [], # Delete higher importance values
        'score_curve_consolo_topk': [], # Only have higher importance values
    }
    
    # Process in batches
    batch_size = 100
    list_audio_sacando_topk = []
    list_solo_topk = []

    for i in range(0, len(sorted_importances_d), batch_size):
        batch_times_d = sorted_times_d[i:i+batch_size]
        
        # Descending modification with the higher importance values
        start_idx_d = int(batch_times_d[0] * 16000)
        end_idx_d = int(batch_times_d[-1] * 16000)
        audio_sacando_topk[start_idx_d:end_idx_d] = audio_baseline[start_idx_d:end_idx_d]
        audio_solo_topk[start_idx_d:end_idx_d] = wav_data[start_idx_d:end_idx_d]

        list_audio_sacando_topk.append(audio_sacando_topk.copy())
        list_solo_topk.append(audio_solo_topk.copy())


    results_descending = predict_fn(list_audio_sacando_topk)
    results_ascending = predict_fn(list_solo_topk)
    score_curves['score_curve_sacando_topk'] = [results_descending[i] for i in range(len(list_audio_sacando_topk))]
    score_curves['score_curve_consolo_topk'] = [results_ascending[i] for i in range(len(list_solo_topk))]

    return {
        'filename': filename,
        'event_label': data['metadata']['id_explained'],
        'actual_score': data['metadata']['true_score'],
        **score_curves
    }


def get(method, mask_percentage, window_size, mask_type, function, base_path, dataset):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_p{mask_percentage}_m{window_size}_f{function}_m{mask_type}\.json$')
        json_files = [f for f in files if pattern.match(f)]

        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            try:
                result = process_audio_file(data, dataset, method, mask_type)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'score_curve_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.tsv'), sep='\t', index=False)


def get_with_name(method: str, name, base_path: str, dataset, mask_type):
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
           
            result = process_audio_file(data, dataset, method, mask_type)
            results.append(result)
        
    output_dir = os.path.join(f'/home/ec2-user/evaluations/{dataset}/')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'score_curve_{method}_{name}.tsv'), sep='\t', index=False)



def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/ec2-user/results1',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()

    names = ["zeros", "noise", "stat", "all"]

    # Select dataset to run

    dataset = 'kws'
    for function in ['euclidean']:
        for mask_type in ['zeros', 'stat', 'noise']:
            for mask_percentage in [0.2, 0.3, 0.4]:
                for window_size in [1, 3, 5]:
                    for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
                        get(method, mask_percentage, window_size, mask_type, function, args.base_path, dataset)
    
    for name in names:
        for method in ['tree_importance', 'linear_regression_noreg_noweights', 'kernel_shap_sumcons']:
            get_with_name(method, name, args.base_path, dataset, name)


if __name__ == '__main__':
    main()
