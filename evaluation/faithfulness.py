import json
from tqdm import tqdm
import os
import pandas as pd
import argparse
import re
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ast.model import ASTModel
from models.drums.model import DrumsModel
from models.kws.model import KWSModel

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
    id_explained = int(data['metadata']['id_explained'])
    segment_length = data["metadata"]['segment_length']
    overlap = 0 

    if dataset == 'audioset':
       model = ASTModel(filename, id_explained) 
    if dataset == 'drums':
       complete_filename = f'mnt/data/drum_dataset/{filename}'
       model = DrumsModel(complete_filename, id_explained) 
       filename = os.path.basename(filename)
    if dataset == 'kws':
       session = filename.split('-')[0]
       folder = filename.split('-')[1]
       complete_filename = f'mnt/data/LibriSpeech24K/test-clean/{session}/{folder}/{filename}'
       model = KWSModel(complete_filename, id_explained) 
    
    wav_data, real_score_id = model.process_input()     
    predict_fn = model.get_predict_fn()   

    # Get importance scores
    if method == 'RF':
        values = data['importance_scores']['RF']['values']
    elif method == 'SHAP':
        values = data['importance_scores']['SHAP']['values']
    elif method == 'LR':
        values = data['importance_scores']['LR']['values']

    # Calculate scores
    audio_baseline = get_audio_base(wav_data, masking_type, segment_length)
    
    audio_sacando_topk = wav_data.copy()
    audio_solo_topk = audio_baseline.copy()
    
    score_curves = {
        'score_curve_sacando_topk': [], # Delete higher importance values
    }
    
    list_audio_sacando_topk = []
    list_solo_topk = []
    sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    L = int((segment_length / 1000) * 16000)   
    O = int((overlap / 1000) * 16000)
    
    step = L - O
    for i in sorted_indices:
        start_idx_d = i * step
        end_idx_d = start_idx_d + step
        audio_sacando_topk[start_idx_d:end_idx_d] = audio_baseline[start_idx_d:end_idx_d]

        list_audio_sacando_topk.append(audio_sacando_topk.copy())
        list_solo_topk.append(audio_solo_topk.copy())
        
        # Process in larger chunks (50 at a time)
        if len(list_audio_sacando_topk) >= 32 or i + 1 >= len(sorted_indices):
            results_descending = predict_fn(list_audio_sacando_topk)
            score_curves['score_curve_sacando_topk'].extend(results_descending)
            list_audio_sacando_topk.clear()

    return {
        'filename': filename,
        'event_label': data['metadata']['id_explained'],
        'actual_score': real_score_id,
        **score_curves
    }


def get(method, mask_percentage, window_size, mask_type, function, base_path, dataset):
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, f'explanations_{dataset}'))):
        pattern = re.compile(rf'ft_.*_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}\.json$')
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


def get_audioset(method: str, mask_percentage, window_size, mask_type, function, base_path: str, dataset):
    results = []
    selected_files = pd.read_csv('/home/ec2-user/explain_where/datasets/audioset/audioset.csv')
    for i in tqdm(range(len(selected_files))):
        id = int(selected_files.loc[i]['event_label'])
        filename = selected_files.loc[i]['filename']
        file_path = f'{base_path}/explanations_{dataset}/{filename}/ast/ft_{id}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.json'
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
    pred_df.to_csv(os.path.join(output_dir,f'score_curve_{method}_p{mask_percentage}_w{window_size}_f{function}_m{mask_type}.tsv'), sep='\t', index=False)


def get_with_name_audioset(method: str, name, base_path: str, dataset, mask_type, id):
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
                    result = process_audio_file(data, dataset, method, mask_type)
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
    pred_df.to_csv(os.path.join(output_dir, f'score_curve_{method}_{name}.tsv'), sep='\t', index=False)


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
                      default='/home/ec2-user/results',
                      help='Base path for AudioSet experiments')
    args = parser.parse_args()


    for dataset in ['audioset']:
        for id in [0, 74, 137]:
            for name in ['noise', 'zeros']:
                for method in ['RF', 'LR', 'SHAP']:
                    get_with_name_audioset(method, name, args.base_path, dataset, name, id)
    # , 'cough', 'kws'
    # datasets = ['kws']
    # for dataset in datasets:
    #     for name in ['noise', 'zeros']:
    #         for method in ['RF', 'LR', 'SHAP']:
    #             get_with_name(method, name, args.base_path, dataset, name)
                 

   
if __name__ == '__main__':
    main()
