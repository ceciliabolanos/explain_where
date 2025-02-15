from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import soundfile as sf
from scipy.signal import resample
import json
from tqdm import tqdm
from utils import process_importance_values
import os
import pandas as pd
import argparse
import re

def save_scores(filename, data, score_curves, method, mask_percentage, window_size):
    scores = {
        'filename': filename,
        'event_label': data['metadata']['label_explained'],
        'actual_score': data['metadata']['true_score'],
        **score_curves
    }
    label_explained = data['metadata']['label_explained']

    dir_path = os.path.join('/home/cbolanos/experiments/audioset_audios_eval', filename)
    os.makedirs(dir_path, exist_ok=True)
    
    with open(os.path.join(dir_path, f'scores_curves_{method}_{label_explained}_p{mask_percentage}_m{window_size}.json'), 'w') as f:
        json.dump(scores, f, indent=2)
    

def get_model_and_extractor(model_name: str):
    if model_name == 'yamnet': # cambiar esto de aca
        model = ASTForAudioClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    elif model_name == 'ast':
        feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model.to('cuda'), feature_extractor

def predict_fn(wav_array, feature_extractor, model):
    if not isinstance(wav_array, list):
        wav_array = [wav_array]
    
    inputs_list = [feature_extractor(audio, sampling_rate=16000, return_tensors="pt") 
                  for audio in wav_array]
    
    inputs = {
        k: torch.cat([inp[k] for inp in inputs_list]).to("cuda")
        for k in inputs_list[0].keys()
    }
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
    return logits.cpu().tolist()

def process_audio_file(file_path, data, model, feature_extractor, method, mask_percentage, window_size):
    filename = data["metadata"]['filename']
    label_explained = data['metadata']['label_explained']
    id_explained = data['metadata']['id_explained']
    
    score_path = os.path.join('/home/cbolanos/experiments/audioset_audios_eval', filename, f'scores_curves_{method}_{label_explained}_p{mask_percentage}_m{window_size}.json')
    
    if os.path.exists(score_path):
        with open(score_path, 'r') as f:
            return json.load(f)
            
    segment_length = data["metadata"]['segment_length']
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
    
    importance_values, times = process_importance_values(values, segment_size=segment_length, step_size=granularidad_ms)

    # Load and process audio
    wav_data, sample_rate = sf.read(file_path)
    if sample_rate != 16000:
        wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))
    if len(wav_data.shape) > 1:
        wav_data = wav_data.mean(axis=1)
    
    # Process importance scores
    importance_time_pairs = list(zip(importance_values, times))

    if method == 'naive':
        sorted_pairs_descending = sorted(importance_time_pairs, key=lambda x: x[0], reverse=True)
        sorted_pairs_ascending = sorted(importance_time_pairs, key=lambda x: x[0])
    else:
        sorted_pairs_descending = sorted(importance_time_pairs, key=lambda x: abs(x[0]), reverse=True)
        sorted_pairs_ascending = sorted(importance_time_pairs, key=lambda x: abs(x[0]))

    sorted_importances_d, sorted_times_d = zip(*sorted_pairs_descending)
    sorted_importances_a, sorted_times_a = zip(*sorted_pairs_ascending)

    # Calculate scores
    audio_descending_higher = wav_data.copy()
    audio_descending_lower = wav_data.copy()
    score_curves = {
        'score_curve_descending_higher': [], # Delete higher importance values
        'score_curve_descending_lower': [],
        'gt_curve_descending': [],
        'importance_curve_descending': []
    }
    
    # Process in batches
    batch_size = 100
    list_descending_higher = []
    list_descending_lower = []
    for i in range(0, len(sorted_importances_d), batch_size):
        batch_times_d = sorted_times_d[i:i+batch_size]
        batch_times_a = sorted_times_a[i:i+batch_size]
        
        # Descending modification with the higher importance values
        start_idx_d = int(batch_times_d[0] * 16000)
        end_idx_d = int(batch_times_d[-1] * 16000)
        audio_descending_higher[start_idx_d:end_idx_d] = 0

        # Descending modification with the lower importance values
        start_idx_a = int(batch_times_a[0] * 16000)
        end_idx_a = int(batch_times_a[-1] * 16000)
        audio_descending_lower[start_idx_a:end_idx_a] = 0
        
        list_descending_higher.append(audio_descending_higher.copy())
        list_descending_lower.append(audio_descending_lower.copy())

        # Calculate scores
        score_curves['importance_curve_descending'].append(sorted_importances_d[i])
        is_there = any(time_tuple[0] <= batch_times_d[0] and batch_times_d[-1] <= time_tuple[1] 
                      for time_tuple in times_gt)
        score_curves['gt_curve_descending'].append(int(is_there))

    results_descending = predict_fn(list_descending_higher, feature_extractor, model)
    results_ascending = predict_fn(list_descending_lower, feature_extractor, model)
    score_curves['score_curve_descending_higher'] = [results_descending[i][id_explained] for i in range(len(list_descending_higher))]
    score_curves['score_curve_descending_lower'] = [results_ascending[i][id_explained] for i in range(len(list_descending_lower))]

    save_scores(filename, data, score_curves, method, mask_percentage, window_size)

    return {
        'filename': filename,
        'event_label': data['metadata']['label_explained'],
        'actual_score': data['metadata']['true_score'],
        **score_curves
    }


def get(method: str, model_name: str, base_path: str, mask_percentage, window_size):
    model, feature_extractor = get_model_and_extractor(model_name)
    results = []
    
    for root, _, files in tqdm(os.walk(os.path.join(base_path, 'audioset_audios_eval'))):
        pattern = re.compile(rf'ft_.*_p{mask_percentage}_m{window_size}\.json$')
        json_files = [f for f in files if pattern.match(f)]

        for json_file in json_files:
            file_path = os.path.join(root, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            audio_path = os.path.join('/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/', 
                                    f"{data['metadata']['filename']}.wav")
            
            try:
                result = process_audio_file(audio_path, data, model, feature_extractor, method, mask_percentage, window_size)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    # Save results
    output_dir = os.path.join(base_path, 'audioset_evaluation/leo_metric/hyperparams')
    os.makedirs(output_dir, exist_ok=True)
    
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(os.path.join(output_dir, f'score_curve_{method}_p{mask_percentage}_m{window_size}.tsv'), sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Process AudioSet data and generate evaluation files.')
    parser.add_argument('--base_path', type=str,
                      default='/home/cbolanos/experiments',
                      help='Base path for AudioSet experiments')

    args = parser.parse_args()
   
    # for mask_percentage in [0.1, 0.15, 0.2, 0.3, 0.4]:
        # for window_size in [1, 2, 3, 4, 5, 6]:


    for mask_percentage, window_size in [(0.1, 4), (0.1, 5), (0.1, 6)]:
        for method in ['shap', 'tree_importance', 'naive', 'linear_regression']:
                get(method, "ast", args.base_path, mask_percentage, window_size)

    for method in ['linear_regression']:
        get(method, "ast", args.base_path, 0.15, 6)
if __name__ == '__main__':
    main()
