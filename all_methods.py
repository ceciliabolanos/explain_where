from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import soundfile as sf
from scipy.signal import resample
from methods.random_forest import RandomForest
from methods.linear_regression import LimeAudioExplainer
from methods.generate_data import DataGenerator
from methods.naive import NaiveAudioAnalyzer
import json
from utils import create_visualization
import numpy as np


def generate_data(filename, 
                  model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                  segment_length=500,
                  overlap=250,
                  num_samples=4500):
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    model = model.to('cuda')

    audio_path = f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{filename}.wav'
    
    ############## Load the audio ##############

    wav_data, sample_rate = sf.read(audio_path)
    if sample_rate != 16000:
        wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))

    # Convert stereo to mono if necessary
    if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
        wav_data = wav_data.mean(axis=1)

    ############## Define prediction function ##############
    def predict_fn(wav_array):
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

    ############## Generate data ##############
    data_generator = DataGenerator(
        wav_data,
        segment_length=segment_length,
        overlap=overlap,
        num_samples=num_samples,
        predict_fn=predict_fn,
        sr=16000
    )
    
    # Generate masked data
    data_generator.mode = 'naive_masked_zeros'
    data_generator.generate(filename)
    
    data_generator.mode = 'all_masked'
    data_generator.generate(filename)


def run_all_methods(
    filename: str,
    id_to_explain,
    label_to_explain,
    markers,
    segment_length: int = 500,
    overlap: int = 250,
    true_score=0.0,
    num_samples: int = 4500, 
    generate_video = True
) -> dict:
    
    output_dir = '/home/cbolanos/experiments/audioset_audios_eval'

    # Naive analysis
    print('Running Naive analysis...')
    naive_analyzer = NaiveAudioAnalyzer(
        path=f'{output_dir}/{filename}/scores_data_naive_masked_zeros.json',
        filename=filename
    )
    importances_naive = naive_analyzer.get_feature_importance(label_to_explain=id_to_explain)

    # Random Forest analysis    
    print('Running Random Forest analysis...')
    rf_analyzer = RandomForest(
        path=f'{output_dir}/{filename}/scores_data_all_masked.json',
        filename=filename
    )
    importances_rf_tree = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='tree')
    importances_rf_shap = rf_analyzer.get_feature_importances(label_to_explain=id_to_explain, method='shap')

    # LIME analysis
    print('Running LIME analysis...')
    lime_analyzer = LimeAudioExplainer(
        path=f'{output_dir}/{filename}/scores_data_all_masked.json',
        verbose=False,
        absolute_feature_sort=False
    )
    importances_lime = lime_analyzer.explain_instance(
        label_to_explain=id_to_explain
    ).get_feature_importances(label=id_to_explain)

    ############## Prepare output ##############
    output_data = {
        "metadata": {
            "filename": filename,
            "id_explained": id_to_explain,
            "label_explained": label_to_explain,
            "segment_length": segment_length,
            "overlap": overlap,
            "num_samples": num_samples,
            "true_markers": markers,
            "true_score": true_score
        },
        "importance_scores": {
            "naive": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_naive.tolist() if hasattr(importances_naive, 'tolist') else importances_naive
            },
            "random_forest": {
                "tree_importance": {
                    "method": "masked rf with tree importance",
                    "values": importances_rf_tree.tolist() if hasattr(importances_rf_tree, 'tolist') else importances_rf_tree
                },
                "shap": {
                    "method": "masked rf with shap importance",
                    "values": importances_rf_shap.tolist() if hasattr(importances_rf_shap, 'tolist') else importances_rf_shap
                }
            },
            "linear_regression": {
                "masked": {
                    "method": "Linear Regression with masking",
                    "values": importances_lime
                }
            }
        }
    }

    # Save results
    output_path = f'{output_dir}/{filename}/ft_{label_to_explain}.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Importance scores saved to: {output_path}")
    
    if generate_video:
        audio_path = f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{filename}.wav'
        wav_data, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))

        # Convert stereo to mono if necessary
        if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
            wav_data = wav_data.mean(axis=1)
        
        predictions_path = f'{output_dir}/{filename}/predictions_yamnet.json'
        with open(predictions_path, 'r') as f:
            scores_yamnet = np.array(json.load(f)['real_scores'])
            
        with open('/home/cbolanos/experiments/audioset/labels/labels_yamnet.json', 'r') as f:
            class_names = json.load(f)['label']
        index = class_names.index(label_to_explain)

        create_visualization(
        waveform=wav_data,
        scores_yamnet=scores_yamnet,
        class_index=index,
        json_file=output_path,
        output_file=f'{output_dir}/{filename}/video_{label_to_explain}_{filename}.mp4',
        markers=markers
    )
    return output_data

