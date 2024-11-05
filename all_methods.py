from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import numpy as np
import soundfile as sf
from scipy.signal import resample
from methods.random_forest import RandomForest
from methods.lime import LimeAudioExplainer
from methods.generate_data import DataGenerator
from methods.naive import NaiveAudioAnalyzer
import json
from utils import create_visualization


def generate_data(filename, 
                  model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                  segment_length=500,
                  overlap=400,
                  num_samples=100):
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name)
    model = model.to('cuda')

    audio_path = f'/mnt/data/audioset_24k/unbalanced_train/{filename}.flac'
    
    ############## Load the audio ##############

    wav_data, sample_rate = sf.read(audio_path)
    resampled_audio = resample(wav_data, int(len(wav_data) * 16000 / 24000))

    # Convert stereo to mono if necessary
    if len(resampled_audio.shape) > 1 and resampled_audio.shape[1] == 2:
        resampled_audio = resampled_audio.mean(axis=1)

    ############## Define prediction function ##############
    def predict_fn(wav_array):
        if not isinstance(wav_array, list):
            wav_array = [wav_array]
        
        inputs_list = [feature_extractor(audio, sampling_rate=16000, return_tensors="pt") 
                      for audio in wav_array]
        
        inputs = {
            k: torch.cat([inp[k] for inp in inputs_list]).to(device)
            for k in inputs_list[0].keys()
        }
        
        with torch.no_grad():
            logits = model(**inputs).logits
            
        return logits.cpu().tolist()

    ############## Generate data ##############
    data_generator = DataGenerator(
        resampled_audio,
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
    segment_length: int = 500,
    overlap: int = 400,
    num_samples: int = 100
) -> dict:
    """
    Analyze the feature importance of an audio file using multiple methods.
    
    Parameters:
    -----------
    filename : str
        Base filename without extension
    segment_length : int, optional
        Length of audio segments for analysis (default: 500)
    overlap : int, optional
        Overlap between segments (default: 400)
    num_samples : int, optional
        Number of samples to generate (default: 100)
    model_name : str, optional
        Name of the pretrained model to use
    device : str, optional
        Device to run the model on ('cuda' or 'cpu')
    
    Returns:
    --------
    dict
        Dictionary containing all importance scores and metadata
    """

    output_dir = '/home/cbolanos/experiments/audioset/'

    # Naive analysis
    print('Running Naive analysis...')
    naive_analyzer = NaiveAudioAnalyzer(
        path=f'{output_dir}/{filename}/scores_data_naive_masked_zeros.json',
        filename=filename
    )
    importances_naive = naive_analyzer.get_feature_importance(label_to_explain=label_to_explain)

    # Random Forest analysis
    print('Running Random Forest analysis...')
    rf_analyzer = RandomForest(
        path=f'{output_dir}/{filename}/scores_data_all_masked.json',
        filename=filename
    )
    importances_rf = rf_analyzer.get_feature_importances(label_to_explain=label_to_explain)

    # LIME analysis
    print('Running LIME analysis...')
    lime_analyzer = LimeAudioExplainer(
        path=f'{output_dir}/{filename}/scores_data_all_masked.json',
        verbose=False,
        absolute_feature_sort=False
    )
    importances_lime = lime_analyzer.explain_instance(
        label_to_explain=label_to_explain
    ).get_feature_importances(label=label_to_explain)

    ############## Prepare output ##############
    output_data = {
        "metadata": {
            "filename": filename,
            "id_explained": id_to_explain,
            "label_explained": label_to_explain,
            "segment_length": segment_length,
            "overlap": overlap,
            "num_samples": num_samples
        },
        "importance_scores": {
            "naive": {
                "method": "NaiveAudioAnalyzer",
                "values": importances_naive.tolist() if hasattr(importances_naive, 'tolist') else importances_naive
            },
            "random_forest": {
                "masked": {
                    "method": "RandomForest with masking",
                    "values": importances_rf.tolist() if hasattr(importances_rf, 'tolist') else importances_rf
                }
            },
            "lime": {
                "masked": {
                    "method": "LIME with masking",
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
    return output_data


########################### Generate Video #############################
 
	
# create_visualization(
#     waveform=resampled_audio,
#     json_file=output_path,
#     output_file=f'/home/cbolanos/experiments/audioset/{filename}/video_{model.config.id2label[label_to_explain]}_{filename}.mp4',
#     markers=markers
# )



# For noise: 

# DataGenerator(resampled_audio, mode='all_noise', segment_length=segment_length, overlap=overlap, num_samples=num_samples, predict_fn=predict_fn, sr=16000).generate(filename)
# importances_rf_2 = RandomForest(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_noise.json',
#                                 filename=filename).get_feature_importances(label_to_explain=label_to_explain)
# importances_lm_2 = LimeAudioExplainer(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_noise.json',
#                                 verbose=False, absolute_feature_sort=False).explain_instance(label_to_explain=label_to_explain).get_feature_importances(label=label_to_explain)



# Video 1
# filename = 'eBbJ6jsZGyI'
# markers = [(0.335, 2.246)]

# Video 2
# filename = 'LvNUyQ3xuAQ'
# markers = [(0.596, 5.677)]

# Video 3
# filename = '7s1hTxxxKW8'
# markers = [(6.495, 9.681)]

# Video 4
# filename = '3Wu1GXUb3MM'
# markers = [(0.585, 0.656), (0.778, 2.876), (3.261, 6.593), (6.8, 7.756), (7.985, 8.249)]

#####
# Video 5
# filename = '0r4zxu38gt0'
# markers = [(7.156, 7.75)]

# Video 6
# filename = '-FI_F5Lclrg'
# markers = [(1.179, 7.371), (9.034, 10)]

# Video 7
# filename = '1G5IEKEzQwE'
# markers = [(6.204, 10)]

# Video 8
# filename = '5sjd9zwDDFg'
# markers = [(0, 8.359)]

# Video 9
# filename = 'GfcQeSO8NKs'
# markers = [(0.803,1.642),(5.821, 10)]