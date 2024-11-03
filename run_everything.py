from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import os 
import numpy as np
import soundfile as sf
from scipy.signal import resample
from random_forest import RandomForest
from lime import LimeAudioExplainer
from generate_data import DataGenerator
from naive import NaiveAudioAnalyzer
import json
from utils import create_visualization

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
filename = 'GfcQeSO8NKs'
markers = [(0.803,1.642),(5.821, 10)]

############## Load the model  ##############
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cuda')

############## Load the audio we are using to test ##############
path = f'/mnt/data/audioset_24k/unbalanced_train/{filename}.flac'

wav_data, sample_rate = sf.read(path)
resampled_audio = resample(wav_data, int(len(wav_data) * 16000 / 24000))

if len(resampled_audio.shape) > 1 and resampled_audio.shape[1] == 2:
    resampled_audio = resampled_audio.mean(axis=1)
else:
    resampled_audio = resampled_audio

############## Define the function predict_fn ##############

def predict_fn(wav_array):
    if not isinstance(wav_array, list):
        wav_array = [wav_array]
    
    inputs_list = [feature_extractor(audio, sampling_rate=16000, return_tensors="pt") for audio in wav_array]
    
    # Combine the processed features
    inputs = {
        k: torch.cat([inp[k] for inp in inputs_list]).to('cuda')
        for k in inputs_list[0].keys()
    }
    with torch.no_grad():
        logits = model(**inputs).logits
        
    return logits.cpu().tolist() 


############## Generate the data ##############

DataGenerator(resampled_audio, mode='naive_masked_zeros', segment_length=500, overlap=100, num_samples=8000, predict_fn=predict_fn, sr=16000).generate(filename)
DataGenerator(resampled_audio, mode='all_noise', segment_length=500, overlap=100, num_samples=8000, predict_fn=predict_fn, sr=16000).generate(filename)
DataGenerator(resampled_audio, mode='all_masked', segment_length=500, overlap=100, num_samples=8000, predict_fn=predict_fn, sr=16000).generate(filename)

real_pred = predict_fn(resampled_audio)
label_to_explain = np.argmax(real_pred)
print(f'We are explaining label {model.config.id2label[label_to_explain]}')


################## Run NAIVE ##################
print('Running Naive')

importances_naive = NaiveAudioAnalyzer(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_naive_masked_zeros.json',
                                        filename=filename).get_feature_importance(label_to_explain=label_to_explain)

################## Run RF ##################
print('Running random forest')
importances_rf_1 = RandomForest(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_masked.json', 
                                filename=filename).get_feature_importances(label_to_explain=label_to_explain)

importances_rf_2 = RandomForest(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_noise.json',
                                filename=filename).get_feature_importances(label_to_explain=label_to_explain)


################### Run Linear model ##################
print('Running Linear Model')

importances_lm_1 = LimeAudioExplainer(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_masked.json',
                                verbose=False, absolute_feature_sort=False).explain_instance(label_to_explain=label_to_explain).get_feature_importances(label=label_to_explain)


importances_lm_2 = LimeAudioExplainer(path=f'/home/cbolanos/experiments/audioset/{filename}/scores_data_all_noise.json',
                                verbose=False, absolute_feature_sort=False).explain_instance(label_to_explain=label_to_explain).get_feature_importances(label=label_to_explain)


output_data = {
    "metadata": {
        "filename": filename,  # Using filename from outer scope
        "label_explained": int(label_to_explain), # Using label_to_explain from outer scope
    },
    "importance_scores": {
        "naive": {
            "method": "NaiveAudioAnalyzer",
            "values": importances_naive.tolist() if hasattr(importances_naive, 'tolist') else importances_naive
        },
        "random_forest": {
            "masked": {
                "method": "RandomForest with masking",
                "values": importances_rf_1.tolist() if hasattr(importances_rf_1, 'tolist') else importances_rf_1
            },
            "noise": {
                "method": "RandomForest with noise",
                "values": importances_rf_2.tolist() if hasattr(importances_rf_2, 'tolist') else importances_rf_2
            }
        },
        "lime": {
            "masked": {
                "method": "LIME with masking",
                "values": importances_lm_1
            },
            "noise": {
                "method": "LIME with noise",
                "values": importances_lm_2
            }
        }
    }
}
output_path = f'/home/cbolanos/experiments/audioset/{filename}/feature_importances.json'
with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Importance scores saved to: {output_path}")

########################### Generate Video #############################
 
	
create_visualization(
    waveform=resampled_audio,
    json_file=output_path,
    output_file=f'/home/cbolanos/experiments/audioset/{filename}/video_{model.config.id2label[label_to_explain]}_{filename}.mp4',
    markers=markers
)


