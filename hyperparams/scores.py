
import json
from utils import process_audioset_csv, process_tsv, process_audio_segments, process_predictions_and_labels, predict_fn, calculate_confusion_matrix
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import soundfile as sf
from scipy.signal import resample
import os
from tqdm import tqdm

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cuda')

############# Preprocess to get the audio files we are going to run and the labels we have segmented for each one.

df = process_audioset_csv('/home/cbolanos/experiments/audioset/labels/unbalanced_train_segments.csv')
df_segmented = process_tsv('/home/cbolanos/experiments/audioset/labels/audioset_train_strong.tsv')
with open('/home/cbolanos/experiments/audioset/labels/ontology.json', 'r') as file:
     ontology_data = json.load(file)


labels_dict = dict(zip(df['YTID'], df['positive_labels']))

df_segmented['positive_labels'] = df_segmented['base_segment_id'].map(labels_dict)

df_segmented_notna = df_segmented[df_segmented['positive_labels'].notna()]

result_df = process_audio_segments(df_segmented_notna, ontology_data)

result_df.to_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')


############ Process each audio file to get the scores

# labels_to_process = result_df['base_segment_id'].unique()

# for filename in tqdm(labels_to_process):

#     path = f'/mnt/data/audioset_24k/unbalanced_train/{filename}.flac'
    
#     try:
#         wav_data, sample_rate = sf.read(path)
#         resampled_audio = resample(wav_data, int(len(wav_data) * 16000 / 24000))

#         if len(resampled_audio.shape) > 1 and resampled_audio.shape[1] == 2:
#             resampled_audio = resampled_audio.mean(axis=1)
#         else:
#             resampled_audio = resampled_audio

#         real_pred = predict_fn(resampled_audio)
#         output_data = {'real_scores': real_pred}

#         os.makedirs(f'/home/cbolanos/experiments/audioset/{filename}', exist_ok=True)

#         output_path = f'/home/cbolanos/experiments/audioset/{filename}/predictions.json'
#         with open(output_path, 'w') as f:
#             json.dump(output_data, f, indent=2)

#     except sf.LibsndfileError as e:
#         print(f"Error reading {filename}: {str(e)}")
#     except Exception as e:
#         print(f"Unexpected error reading {filename}: {str(e)}")
   
############# Process once we have the score to get the distribution

# results_dict = process_predictions_and_labels('/home/cbolanos/experiments/audioset/', result_df, model)

# output_path = f'/home/cbolanos/experiments/audioset/labels/distributions.json'
# with open(output_path, 'w') as f:
#     json.dump(results_dict, f, indent=2)

############# Process and get the metrics

metrics = calculate_confusion_matrix('/home/cbolanos/experiments/audioset/', result_df, model, threshold=-0.5)

output_path = f'/home/cbolanos/experiments/audioset/labels/metrics.json'
with open(output_path, 'w') as f:
    json.dump(metrics, f, indent=2)