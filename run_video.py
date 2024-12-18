import os, ast, json
import pandas as pd
from utils import open_json
from transformers import ASTForAudioClassification
from all_methods import run_all_methods, generate_data
from scipy.signal import resample
import soundfile as sf

FOLDER = '-aOxR6ILsw8'
LABEL = 'Engine starting'
MASK_PERCENTAGE = 0.3
WINDOW_SIZE = 1

BASE_PATH = '/home/cbolanos/experiments/audioset_audios_eval/'
df = pd.read_csv('/home/cbolanos/experiments/audioset/labels/labels_segments.csv')

model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = model.to('cpu')

# Get ground truth labels for this folder
folder_data = df[df['base_segment_id'] == FOLDER]

# Get predictions
predictions_path = os.path.join(os.path.join(BASE_PATH, FOLDER), 'predictions_ast.json')
predictions = open_json(predictions_path)


wav_data, sample_rate = sf.read(f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{FOLDER}.wav')
if sample_rate != 16000:
    wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))
if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
    wav_data = wav_data.mean(axis=1)
duration_ms = (len(wav_data) / 16000) * 1000

output_path = '/home/cbolanos/experiments/audioset/labels/true_medians.json'
with open(output_path, 'f') as f:
    median_score = json.load(f)

# Get the ids of the labels, if father_labels is not present, we try with a child label
true_ids = folder_data['father_labels_ids'].tolist()
child_ids = []
for id_list_str in folder_data['other_labels_ids']:
    id_list = ast.literal_eval(id_list_str)  # Convert string to list
    child_ids.extend([x for x in id_list if x != -1])  # Optional: filter out -1s

ids_intersection = []
for i in range(len(true_ids)):
    id = true_ids[i]
    label = model.config.id2label[id]
    mask = (folder_data['father_labels'] == label)

    if predictions['real_scores'][0][id] > median_score[id]:
        # If any label fails the criteria, set flag to False and break
        if (all(folder_data[mask]['label_duration'] < duration_ms*0.3) and 
                (folder_data[mask]['label_duration'].sum() < duration_ms*0.4)):
            all_labels_meet_criteria = True
            ids_intersection[id] = id
            break
    else:
        for j in range(len(child_ids)):
            if predictions['real_scores'][0][child_ids[j]] > median_score[child_ids[j]]:
                # If any label fails the criteria, set flag to False and break
                if (all(folder_data[mask]['label_duration'] < duration_ms*0.3) and 
                        (folder_data[mask]['label_duration'].sum() < duration_ms*0.4)):
                    all_labels_meet_criteria = True
                    ids_intersection[id] = child_ids[j]
                    print(f"Using child label {child_ids[j]} instead of father label {id}")
                    break

# Process each label
for id in ids_intersection.keys():
    label = model.config.id2label[id]
    if label == LABEL:
        print(label)
        mask = (folder_data['father_labels'] == label)
        time_tuples = list(zip(
                folder_data[mask]['start_time_seconds'],
                folder_data[mask]['end_time_seconds']
        ))

        results = run_all_methods(
            filename=FOLDER,
            id_to_explain=id,
            label_to_explain=label,
            markers=time_tuples,
            segment_length=100,
            mask_percentage=MASK_PERCENTAGE,
            window_size=WINDOW_SIZE,
            true_score=predictions['real_scores'][0][ids_intersection[id]],
            num_samples=4500,
            generate_video=True
        )
        

                
