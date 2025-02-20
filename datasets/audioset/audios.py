"""Script to process every audio file for AST o Yamnet model"""

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from scipy.signal import resample
import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_csv('/home/ec2-user/Datasets/Audioset/labels/audioset_eval.csv')

feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model_ast = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model_ast = model_ast.to('cuda')


def predict_with_ast(filename, label):
    try:
        wav_data, sample_rate = sf.read(f'/home/ec2-user/Datasets/audioset_eval_wav/{filename}.wav')
    except:
        return -1000
    if sample_rate != 16000:
        wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))
    if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
        wav_data = wav_data.mean(axis=1)
    inputs = feature_extractor(wav_data, sampling_rate=16000, return_tensors="pt") 
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        logits = model_ast(**inputs).logits
        
    return logits.cpu().tolist()[0][label]


# Process predictions
df['score_ast'] = df.progress_apply(
    lambda row: predict_with_ast(row['base_segment_id'], row['father_id_ast']),
    axis=1
)

df['multiple_segments'] = df.apply(
    lambda row: any(not pd.isna(row[col]) for col in df.columns if col.startswith('segment_') and col != 'segment_1'),
    axis=1
)

df_to_eval = df[df['score_ast'] > 0]
df_to_eval = df_to_eval.groupby(['base_segment_id', 'score_ast'], as_index=False).agg({
    'duration': 'sum',  
    'father_id_ast': 'first',  # or another function like 'max' if needed
    'multiple_segments': 'first'  # Preserve 'multiple_segments' by taking the first value in each group
})
df_to_eval = df_to_eval[df_to_eval['duration'] < 5.0]


train_df, test_df = train_test_split(
    df_to_eval, 
    test_size=0.25,  # 25% test, 75% train
    stratify=df_to_eval['multiple_segments'],  # Stratify by the 'multiple_segments' column
    random_state=42  
)
train_df = train_df.reset_index()
test_df = test_df.reset_index()

test_df.to_csv('/home/ec2-user/Datasets/Audioset/labels/audioset_eval_test.csv', index=False)
train_df.to_csv('/home/ec2-user/Datasets/Audioset/labels/audioset_eval_train.csv', index=False)

df.to_csv('/home/ec2-user/Datasets/Audioset/labels/audioset_eval.csv')