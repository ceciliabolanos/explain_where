import os
import json
import argparse
from tqdm import tqdm
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from scipy.signal import resample
import tensorflow as tf
import tensorflow_hub as hub

def parse_args():
    parser = argparse.ArgumentParser(description='Process audio files using a model')
    parser.add_argument('--labels_path', type=str, required=False, default='/home/cbolanos/experiments/audioset/labels/labels_segments.csv')
    parser.add_argument('--audio_dir', type=str, required=False, default='/mnt/data/audioset_24k/unbalanced_train')
    parser.add_argument('--output_dir', type=str, required=False, default='/home/cbolanos/experiments/audioset')
    parser.add_argument('--model_name', type=str, default="yamnet")
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--output_filename', type=str, 
                      default='predictions_yamnet.json',
                      help='Name of the output JSON file')

    return parser.parse_args()

def setup_model(model_name, device):
    if model_name == "ast":
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = ASTForAudioClassification.from_pretrained(model_name)
        model = model.to(device)
        audio_file_name = 'predictions_ast.json'
        
    elif model_name == "yamnet": 
        audio_file_name = 'predictions_yamnet.json'
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        feature_extractor = None
    return feature_extractor, model, audio_file_name

def predict_fn(wav_array, feature_extractor, model, device, model_name):
    if model_name == 'ast':
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
    
    elif model_name == 'yamnet': 
        waveform = wav_array / tf.int16.max
        scores, embeddings, spectrogram = model(waveform)
        return scores.numpy()


def process_audio_file(filename, args, feature_extractor, model,output_filename):
    path = os.path.join(args.audio_dir, f'{filename}.flac')
    
    try:
        wav_data, sample_rate = sf.read(path)
        resampled_audio = resample(
            wav_data, 
            int(len(wav_data) * 16000 / 24000)
        )

        if len(resampled_audio.shape) > 1 and resampled_audio.shape[1] == 2:
            resampled_audio = resampled_audio.mean(axis=1)

        real_pred = predict_fn(resampled_audio, feature_extractor, model, args.device, args.model_name)
        output_data = {'real_scores': real_pred}

        output_dir = os.path.join(args.output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

    except sf.LibsndfileError as e:
        print(f"Error reading {filename}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error reading {filename}: {str(e)}")

def main():
    args = parse_args()
    
    # Load labels
    result_df = pd.read_csv(args.labels_path)
    
    # Setup model
    feature_extractor, model, audio_file_name = setup_model(args.model_name, args.device)
    
    # Process files
    labels_to_process = result_df['base_segment_id'].unique()
    
    for filename in tqdm(labels_to_process):
        process_audio_file(filename, args, feature_extractor, model, audio_file_name)

if __name__ == "__main__":
    main()