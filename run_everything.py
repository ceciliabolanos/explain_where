import os
import json
import logging
import soundfile as sf
from scipy.signal import resample
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from utils import open_json
from all_methods import run_all_methods, generate_data

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('audio_analysis.log'),
            logging.StreamHandler()
        ]
    )

def load_model(model_name):
    """Load the AST model and feature extractor"""
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = ASTForAudioClassification.from_pretrained(model_name)
        model = model.to('cuda')
        return feature_extractor, model
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise

def process_folder(folder, folder_path, df, model, threshold, segment_length, overlap, num_samples):
    """Process a single folder of audio data"""
    # Get ground truth labels for this folder
    folder_data = df[df['base_segment_id'] == folder]

    if folder_data.empty:
        logging.warning(f"No labels found for folder: {folder}")
        return

    true_labels = folder_data['father_labels'].tolist()
    logging.info(f"Processing folder {folder} with labels: {true_labels}")

    # Get predictions
    predictions_path = os.path.join(folder_path, 'predictions_ast.json')
    predictions = open_json(predictions_path)
    
    # Skip the folder if: all predictions are below threshold or 
    #                     the ones that are above are also in the true labels or
    #                     we only have one labels to predict (to simple) or
    #                     if the label duration is longer than 20% of the audio duration (this is below)

    if all(pred <= threshold for pred in predictions) or len(true_labels) == 1:
        return

    # Check predictions above threshold
    above_threshold_labels = [label for label, idx in model.config.label2id.items() 
                            if predictions[idx] > threshold]
    if above_threshold_labels and all(label not in true_labels for label in above_threshold_labels):
        return
    
    wav_data, sample_rate = sf.read(f'/mnt/data/audioset_24k/unbalanced_train/{folder}.flac')
    resampled_audio = resample(wav_data, int(len(wav_data) * 16000 / 24000))
    if len(resampled_audio.shape) > 1 and resampled_audio.shape[1] == 2:
        resampled_audio = resampled_audio.mean(axis=1)
    duration_ms = (len(resampled_audio) / 16000) * 1000

    # Process each label
    data_generate = False
    for label, idx in model.config.label2id.items():
        pred_positive = predictions[idx] > threshold
        true_positive = label in true_labels

        if pred_positive and true_positive:
            mask = (folder_data['father_labels'] == label)
            if all(folder_data[mask]['label_duration'] < duration_ms*0.3) and (folder_data[mask]['label_duration'].sum() < duration_ms*0.6):
                if not data_generate:
                    generate_data(folder, 
                                model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                                segment_length=segment_length,
                                overlap=overlap,
                                num_samples=num_samples)
                    data_generate = True
                
                time_tuples = list(zip(
                    folder_data[mask]['start_time_seconds'],
                    folder_data[mask]['end_time_seconds']
                ))

                results = run_all_methods(
                    filename=folder,
                    id_to_explain=idx,
                    label_to_explain=label,
                    markers=time_tuples,
                    segment_length=segment_length,
                    overlap=overlap,
                    threshold=threshold,
                    num_samples=num_samples,
                    generate_video=False
                )
            else:
                logging.info(f'{label} too long')
        elif true_positive:
            logging.info(f'Could not predict label: {label}')

def main():
    # Configuration
    BASE_PATH = '/home/cbolanos/experiments/audioset/'
    LABELS_PATH = os.path.join(BASE_PATH, 'labels/labels_segments.csv')
    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    THRESHOLD = 0.0
    SEGMENT_LENGTH = 500
    OVERLAP = 400
    NUM_SAMPLES = 1000

    # Setup logging
    setup_logging()
    logging.info("Starting audio analysis script")

    try:
        # Load data and model
        df = pd.read_csv(LABELS_PATH)
        feature_extractor, model = load_model(MODEL_NAME)

        # Process each folder
        for folder in tqdm(os.listdir(BASE_PATH)):
            folder_path = os.path.join(BASE_PATH, folder)
            
            if folder == 'labels' or not os.path.isdir(folder_path):
                continue
                
            try:
                process_folder(
                    folder=folder,
                    folder_path=folder_path,
                    df=df,
                    model=model,
                    threshold=THRESHOLD,
                    segment_length=SEGMENT_LENGTH,
                    overlap=OVERLAP,
                    num_samples=NUM_SAMPLES
                )
            except Exception as e:
                logging.error(f"Error processing folder {folder}: {str(e)}")
                continue

    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()