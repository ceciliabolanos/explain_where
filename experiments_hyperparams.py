import os
import logging
import soundfile as sf
from scipy.signal import resample
import pandas as pd
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from utils import open_json
from all_methods import run_all_methods, generate_data
import shutil
import re

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
        model = model.to('cpu')
        return feature_extractor, model
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {str(e)}")
        raise

def process_folder(folder, 
                   folder_path, 
                   df, 
                   model, 
                   segment_length, 
                   mask_percentage, 
                   window_size, 
                   num_samples):
    
    """Process a single folder of audio data"""
    # Get ground truth labels for this folder
    folder_data = df[df['base_segment_id'] == folder]

    data_generate = False

    target_filename = f"scores_data_all_masked_p{mask_percentage}_m{window_size}.json"
    full_path = os.path.join(folder_path, target_filename)
    
    # Check if the exact file exists
    if os.path.isfile(full_path):
         data_generate = True

    target_filename = rf'ft_.*_p{mask_percentage}_m{window_size}\.json$'

    # Check if any file matches the pattern
    
    # for file in os.listdir(folder_path):
    #     if re.match(target_filename, file):
    #         return  # Exit function if any matching file is found
        
    true_ids = folder_data['father_labels_ids'].tolist()

    # Get predictions 
    predictions_path = os.path.join(folder_path, 'predictions_ast.json')
    predictions = open_json(predictions_path)
    
    ids_intersection = list(set(predictions['positive_indices']) & set(true_ids))
    
    if len(ids_intersection) == 0:
        shutil.rmtree(folder_path)
        return
    
    wav_data, sample_rate = sf.read(f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{folder}.wav')
    if sample_rate != 16000:
        wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))

    if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
        wav_data = wav_data.mean(axis=1)

    duration_ms = (len(wav_data) / 16000) * 1000
    
    all_labels_meet_criteria = False 

    # Check each label
    for id in ids_intersection:
        label = model.config.id2label[id]
        mask = (folder_data['father_labels'] == label)
        
        if (all(folder_data[mask]['label_duration'] < duration_ms*0.3) and 
                (folder_data[mask]['label_duration'].sum() < duration_ms*0.4)):
            all_labels_meet_criteria = True
            break

    if not all_labels_meet_criteria:
        shutil.rmtree(folder_path)
        return   
   
    # Process each label
    for id in ids_intersection:
        label = model.config.id2label[id]
        mask = (folder_data['father_labels'] == label)
        if all(folder_data[mask]['label_duration'] < duration_ms*0.3) and (folder_data[mask]['label_duration'].sum() < duration_ms*0.4):
            if not data_generate:
                generate_data(folder, 
                            model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                            segment_length=segment_length,
                            mask_percentage=mask_percentage,
                            window_size=window_size,
                            num_samples=num_samples)
                data_generate = True
            
            time_tuples = list(zip(
                folder_data[mask]['start_time_seconds'],
                folder_data[mask]['end_time_seconds']
            ))

            results = run_all_methods(
                filename=folder,
                id_to_explain=id,
                label_to_explain=label,
                markers=time_tuples,
                segment_length=segment_length,
                mask_percentage=mask_percentage,
                window_size=window_size,
                true_score=predictions['real_scores'][0][id],
                num_samples=num_samples,
                generate_video=False
            )
        else:
            logging.info(f'{label} too long')
    return
          
def main():
    # Configuration
    BASE_PATH = '/home/cbolanos/experiments/audioset_audios_eval/'
    LABELS_PATH = '/home/cbolanos/experiments/audioset/labels/labels_segments.csv'
    MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    SEGMENT_LENGTH = 100
    NUM_SAMPLES = 4500
    
    # Masking parameters to try
    mask_percentages = [0.1, 0.15, 0.2, 0.3, 0.4]
    window_sizes = [1, 2, 3, 4, 5, 6]
    
    # Setup logging
    setup_logging()
    logging.info("Starting audio analysis script")

    try:
        # Load data and model
        df = pd.read_csv(LABELS_PATH)
        feature_extractor, model = load_model(MODEL_NAME)

        # Get first 70 folders in alphabetical order
        all_folders = sorted(os.listdir(BASE_PATH))[0:10000]
        
        # Iterate through masking parameters
        for mask_percentage in mask_percentages:
            for window_size in window_sizes:                
                # Process each folder
                for folder in tqdm(all_folders):
                    folder_path = os.path.join(BASE_PATH, folder)
                    target_filename = f"scores_data_all_masked_p{mask_percentage}_m{window_size}.json"
                    full_path = os.path.join(folder_path, target_filename)
                    
                    # Check if the exact file exists
                    if os.path.isfile(full_path):
                        try:
                            process_folder(
                                folder=folder,
                                folder_path=folder_path,
                                df=df,
                                model=model,
                                segment_length=SEGMENT_LENGTH,
                                mask_percentage=mask_percentage,
                                window_size=window_size,
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