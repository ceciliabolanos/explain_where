import os
import tempfile
import subprocess
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, writers
import scipy.io.wavfile as wav
import json
import pandas as pd
import librosa
from confidence_intervals import evaluate_with_conf_int
from evaluation.auc_relaxed import create_segmentation_vector, generate_sequence
import re, ast
from sklearn.metrics import roc_curve, auc

def calculate_std(dataset):
    stds = []
    if dataset == 'drums':
        audio_paths = '/home/ec2-user/mnt/data/drum_dataset'
    if dataset == 'audioset':
        audio_paths = '/home/ec2-user/Datasets/audioset_eval_wav'
    if dataset == 'kws':
        audio_paths = '/home/ec2-user/mnt/data/LibriSpeech24K/test-clean'
    if dataset == 'cough':
        audio_paths = '/home/ec2-user/mnt/data/IEMOCAP-happy-cough/Session5/sentences/wav'
    
    for root, _, files in os.walk(audio_paths):
        for file_name in files:
            if file_name.endswith((".wav", ".flac")):  # Process both WAV and FLAC files
                file_path = os.path.join(root, file_name)
                
                # Load audio and compute standard deviation
                audio, sr = librosa.load(file_path, sr=16000)
                std = np.std(audio)
                stds.append(std)
    return np.mean(stds)

def get_segments(base_segment_id, label, model):
    segment_lists = []
    if model == 'ast':
        df = pd.read_csv("/home/ec2-user/Datasets/Audioset/labels/audioset_eval.csv")  # Adjust path if necessary
        filtered_df = df[(df['base_segment_id'] == base_segment_id) & (df['father_id_ast'] == label)]
        if filtered_df.empty:
            return []  
        segment_columns = [col for col in df.columns if col.startswith("segment_")]
        for col in segment_columns:
            value = filtered_df[col].values[0]  
            if isinstance(value, str) and value.startswith("["):  
                segment_lists.append(ast.literal_eval(value))  
        
    if model == 'cough':
        df = pd.read_csv("/home/ec2-user/explain_where/datasets/cough/cough_happy.csv")
        filtered_df = df[df['filename'] == base_segment_id]
        filtered_df = filtered_df.reset_index(drop=True)
        segment_lists = [[filtered_df['cough_start'][0]/16000, filtered_df['cough_end'][0]/16000]]
    
    if model == 'drums':
        df = pd.read_csv("/home/ec2-user/explain_where/datasets/drums/drums_dataset.csv")
        filtered_df = df[df['filename'] == base_segment_id]
        filtered_df = filtered_df.reset_index(drop=True)
        pattern = filtered_df['pattern'].values[0]  # Get the string directly
        durations = ast.literal_eval(filtered_df['durations'].values[0])  # Assuming durations are a valid Python literal
        actual_samples = 0
        for p, d in zip(pattern, durations):
            if p == 'K':
                segment_lists.append([actual_samples / 16000, (actual_samples + d) / 16000])
            actual_samples += d

    if model == 'kws':
        df = pd.read_csv("/home/ec2-user/explain_where/datasets/kws/kws_dataset.csv")
    elif model == 'kws_fp':
        df = pd.read_csv("/home/ec2-user/explain_where/datasets/kws/kws_fp_dataset.csv")
    elif model == 'kws_fn':
        df = pd.read_csv("/home/ec2-user/explain_where/datasets/kws/kws_fn_dataset.csv")

    if model in ['kws', 'kws_fp', 'kws_fn']:
        filtered_df = df[df['filename'] == base_segment_id]
        filtered_df = filtered_df.reset_index(drop=True)
        segment_lists = [[filtered_df['word_start'].loc[0], filtered_df['word_end'].loc[0]]]
    
    return segment_lists
    

def process_importance_values(values, segment_size=100, step_size=100):
    # Calculate total duration
    num_segments = len(values)
    total_duration = (num_segments - 1) * step_size + segment_size
    timeline = np.arange(0, total_duration) / 1000
    
    # Create matrix for accumulating contributions
    accumulated_importance = np.zeros(total_duration)
    
    # For each segment, distribute its importance across its duration using Bartlett window
    for i, importance in enumerate(values):
        start_idx = i * step_size
        end_idx = start_idx + segment_size
        accumulated_importance[start_idx:end_idx] += importance 

    # Calculate overlap count for normalization
    overlap_count = np.zeros(total_duration)
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + segment_size
        overlap_count[start_idx:end_idx] += 1
    
    # Avoid division by zero and normalize
    overlap_count = np.maximum(overlap_count, 1)
    processed_importance = accumulated_importance / overlap_count
    
    return processed_importance, timeline

def read_and_process_importance_scores(file_path):
    try:
        # Read JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract metadata
        metadata = {
            'filename': data['metadata']['filename'],
            'label_explained': data['metadata']['id_explained'],
            'segment_length': data['metadata']['segment_length']
        }
        
        # Process importance scores
        processed_scores = {
            'RF': {
                    'method': data['importance_scores']['RF']['method'],
                    'processed_values': np.array(data['importance_scores']['RF']['values'])
            },
            'LR': {
                    'method': data['importance_scores']['LR']['method'],
                    'processed_values': np.array(data['importance_scores']['LR']['values'])
            },
            'SHAP': {
                    'method': data['importance_scores']['SHAP']['method'],
                    'processed_values': np.array(data['importance_scores']['SHAP']['values'])
            }
        }
    
        return {
            'metadata': metadata,
            'processed_scores': processed_scores
        }
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        return None
    except KeyError as e:
        print(f"Error: Missing expected key in JSON structure: {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        return None

def calculate_auc(values, segment_length, markers, intersection):
    overlap = 0  # adjust if necessary
    granularidad_ms = segment_length - overlap
    times_gt = markers

    times = generate_sequence(len(values))
    
    # Create the segmentation vector (labels: 1, -1, or 0)
    segmentation_labels = create_segmentation_vector(times_gt, times, granularidad_ms, intersection)

    # Filter out segments that are in the discard region (i.e., with label -1).
    valid_mask = segmentation_labels != -1
    filtered_segmentation = segmentation_labels[valid_mask]
    filtered_values = np.array(values)[valid_mask]

    if not (np.any(filtered_segmentation == 1) and np.any(filtered_segmentation == 0)):
        return None

    # Compute ranking scores.
    # Here we rank the importance scores: higher scores receive higher ranks.
    sorted_indices = np.argsort(filtered_values)[::-1]  
    ranking_scores = np.zeros_like(filtered_values, dtype=float)
    ranking_scores[sorted_indices] = np.linspace(1, 0, len(filtered_values))

    # Compute ROC curve and AUC using the filtered values.
    fpr, tpr, thresholds = roc_curve(filtered_segmentation, ranking_scores)

    roc_auc = auc(fpr, tpr)

    return roc_auc

def create_waveform_video_with_importances(waveform, processed_scores, output_file, 
                                         sample_rate=16000, fps=30, markers=None):
    """
    Create a video visualization of a waveform with multiple importance value plots
    
    Args:
        waveform: np.array of audio waveform
        processed_scores: dict of processed importance scores from JSON
        output_file: path to save the output video
        sample_rate: audio sample rate (default: 16000)
        fps: frames per second for video (default: 30)
        markers: list of time markers to show vertical lines (optional)
    """
    marker_colors = [
        '#FF0000',  # Red
        '#00FF00',  # Green
        '#0000FF',  # Blue
        '#FFA500',  # Orange
        '#800080',  # Purple
        '#00FFFF',  # Cyan
        '#FF00FF',  # Magenta
        '#FFD700',  # Gold
        '#98FB98',  # Pale Green
        '#DDA0DD',  # Plum
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the audio
        temp_audio = os.path.join(temp_dir, "temp_audio.wav")
        waveform_int16 = (waveform * 32767).astype(np.int16)
        wav.write(temp_audio, sample_rate, waveform_int16)
        
        temp_video = os.path.join(temp_dir, "temp_video.mp4")
        
        # Create time array
        times = np.arange(len(waveform)) / sample_rate
        total_duration = len(waveform) / sample_rate
        # Assuming values is the array you are working with

        
        interval_ms = processed_scores['metadata']['segment_length']
        

        # Create segments for LineCollection
        points = np.array([times, waveform]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create custom colormap (light blue to dark blue)
        colors = ['#E6F3FF', '#0343DF']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Calculate number of importance plots needed
        n_importance_plots = sum([
            1, 
            1,
            1, 
            1 
        ])
        
        # Create figure and axes
        fig, axes = plt.subplots(n_importance_plots, 1, figsize=(12, 3*n_importance_plots), 
                                gridspec_kw={'height_ratios': [1]*n_importance_plots},
                                sharex=True)
        
        importance_data = []
        methods = ['SHAP', 'LR', 'RF']
        for key in methods:
            lime_values = processed_scores['processed_scores'][key]['processed_values']
            num_segments = len(lime_values)
            timeline = np.arange(0, num_segments * interval_ms, interval_ms) / 1000  # Time in seconds
            lime_times = timeline
            importance_data.append((key, lime_values, lime_times))

        for idx, (method, values, time_points) in enumerate(importance_data):
            ax = axes[idx]
            
            # Plot the importance values
            ax.step(time_points, values, where='post', linewidth=1, color='black')
            ax.axhline(0, color='black', linewidth=1)
            
            y_min = min(0, values.min())
            y_max = values.max()
            padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Importance')
            
            # Compute AUC for the current method using your helper function.
            # Here, 'data' should be a dict with keys "metadata" and "importance_scores".
            # Adjust the 'intersection' parameter as needed.
            roc_auc = calculate_auc(values, interval_ms, markers, intersection=0.09)
            if roc_auc is not None:
                title_with_auc = f"{method} (AUC: {roc_auc:.2f})"
            else:
                title_with_auc = f"{method} (AUC: N/A)"
            
            ax.set_title(title_with_auc)
            if markers:
                for i, (start, end) in enumerate(markers):
                    # Get color from the list, cycling if needed
                    color = marker_colors[i % len(marker_colors)]
                    
                    # Add span with very light color
                    ax.axvspan(start, end, color=color, alpha=0.1)
                    
                    # Add vertical lines with stronger color
                    ax.axvline(x=start, color=color, linestyle='-', alpha=0.7, linewidth=2)
                    ax.axvline(x=end, color=color, linestyle='--', alpha=0.7, linewidth=2)
      
        ax_waveform = axes[-1]
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(lime_values.min(), lime_values.max()))
        lc.set_array(lime_values[:-1])  # Using naive importance for coloring
        lc.set_linewidth(1.5)
        
        ax_waveform.add_collection(lc)
        ax_waveform.set_xlim(0, total_duration)
        ax_waveform.set_ylim(waveform.min() - 0.1, waveform.max() + 0.1)
        ax_waveform.set_xlabel('Time (s)')
        ax_waveform.set_ylabel('Amplitude')
        ax_waveform.set_title('Waveform')
        
        # Adjust layout
        plt.tight_layout()
        
        # Store the previous lines to remove them in each frame
        prev_lines = []
        
        def animate(frame):
            # Remove previous lines
            for line in prev_lines:
                line.remove()
            prev_lines.clear()
            
            current_time = frame / fps
            
            # Add vertical line showing current position to all plots
            lines = []
            for ax in axes:
                line = ax.axvline(x=current_time, color='black', linestyle='-', alpha=0.3)
                lines.append(line)
                prev_lines.append(line)
                        
            return tuple(lines)
            
            # Set up the animation
        total_frames = int(total_duration * fps)
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=1000/fps, blit=True)
        
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        
        print("Saving temporary video...")
        anim.save(temp_video, writer=writer)
        
        # Clean up matplotlib
        plt.close()
        
        # Combine video and audio using ffmpeg
        print("Combining video and audio...")
        cmd = [
            'ffmpeg',
            '-i', temp_video,
            '-i', temp_audio,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-y',
            output_file
        ]
        subprocess.run(cmd, check=True)
        
        print(f"Final video with audio saved as {output_file}")

def create_visualization(waveform, json_file, output_file):
    processed_data = read_and_process_importance_scores(json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)

    create_waveform_video_with_importances(
        waveform=waveform,
        processed_scores=processed_data,
        output_file=output_file,
        sample_rate=16000,
        markers=data['metadata']['true_markers'],
    )


def map_metric_name(metric):

    metric_name_in_tsv = metric
    value = metric

    if metric == 'auc':
        value = 'roc_auc'
        intersection = '_0.05'
    if metric == 'leo_metric':
        intersection = '_0'
    if metric == 'auc_relaxed':
        value = 'roc_auc'
        intersection = '_0.09'
    if 'top' in metric:
        metric_name_in_tsv = 'score_curve'
        intersection = ''
    if 'adapt' in metric:
        metric_name_in_tsv = 'score_curve'
        intersection = ''

    return value, metric_name_in_tsv, intersection


def clean_and_eval(value):
    try:
        # Remove brackets
        value = value.strip("[]")
        # Replace multiple spaces with commas
        value = re.sub(r'\s+', ',', value)
        # Evaluate as a Python list
        return ast.literal_eval(f"[{value}]")
    except (SyntaxError, ValueError):
        return None  # Handle errors safely

def perc_segments_gt(filename, event_label, dataset):
    label = int(event_label)
    if dataset == 'drums':
        with open(f'/home/ec2-user/results/explanations_drums/{filename}/drums/ft_{label}_noise.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    if 'audioset' in dataset:
        with open(f'/home/ec2-user/results/explanations_audioset/{filename}/ast/ft_{label}_noise.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    if dataset == 'kws':
        with open(f'/home/ec2-user/results/explanations_kws/{filename}/kws/ft_{label}_noise.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    time_segments = data['metadata']['true_markers']
    ms = data['metadata']['segment_length'] / 1000
    if len(time_segments) ==0:
        print(filename, event_label)
    max_time = max(end for _, end in time_segments)
    fixed_intervals = np.arange(0, max_time + ms, ms)  # Adding 0.1 to include last segment

    # Count segments that overlap with any given time interval
    count = sum(
        any(start < t + 0.1 and end > t for start, end in time_segments) 
        for t in fixed_intervals
    )
    total_segments = len(data['importance_scores']['SHAP']['values'])

    return count*100/total_segments

def read_results_file(file_path, metric=None, method=None, name=None, dataset=None):

    try:
        df_combination = pd.read_csv(file_path, sep='\t')
        df_combination['method'] = method
        df_combination['name'] = name
        df_combination[metric] = 0

        if 'top' in metric and dataset != 'drums':
            df_combination['log_odds_curve'] = df_combination['score_curve_sacando_topk'].apply(ast.literal_eval)
            df_combination['event_label'] = df_combination['event_label'].apply(int)
            df_combination['log_odds'] = df_combination['actual_score'].apply(ast.literal_eval)
            
            # Loop through the rows of df 
            for index, row in df_combination.iterrows():
                if 'adapt' in metric:
                    perc = perc_segments_gt(row['filename'], row['event_label'], dataset)/100
                    print(row['filename'], row['event_label'], perc)
                else:
                    perc = int(re.sub('perc','',re.sub('top','', metric)))/100

                idx = int(len(row['log_odds_curve']) * perc)
                df_combination.at[index,metric] = float(row['log_odds'][row['event_label']] - row['log_odds_curve'][idx][row['event_label']])

                if 'rel' in metric:
                    df_combination.at[index,metric] /= row['log_odds'][row['event_label']]

        if 'top' in metric and dataset == 'drums':
            df_combination['log_odds_curve'] = df_combination['score_curve_sacando_topk'].apply(ast.literal_eval)
            df_combination['event_label'] = df_combination['event_label'].apply(int)
            df_combination['log_odds'] = df_combination['actual_score'].apply(clean_and_eval)

            # Loop through the rows of df 
            for index, row in df_combination.iterrows():
                if 'adapt' in metric:
                    perc = perc_segments_gt(row['filename'], row['event_label'], dataset)/100
                    print(row['filename'], row['event_label'], perc)
                else:
                    perc = int(re.sub('perc','',re.sub('top','', metric)))/100
                
                idx = int(len(row['log_odds_curve']) * perc)
                df_combination.at[index,metric] = row['log_odds'][row['event_label']] - row['log_odds_curve'][idx][row['event_label']]

                if 'rel' in metric:
                    df_combination.at[index,metric] /= row['log_odds'][row['event_label']]
        return df_combination

    except FileNotFoundError:
        raise Exception(f"File not found: {file_path}")

def mean_with_confint(samples):
    return evaluate_with_conf_int(np.array(samples), np.mean)


def barplot_with_ci(ax, data, dataset_name, metric, figsize=None, colormap='Spectral', legend=False):
    """ Make a bar plot for the input data. This should be a dictionary with one entry for each
    name in the legend (eg, each system). For each of those, the value should be another dictionary
    with one entry for each group being plotted (eg, each dataset). The entries within that inner
    dictionary should be a list with the center of the bar (the performance measured on the full
    test set), and the confidence interval as a list.
    For example:

        data = {'sys1': {'db1': (center11, (min11, max11)), 'db2': (center12, (min12, max12))},
                'sys2': {'db1': (center21, (min21, max21)), 'db2': (center22, (min22, max22))}}

        """

    cmap = matplotlib.cm.get_cmap(colormap)
    colors = [cmap(i/len(data)) for i in np.arange(len(data))]
    colors = [ 'royalblue', 'indianred', 'yellowgreen', 'grey']

    # The groups should be the same for all labels
    allgroups = list(data.values())[0].keys()

    barWidth = 1/(len(data)+1)

    group_starts = np.arange(len(allgroups))

    for j, (lname, lvalues) in enumerate(data.items()):

        xvalues = group_starts + barWidth * j

        # Plot the bars for the given top label across all groups
        yvalues = [lvalues[group][0] if group in lvalues else 0 for group in allgroups]
        ax.bar(xvalues, yvalues, color=colors[j], width = barWidth, label=lname)

        # Now plot a line on top of the bar to show the confidence interval
        for k, group in enumerate(allgroups):
            ci = lvalues[group][1]
            ax.plot(xvalues[k]*np.ones(2), ci, 'k')

    ax.set_xticks(group_starts + barWidth * (len(data)-1)/2 , allgroups)
    if legend == 1 or legend is True:
        ax.legend()
        ax.set_ylabel(f'{metric}')
    

    if 'AUC' in metric:
        ax.set_ylim([0.5,1])
    ax.set_title(dataset_name)
