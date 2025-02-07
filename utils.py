import os
import tempfile
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, writers
import scipy.io.wavfile as wav
import json
import pandas as pd
import ast


def open_json(file_path):
    """
    Open and read a JSON file
    """
    with open(file_path, 'r') as f:
        return json.load(f)


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


def get_segments(base_segment_id, label):
    # Filter DataFrame for the given base_segment_id and label
    df = pd.read_csv("/home/ec2-user/Datasets/Audioset/labels/audioset_eval.csv")  # Adjust path if necessary

    filtered_df = df[(df['base_segment_id'] == base_segment_id) & (df['father_id_ast'] == label)]
    
    if filtered_df.empty:
        return []  # Return empty list if no matching row is found
    
    # Select columns that start with "segment_"
    segment_columns = [col for col in df.columns if col.startswith("segment_")]
    
    # Extract non-null segment values and parse lists
    segment_lists = []
    for col in segment_columns:
        value = filtered_df[col].values[0]  # Get the value
        if isinstance(value, str) and value.startswith("["):  # Ensure it's a list-like string
            segment_lists.append(ast.literal_eval(value))  # Convert to actual list
    
    return segment_lists

def read_and_process_importance_scores(file_path):
    """
    Read and process the importance scores JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Processed data with metadata and processed importance scores
    """
    try:
        # Read JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract metadata
        metadata = {
            'filename': data['metadata']['filename'],
            'label_explained': data['metadata']['label_explained']
        }
        
        # Process importance scores
        processed_scores = {
            'naive': {
                'method': data['importance_scores']['naive']['method'],
                'original_values': np.array(data['importance_scores']['naive']['values'])
            },
            'random_forest': {
                'tree_importance': {
                    'method': data['importance_scores']['random_forest']['tree_importance']['method'],
                    'original_values': np.array(data['importance_scores']['random_forest']['tree_importance']['values'])
                },
                'shap': {
                    'method': data['importance_scores']['random_forest']['shap']['method'],
                    'original_values': np.array(data['importance_scores']['random_forest']['shap']['values'])
                }
            },
            'linear_regression': {
                'masked': {
                    'method': data['importance_scores']['linear_regression']['masked']['method'],
                    'original_values': np.array(data['importance_scores']['linear_regression']['masked']['values']['coefficients'])
                },
            }
        }
        
        # Process naive scores
        processed_scores['naive']['processed_values'], processed_scores['naive']['time_points'] = \
            process_importance_values(processed_scores['naive']['original_values'])
        
        # Process random forest scores
        for key in ['tree_importance', 'shap']:
            processed_scores['random_forest'][key]['processed_values'], \
            processed_scores['random_forest'][key]['time_points'] = \
                process_importance_values(processed_scores['random_forest'][key]['original_values'])
        
        # Process linear_regression scores
        for key in ['masked']:
            processed_scores['linear_regression'][key]['processed_values'], \
            processed_scores['linear_regression'][key]['time_points'] = \
                process_importance_values(processed_scores['linear_regression'][key]['original_values'])
        
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
    
def create_waveform_video_with_importances(waveform, scores_yamnet, class_index, processed_scores, output_file, 
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
        
        # Create segments for LineCollection
        points = np.array([times, waveform]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create custom colormap (light blue to dark blue)
        colors = ['#E6F3FF', '#0343DF']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        
        # Calculate number of importance plots needed
        n_importance_plots = sum([
            1,  # naive
            1,  # tree (masked, random forest)
            1,  # shap (masked, random forest)
            1,  # linear regression (masked, noise)
        ])
        
        # Create figure and axes
        fig, axes = plt.subplots(n_importance_plots + 2, 1, figsize=(12, 3*n_importance_plots), 
                                gridspec_kw={'height_ratios': [1]*n_importance_plots +  [2, 2]},
                                sharex=True)
        
        importance_data = []


        # Add naive importance values
        naive_values = processed_scores['naive']['processed_values']
        naive_times = processed_scores['naive']['time_points']
        importance_data.append(('Naive', naive_values, naive_times))
        
        # Add Random Forest importance values
        for key in ['tree_importance', 'shap']:
            rf_values = processed_scores['random_forest'][key]['processed_values']
            rf_times = processed_scores['random_forest'][key]['time_points']
            importance_data.append((f'Random Forest ({key})', rf_values, rf_times))
        
        # Add LIME importance values
        for key in ['masked']:
            lime_values = processed_scores['linear_regression'][key]['processed_values']
            lime_times = processed_scores['linear_regression'][key]['time_points']
            importance_data.append((f'Linear Regression ({key})', lime_values, lime_times))

        for idx, (title, values, time_points) in enumerate(importance_data):
            ax = axes[idx]
            
            # Plot the importance values as a line
            ax.plot(time_points, values, 'k-', linewidth=1, color='black')
            
            # Fill between the line and zero
            ax.fill_between(time_points, values, 0, alpha=0.2, color='gray')
            
            # Calculate dynamic y-axis limits with padding
            y_min = min(0, values.min())  # Include 0 in range
            y_max = values.max()
            padding = (y_max - y_min) * 0.1  # Add 10% padding
            
            ax.set_title(title)
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel('Importance')
            
            # Add markers if provided
            if markers:
                for i, (start, end) in enumerate(markers):
                    # Get color from the list, cycling if needed
                    color = marker_colors[i % len(marker_colors)]
                    
                    # Add span with very light color
                    ax.axvspan(start, end, color=color, alpha=0.1)
                    
                    # Add vertical lines with stronger color
                    ax.axvline(x=start, color=color, linestyle='-', alpha=0.7, linewidth=2)
                    ax.axvline(x=end, color=color, linestyle='--', alpha=0.7, linewidth=2)
                    
        ################ recently added
        ax_scores = axes[-2]
        class_scores = scores_yamnet[:, class_index]
        print(class_scores)
        score_times = np.linspace(0, len(waveform)/sample_rate, len(class_scores))
        
        # Plot the class scores as a line
        ax_scores.plot(score_times, class_scores, color='blue', linewidth=2)
        ax_scores.fill_between(score_times, class_scores, 0, alpha=0.2, color='blue')
        
        # Calculate dynamic y-axis limits with padding for scores
        score_min = min(0, class_scores.min())
        score_max = class_scores.max()
        score_padding = (score_max - score_min) * 0.1
        
        ax_scores.set_ylim(score_min - score_padding, score_max + score_padding)
        ax_scores.grid(True, alpha=0.3)
        ax_scores.set_title(f'Yamnet')
        ax_scores.set_ylabel('Score')
        ax_scores.set_xlabel('Time (seconds)')
      
        ax_waveform = axes[-1]
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(naive_values.min(), naive_values.max()))
        lc.set_array(naive_values[:-1])  # Using naive importance for coloring
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
            
            # Update title with current time
            axes[0].set_title(f'Importance Values Over Time (Current: {current_time:.2f}s)')
            
            return tuple(lines)
            
            # Set up the animation
        total_frames = int(total_duration * fps)
        anim = FuncAnimation(fig, animate, frames=total_frames, 
                           interval=1000/fps, blit=True)
        
        # Set up the writer
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        
        # Save the animation without audio
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


def create_visualization(waveform, scores_yamnet, class_index, json_file, output_file, markers=None):
    """
    Create a complete visualization from audio and JSON files
    
    Args:
        audio_file: path to the audio file
        json_file: path to the JSON file with importance scores
        output_file: path for the output video
        markers: optional list of time markers
    """
    processed_data = read_and_process_importance_scores(json_file)

    create_waveform_video_with_importances(
        waveform=waveform,
        scores_yamnet=scores_yamnet,
        class_index=class_index,
        processed_scores=processed_data['processed_scores'],
        output_file=output_file,
        sample_rate=16000,
        markers=markers
    )