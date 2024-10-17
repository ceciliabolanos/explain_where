import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, writers
import librosa
import torchaudio
import json
import os
import subprocess
import ffmpeg

def mask_audio(audio_array, sample_rate, mask_duration_ms=500, shift_interval_ms=50, mask_value=0, predict_fn=None):
    # Calculate the number of samples for the mask duration and shift interval
    mask_samples = int(sample_rate * (mask_duration_ms / 1000))
    shift_samples = int(sample_rate * (shift_interval_ms / 1000))
    
    # Create a copy of the input array to avoid modifying the original
    masked_audio = np.copy(audio_array)
    results = []

    audios_mask = []
    for start in range(0, len(masked_audio), shift_samples):
        end = min(start + mask_samples, len(masked_audio))
        masked_audio[start:end] = 0
        results.append(predict_fn([masked_audio]))
        audios_mask.append(masked_audio)
        masked_audio = np.copy(audio_array)

    return results, audios_mask

def initialize_components(wav_array):
    samples_per_segment = int(500 * 16000 / 1000)
    components = []
    total_samples = len(wav_array)
    n_segments = total_samples // samples_per_segment

    for i in range(n_segments):
        start = i * samples_per_segment
        end = start + samples_per_segment

        segment = np.zeros_like(wav_array)
        segment[start:end] = wav_array[start:end]

        components.append(segment)

    if end < total_samples:
        start = end
        end = total_samples

        segment = np.zeros_like(wav_array)
        segment[start:end] = wav_array[start:end]

        components.append(segment)
    return components

def get_components(wav_array, ordered_weights, label='positive'):
    components = initialize_components(wav_array)
    if label=='positive':
        indices = [i for i, w in enumerate(ordered_weights) if w > 0]
    elif label=='negative':
        indices = [i for i, w in enumerate(ordered_weights) if w < 0]
    top_components = [components[i] * ordered_weights[i] for i in indices]

    return sum(top_components)


def distance_to_prob(results, real_log_probs, real_predicted_class, probability=None):
    if probability == 'yes':
        probabilities = [np.exp(pred) / np.sum(np.exp(pred)) for pred in results]
        gt_prob = (np.exp(real_log_probs) / np.sum(np.exp(real_log_probs)))[0][real_predicted_class]
        gt_probs = [prob[0][real_predicted_class] for prob in probabilities]
    else: 
        gt_prob = real_log_probs[0][real_predicted_class]
        gt_probs = [prob[0][real_predicted_class] for prob in results]

    # Calculate the distances between the GT probability and the probabilities for other classes
    distances = []
    for prob in gt_probs:
        distances.append([gt_prob - prob])

    return distances

def find_prediction_changes(results, predicted_class):
    return [i for i, pred in enumerate(results) if np.argmax(pred) != predicted_class]

def create_combined_video(wav_file, json_file, results_500, results_1000, results_2000, log_probs, predicted_class, emo):
    # Load and process audio file
    wav, sample_rate = torchaudio.load(wav_file)
    wav = wav[0, :]
    if sample_rate != 16000:
        wav = torchaudio.functional.resample(wav, sample_rate, 16000)
    wav = wav.view(-1)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    weights = data['weights']
    used_features = data['used_features']
    prediction_lime = data['prediction_lime']
    prediction_real = data['prediction_real']

    # Calculate probability distances and change points
    time_ms = np.arange(0, len(distance_to_prob(results_500, log_probs, predicted_class)) * 50, 50)
    change_points_500 = find_prediction_changes(results_500, predicted_class)
    change_points_1000 = find_prediction_changes(results_1000, predicted_class)
    change_points_2000 = find_prediction_changes(results_2000, predicted_class)
    
    # Compute spectrogram
    y, sr = librosa.load(wav_file, sr=16000)
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})
    
    # Upper plot (unified view)
    ax1_twin = ax1.twinx()
    
    total_duration = len(wav) / sample_rate
    segment_duration = total_duration / len(weights)
    weight_times = np.arange(0, total_duration, segment_duration)
    ordered_weights = [weights[i] for i in used_features]
    step_times = np.dstack((weight_times, np.append(weight_times[1:], total_duration))).flatten()
    step_weights = np.repeat(ordered_weights, 2)
    
    ax1.plot(step_times, step_weights, color='red', linewidth=2, label='Segment Weights')
    ax1.set_ylabel('Weight', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(-0.5, 0.5)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    ax1_twin.plot(time_ms/1000, distance_to_prob(results_500, log_probs, predicted_class), marker='', linestyle='-', color='blue', label='500ms window', alpha=0.7)
    ax1_twin.plot(time_ms/1000, distance_to_prob(results_1000, log_probs, predicted_class), marker='', linestyle='-', color='green', label='1000ms window', alpha=0.7)
    ax1_twin.plot(time_ms/1000, distance_to_prob(results_2000, log_probs, predicted_class), marker='', linestyle='-', color='purple', label='2000ms window', alpha=0.7)
    ax1_twin.set_ylabel('Score (Log Softmax) Distance (real - masked)', color='black')
    ax1_twin.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Horizontal line at y=0 for the twin y-axis

    for change_point in change_points_500:
        ax1.axvline(x=time_ms[change_point]/1000, color='blue', linestyle=':', alpha=0.7)
    for change_point in change_points_1000:
        ax1.axvline(x=time_ms[change_point]/1000, color='green', linestyle=':', alpha=0.7)
    for change_point in change_points_2000:
        ax1.axvline(x=time_ms[change_point]/1000, color='purple', linestyle=':', alpha=0.7)
    
    ax1.set_xlim(0, total_duration)
    ax1.set_xlabel('Time [sec]')
    ax1.set_title(f'Score for emotion: {emo} with Lime {prediction_lime} vs real prediction: {prediction_real}')
    
    # Lower plot (spectrogram)
    img = ax2.imshow(S_db, origin='lower', aspect='auto', cmap='viridis', 
                     extent=[0, total_duration, 0, sr/2])
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    
    # Animation function
    def animate(frame):
        current_time = frame / fps
        line1 = ax1.axvline(x=current_time, color='black', linestyle='-', alpha=0.3)
        line2 = ax2.axvline(x=current_time, color='red', linestyle='-', alpha=0.3)
        ax2.set_title(f'Spectrogram (Time: {current_time:.2f}s)')
        return line1, line2

    # Set up the animation
    fps = 30  # Frames per second for the video
    total_frames = int(total_duration * fps)
    
    anim = FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=True)
    
    # Set up the writer
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    
    # Save the animation without audio
    temp_video = "temp_combined_animation.mp4"
    anim.save(temp_video, writer=writer)
    
    # Clean up matplotlib resources
    plt.close(fig)
    
    # Add audio to the video
    output_file = "combined_animation_with_audio.mp4"
    
    if 'ffmpeg' in globals():
        # Use ffmpeg-python if available
        input_video = ffmpeg.input(temp_video)
        input_audio = ffmpeg.input(wav_file)
        ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_file).run(overwrite_output=True)
    else:
        # Fall back to subprocess if ffmpeg-python is not available
        cmd = [
            'ffmpeg',
            '-i', temp_video,
            '-i', wav_file,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_file
        ]
        subprocess.run(cmd, check=True)
    
    print(f"Combined animation with audio saved as {output_file}")
    
    # Remove the temporary video file
    os.remove(temp_video)