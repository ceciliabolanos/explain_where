import librosa
import numpy as np
import os
import matplotlib.pyplot as plt

# Función para calcular la desviación estándar de una lista de audios
def calculate_std_of_waveforms(audio_paths):
    stds = []
    for folder_name in os.listdir(audio_paths):
        # Cargar el audio (ajusta el sr si es necesario)
        # drums = os.listdir(f'/home/ec2-user/mnt/data/drum_dataset/{folder_name}')
        # /home/ec2-user/Datasets/audioset_eval_wav
        waveform, sr = librosa.load(f'/home/ec2-user/mnt/data/drum_dataset/{folder_name}', sr=16000)
        # Calcular la desviación estándar de la waveform
        std = np.std(waveform)
        stds.append(std)
    
    plt.figure(figsize=(10, 6))
    plt.hist(stds, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribución de la Desviación Estándar de las Waveforms')
    plt.xlabel('Desviación Estándar')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.savefig('std_distribution_drums.png')  # Guardar la imagen del gráfico
    plt.show()

    return np.mean(stds)

# Listar los audios en la base de datos 1 y 2 (ajusta las rutas a tus carpetas)
# db1_audio_paths = [os.path.join('/home/ec2-user/mnt/data/drum_dataset', f) for f in os.listdir('/home/ec2-user/mnt/data/drum_dataset') if f.endswith('.wav')]
db2_audio_paths = [os.path.join('/home/ec2-user/Datasets/audioset_eval_wav', f) for f in os.listdir('/home/ec2-user/Datasets/audioset_eval_wav') if f.endswith('.wav')]

# Calcular la desviación estándar de las waveforms de cada base de datos
# std_db1 = calculate_std_of_waveforms(db1_audio_paths)
std_db2 = calculate_std_of_waveforms('/home/ec2-user/results1/explanations_drums')

print(f"Desviación estándar de la base de datos Drums: {std_db2}")
# print(f"Desviación estándar de la base de datos Audioset: {std_db2}")
