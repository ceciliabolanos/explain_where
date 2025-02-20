from torch.utils.data import Dataset
import numpy as np
import torch
from models.base_model import UpstreamDownstreamModel
import librosa
import pandas as pd

class DrumDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        row['filename'] = '/home/ec2-user' + row['filename']
        x, fs = librosa.core.load(row['filename'], sr=16000)
        return {'wav': x.astype(np.float32),
                'emotion': row['num_kicks'].astype(int)}
    def __len__(self):
        return len(self.metadata)
    

METADATA_PATH = '/home/ec2-user/mnt/data/drum_dataset/metadata.csv'

df_metadata = pd.read_csv(METADATA_PATH)

def partition(x):
    if x[-7] == '0':
        return 'test'
    elif x[-7] == '1':
        return 'validation'
    else:
        return 'train'
df_metadata['partition'] = df_metadata['filename'].apply(partition)

df_test = df_metadata.loc[df_metadata['partition']=='test']
dataset_test = DrumDataset(df_test)

ud_model = UpstreamDownstreamModel(upstream='wav2vec2',
                                   downstream='lstm',
                                   num_layers=13, 
                                   num_classes=6,
                                   hidden_sizes=[128],
                                   lstm_size=256)

ud_model.load_state_dict(torch.load('/home/ec2-user/Models/drums-step15000.ckpt')['state_dict'])

ud_model.eval()
dataframe = []

for i, xin in enumerate(dataset_test):
    xin['wav_lens'] = torch.tensor([xin['wav'].shape[0]])
    xin['wav'] = torch.from_numpy(xin['wav'])[None,:]
    
    if xin['emotion'] != 0:
        ud_model(xin)
        dataframe.append(dataset_test.metadata.iloc[i])

drums_dataset = pd.DataFrame(dataframe)
drums_dataset = drums_dataset.sample(frac=1, random_state=None).reset_index(drop=True)

drums_dataset.to_csv('/home/ec2-user/explain_where/datasets/drums/drums_dataset.csv', index=False)