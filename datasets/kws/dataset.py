from torch.utils.data import Dataset
import numpy as np
import torch
import librosa
import pandas as pd
from models.base_model import UpstreamDownstreamModel

class LibrispeechDataset(Dataset):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        row['filename'] = '/home/ec2-user' + row['filename']
        x, fs = librosa.core.load(row['filename'], sr=16000)
        return {'wav': x.astype(np.float32),
                'emotion': row['has_word'].astype(int)}
        
    def __len__(self):
        return len(self.metadata)


METADATA_PATH = '/home/ec2-user/mnt/data/librilight_med_24k/little_metadata.csv'

df_metadata = pd.read_csv(METADATA_PATH)
df_test = df_metadata.loc[df_metadata['split']=='test-clean']
dataset_test = LibrispeechDataset(df_test)

ud_model = UpstreamDownstreamModel(upstream='wav2vec2', 
                                   num_layers=13, 
                                   num_classes=1,
                                   hidden_sizes=[256])


checkpoint = torch.load('/home/ec2-user/Models/librispeech-kws-step2200.ckpt')['state_dict']
new_state_dict = {k.replace("mlp", "downstream"): v for k, v in checkpoint.items()}
ud_model.load_state_dict(new_state_dict)

ud_model.eval()
dataframe = []

for i, xin in enumerate(dataset_test):
    xin['wav_lens'] = torch.tensor([xin['wav'].shape[0]])
    xin['wav'] = torch.from_numpy(xin['wav'])[None,:]
    pred = ud_model(xin)
    if xin['emotion'] == 1 and (pred[0].item() > 0):
        dataframe.append(dataset_test.metadata.iloc[i])

kws_dataset = pd.DataFrame(dataframe)
kws_dataset = kws_dataset.sample(frac=1, random_state=None).reset_index(drop=True)

kws_dataset.to_csv('/home/ec2-user/explain_where/datasets/kws/kws_dataset.csv', index=False)