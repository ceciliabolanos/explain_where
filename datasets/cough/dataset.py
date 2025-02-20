"""
Script that selects the instances that the model trained with cough predict well.

"""
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import torch
import pandas as pd
from models.base_model import UpstreamDownstreamModel


METADATA_PATH = '/home/ec2-user/mnt/data/IEMOCAP-happy-cough/metadata.csv'


class IEMOCAPDataset(Dataset):
    def __init__(self, metadata, class_map=['neutral','happiness','anger','sadness']):
        super().__init__()
        self.metadata = metadata
        self.class_map = {k:i for i,k in enumerate(class_map)}

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        row['filename'] = '/home/ec2-user' + row['filename']
        x, fs = sf.read(row['filename'])
        return {'wav': x.astype(np.float32),
                'emotion': self.class_map[row['emotion']]}
        
    def __len__(self):
        return len(self.metadata)

df_metadata = pd.read_csv(METADATA_PATH)
df_metadata['session'] = df_metadata['filename'].apply(lambda x: int(x.split('/')[-5][-1]))
df_metadata['partition'] = df_metadata['session'].apply(lambda x: 'Train' if x<5 else 'Test')
df_test = df_metadata.loc[df_metadata['partition']=='Test']
dataset_test = IEMOCAPDataset(df_test)

ud_model_cough = UpstreamDownstreamModel(upstream='wav2vec2', 
                                   num_layers=13, 
                                   num_classes=4,
                                   hidden_sizes=[256])

ud_model_nocough = UpstreamDownstreamModel(upstream='wav2vec2', 
                                   num_layers=13, 
                                   num_classes=4,
                                   hidden_sizes=[256])

checkpoint = torch.load('/home/ec2-user/Models/iemocap-cough-1340.ckpt')['state_dict']
new_state_dict = {k.replace("mlp", "downstream"): v for k, v in checkpoint.items()}
ud_model_cough.load_state_dict(new_state_dict)


checkpoint1 = torch.load('/home/ec2-user/Models/iemocap-nocough-1340.ckpt')['state_dict']
new_state_dict1 = {k.replace("mlp", "downstream"): v for k, v in checkpoint1.items()}
ud_model_nocough.load_state_dict(new_state_dict1)

dataframe = []

for i, xin in enumerate(dataset_test):
    if xin['emotion'] == 1:
        xtin = xin.copy()
        xtin['wav_lens'] = torch.tensor([xtin['wav'].shape[0]])
        xtin['wav'] = torch.from_numpy(xtin['wav'])[None,:]
        xin['wav_lens'] = torch.tensor([xin['wav'].shape[0]])
        xin['wav'] = torch.from_numpy(xin['wav'])[None,:]
        
        pred_model_cough = ud_model_cough(xin)
        pred_model_nocough = ud_model_nocough(xtin)
        if (pred_model_cough.detach().cpu().numpy()[0].argmax() == 1) and (pred_model_nocough.detach().cpu().numpy()[0].argmax() != 1):
            dataframe.append(dataset_test.metadata.iloc[i])


cough_happy = pd.DataFrame(dataframe)

cough_happy.to_csv('/home/ec2-user/explain_where/datasets/cough/cough_happy.csv', index=False)