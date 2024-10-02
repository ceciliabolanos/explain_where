
from argparse import Namespace
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor
import logging
from typing import List, Tuple


class SuperbBaseModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.pre_net = nn.Linear(input_size, hidden_size)
        self.post_net = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, padding_mask = None):
        x = self.relu(self.pre_net(x))
        
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()
            x = x.sum(dim=1) / padding_mask.float().sum(dim=1, keepdim=True)  # Compute average
        else:
            x = x.mean(dim=1)
        x = self.post_net(x)
        return x
    
class CustomWhisperForAudioClassification(nn.Module):
        def __init__(self, num_classes: int, hidden_size: int):
            super().__init__()
            self.whisper = WhisperModel.from_pretrained("openai/whisper-large-v3", cache_dir='/home/cbolanos/experiments/pretrained_models/whisper').encoder
            self.classifier = SuperbBaseModel(
                input_size=1280,
                output_size=num_classes,
                hidden_size=hidden_size
            )

        def forward(self, input_features, feat_len):
            outputs = self.whisper(input_features, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            hidden_states = hidden_states[:,:feat_len,:].data.cpu().numpy()
            # hidden_states = hidden_states[0]
            hidden_states = torch.from_numpy(hidden_states).to('cuda')
            print(f"Input feature shape: {input_features.shape}")
            print(f"Hidden states shape: {hidden_states.shape}")
            logits = self.classifier(hidden_states, padding_mask=None)
            

            print(f"Logits shape: {logits.shape}")
            return logits




# sota config
def prepare_config(num_classes=4, hidden_size=128, model_path='', feat_len=''):
    config = Namespace()
    config.num_classes = num_classes
    config.hidden_size = hidden_size
    config.model_path = model_path
    config.feat_len = feat_len
    return config


def get_predict_fn(config):
    model = CustomWhisperForAudioClassification(config.num_classes, config.hidden_size).to("cuda")
    
    # Load the state dict
    state_dict = torch.load(config.model_path, map_location="cuda")
    
    # Modify the state dict keys to match the model structure
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('0.'):
            new_k = k[2:]  # Remove the '0.' prefix
            new_state_dict[new_k] = v
    
    # Load only the classifier weights
    model.classifier.load_state_dict(new_state_dict)
    model.eval()

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    def predict_fn(wav_array):
        input_features_list = []
        
        for wav in wav_array:
            input_features = processor(wav, sampling_rate=16000, return_tensors="pt").input_features
            input_features_list.append(input_features)
        
        input_features_batch = torch.cat(input_features_list, dim=0)
        input_features_batch = input_features_batch.to('cuda')

        with torch.no_grad():
            logits = model(input_features_batch, config.feat_len)

        probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

        return probabilities.detach().cpu().numpy()

    return predict_fn





    

