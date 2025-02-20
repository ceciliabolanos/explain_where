import torch
import numpy as np
import librosa
from models.base_model import Model, UpstreamDownstreamModel
from models.utils import compute_log_odds

PATH_DRUMS_MODEL = '/home/ec2-user/Models/drums-step15000.ckpt'

class DrumsModel(Model):
    def __init__(self, audio_path, id_to_explain: int):
        self.id_to_explain = id_to_explain
        self.audio_path = audio_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UpstreamDownstreamModel(upstream='wav2vec2',
                                   downstream='lstm',
                                   num_layers=13, 
                                   num_classes=6,
                                   hidden_sizes=[128],
                                   lstm_size=256)

        self.model.load_state_dict(torch.load(PATH_DRUMS_MODEL)['state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def get_predict_fn(self):
        def predict_fn(wav_array):
            if not isinstance(wav_array, list):
                wav_array = [wav_array]
            
            inputs = {
                'wav': torch.stack([torch.from_numpy(audio).float() for audio in wav_array]).to(self.device),
                'wav_lens': torch.tensor([len(audio) for audio in wav_array]).to(self.device)
            }
        
            with torch.no_grad():
                logits = self.model(inputs)
            logodds = compute_log_odds(logits.cpu().tolist())
            return logodds.tolist()
        
        return predict_fn
    
    def process_input(self):
        # Load audio file
        x, fs = librosa.core.load(f'/home/ec2-user/{self.audio_path}', sr=16000)
        x = x.astype(np.float32)
        
        xin = {
            'wav': torch.from_numpy(x)[None, :],
            'wav_lens': torch.tensor([x.shape[0]])
        }
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in xin.items()}
            logits = self.model(inputs)
        
        # For this model, we need to calculate log-odds
        logodds = compute_log_odds(logits.cpu().tolist())

        return x, logodds[0]
    