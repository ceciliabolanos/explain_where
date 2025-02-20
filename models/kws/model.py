import torch
import numpy as np
import librosa
from models.base_model import Model, UpstreamDownstreamModel

PATH_KWS_MODEL = '/home/ec2-user/Models/librispeech-kws-step2200.ckpt'

class KWSModel(Model):
    def __init__(self, audio_path, id_to_explain: int):
        self.id_to_explain = id_to_explain
        self.audio_path = audio_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UpstreamDownstreamModel(upstream='wav2vec2', 
                                   num_layers=13, 
                                   num_classes=1,
                                   hidden_sizes=[256])

        checkpoint = torch.load(PATH_KWS_MODEL)['state_dict']
        new_state_dict = {k.replace("mlp", "downstream"): v for k, v in checkpoint.items()}
        self.model.load_state_dict(new_state_dict)
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
            
            return logits.cpu().tolist()
        
        return predict_fn
    
    def process_input(self):
        x, fs = librosa.core.load(f'/home/ec2-user/{self.audio_path}', sr=16000)
        x = x.astype(np.float32)
        
        xin = {
            'wav': torch.from_numpy(x)[None, :],
            'wav_lens': torch.tensor([x.shape[0]])
        }
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in xin.items()}
            logits = self.model(inputs)
                
        return x, logits.cpu().tolist()[0]
    
