import torch
import pandas as pd
import soundfile as sf
import pytorch_lightning as pl
from s3prl.nn import S3PRLUpstream
import torchmetrics
import numpy as np


class UpstreamDownstreamModel(pl.LightningModule):
    def __init__(self, upstream, num_layers=13, hidden_sizes=[128], num_classes=4, downstream='mlp', lstm_size=None):
        super().__init__()
        self.upstream = S3PRLUpstream(upstream)
        self.num_features = self.upstream.hidden_sizes[-1]
        self.upstream.eval()
        self.layer_weights = torch.nn.Parameter(torch.randn(num_layers))
        self.downstream_type = downstream
        self.num_classes = num_classes
        if downstream == 'mlp':
            mlp_ch = [self.num_features] + hidden_sizes + [num_classes]
            mlp_layers = [torch.nn.Sequential(torch.nn.Linear(chi, cho), torch.nn.ReLU()) for chi, cho in zip(mlp_ch[:-2],mlp_ch[1:-1])]
            mlp_layers += [torch.nn.Linear(mlp_ch[-2], mlp_ch[-1])]
            self.downstream = torch.nn.Sequential(*mlp_layers)
        elif downstream == 'lstm':
            self.downstream_lstm = torch.nn.LSTM(self.num_features, lstm_size, batch_first=True)
            mlp_ch = [lstm_size] + hidden_sizes + [num_classes]
            mlp_layers = [torch.nn.Sequential(torch.nn.Linear(chi, cho), torch.nn.ReLU()) for chi, cho in zip(mlp_ch[:-2],mlp_ch[1:-1])]
            mlp_layers += [torch.nn.Linear(mlp_ch[-2], mlp_ch[-1])]
            self.downstream_mlp = torch.nn.Sequential(*mlp_layers)
            
        if num_classes>1:
            self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
            self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        else:
            self.train_acc = torchmetrics.Accuracy(task='binary')
            self.val_acc = torchmetrics.Accuracy(task='binary')

    def forward(self, x):
        with torch.no_grad():
            x['upstream_outs'], x['upstream_lens'] = self.upstream(x['wav'], x['wav_lens'])
        upstream_outs = torch.stack(x['upstream_outs'])
        layer_w = torch.softmax(self.layer_weights, dim=0)[:, None, None, None]
        upstream_pooled = torch.sum(layer_w*upstream_outs, dim=0)
        if self.downstream_type == 'mlp':
            padding_mask = torch.arange(0, upstream_pooled.shape[1], device=upstream_outs.device)[None,:] < x['upstream_lens'][0][:, None]
            upstream_pooled = torch.sum(upstream_pooled * padding_mask[:,:,None], dim=1)/torch.sum(padding_mask[:,:,None], dim=1)
            x['pooled_upstream'] = upstream_pooled
            x['padding_mask'] = padding_mask
            return self.downstream(x['pooled_upstream'])
        elif self.downstream_type == 'lstm':
            lstm_in = torch.nn.utils.rnn.pack_padded_sequence(upstream_pooled, 
                                                              x['upstream_lens'][0].to(device='cpu', dtype=torch.int64),
                                                              batch_first=True,
                                                              enforce_sorted=False)
            lstm_out, (hn, cn) = self.downstream_lstm(lstm_in)
            return self.downstream_mlp(hn[0])
        else:
            raise ValueError('Unknown downstream_type {}'.format(self.downstream_type))
        
class CoughModel():
    def __init__(self, audio_path, id_to_explain: int):
        self.id_to_explain = id_to_explain
        self.audio_path = audio_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UpstreamDownstreamModel(upstream='wav2vec2', 
                                   num_layers=13, 
                                   num_classes=4,
                                   hidden_sizes=[256])

        checkpoint = torch.load('/home/ec2-user/Models/iemocap-cough-1340.ckpt')['state_dict']
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
        # Load audio file
        x, fs = sf.read(self.audio_path)
        x = x.astype(np.float32)
        
        # Prepare input dictionary
        xin = {
            'wav': torch.from_numpy(x)[None, :],
            'wav_lens': torch.tensor([x.shape[0]])
        }
        
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in xin.items()}
            logits = self.model(inputs)
        
        # Extract predicted emotion
        pred_emotion = logits.cpu().tolist()[0][self.id_to_explain]
        
        return x, pred_emotion
    
