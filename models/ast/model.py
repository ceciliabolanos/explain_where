from transformers import AutoFeatureExtractor, ASTForAudioClassification
import torch
import pandas as pd
import soundfile as sf
from scipy.signal import resample

class ASTModel():
    def __init__(self, filename, id_to_explain: int):
        self.id_to_explain = id_to_explain
        self.filename = filename

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = self.model.to(self.device)
        ### cambiar esto
        self.df = pd.read_csv('/home/ec2-user/IEMOCAP/partitions/iemocap_data.csv')
        self.df.set_index(self.df.columns[0], inplace=True)
     
    
    def get_predict_fn(self):
      
        def predict_fn(wav_array):
            if not isinstance(wav_array, list):
                wav_array = [wav_array]
            
            inputs_list = [self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt") 
                        for audio in wav_array]
            
            inputs = {
                k: torch.cat([inp[k] for inp in inputs_list]).to("cuda")
                for k in inputs_list[0].keys()
            }
            with torch.no_grad():
                logits = self.model(**inputs).logits
                
            return logits.cpu().tolist()
            
        return predict_fn
    
    def process_input(self): 
        audio_path = f'/mnt/shared/alpha/hdd6T/Datasets/audioset_eval_wav/{self.filename}.wav'

        wav_data, sample_rate = sf.read(audio_path)
        if sample_rate != 16000:
            wav_data = resample(wav_data, int(len(wav_data) * 16000 / sample_rate))
        if len(wav_data.shape) > 1 and wav_data.shape[1] == 2:
            wav_data = wav_data.mean(axis=1)

        inputs = self.feature_extractor(wav_data, 
                                        sampling_rate=16000, 
                                        return_tensors="pt") 
        
        logits = self.model(**inputs).logits

        print(logits)        
        pred_emotion = logits[0][self.id_to_explain]

        return wav_data, pred_emotion