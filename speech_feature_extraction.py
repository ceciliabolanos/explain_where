import torch
import os
import json
import numpy as np
import argparse
from tqdm import tqdm
import torchaudio
import torch.nn.functional as F
SAMPLING_RATE=16000

from transformers import WhisperFeatureExtractor, WhisperForAudioClassification, WhisperConfig


def load_model(source, model_path, device):
    config = WhisperConfig
    model = WhisperForAudioClassification
    processor = WhisperFeatureExtractor.from_pretrained(source)
    
    model = model.from_pretrained(
        source, cache_dir=model_path
    )
    
    model.eval()
    model.to(device)
    return (model, processor)

def extract_whisper_feature(wav_path, channel, model, output_norm, all_layers, device, start_time = None, end_time = None):
    model, processor = model
    if start_time is not None and end_time is not None:
        sample_rate = torchaudio.info(wav_path).sample_rate
        frame_offset = int(start_time * sample_rate)
        num_frames = int(end_time * sample_rate) - frame_offset
        wav, sr = torchaudio.load(wav_path, frame_offset = frame_offset, num_frames = num_frames)
    else:    
        wav, sr = torchaudio.load(wav_path)
    channel = channel -1
    wav = wav[channel, :]
    if sr != SAMPLING_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLING_RATE)
    wav = wav.view(-1)
    feat_len = int(wav.size(0) // 320)
    input_features = processor(wav, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features
    padding_mask = torch.ones_like(input_features)
    input_features = input_features.to(model.device)

    out = model(input_features, output_hidden_states = True)

    if all_layers:
        out = torch.stack(list(out.hidden_states), dim=0)
        out = out[:,:,:feat_len,:]
        norm_shape = out.shape[-3:]
    else:
        out = out.hidden_states[-1]
        out = out[:,:feat_len,:]
        norm_shape = out.shape

    if output_norm:
        out = F.layer_norm(out, norm_shape[1:])

    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type = str, required = True)
    parser.add_argument('--dump_dir', type = str, required = True)
    parser.add_argument('--device', type = str, default = 'cuda')
    parser.add_argument('--data', type =str, required = True)
    parser.add_argument('--all_layers', default = False, action = 'store_true')
    parser.add_argument('--output_norm', default = False, action = 'store_true')
    
    args = parser.parse_args()
    print(args)

    # load metadata
    f = open(args.data)
    data = json.load(f)
    f.close()
    
    seg_ids = data.keys()
    print(f'load in {len(seg_ids)} segments')

    # load models
    source = 'openai/whisper-large-v3' 
    model = load_model(source, args.model_path, args.device)
    feat_func = extract_whisper_feature    
        
    # load speech ssl models
    for seg_id in tqdm(seg_ids):
        sample = data[seg_id]
        wav_path = sample['wav']
        channel = sample['channel']
        dur = float(sample['length'])
        if dur > 30. :
            print(f"SKIP {wav_path} because its duration is {dur}, which is too long!")
            continue

        if 'start_time' in sample and 'end_time' in sample:
            start_time = sample['start_time']
            end_time = sample['end_time']
        else:
            start_time = None
            end_time = None    
        assert os.path.exists(wav_path), f'{wav_path} does not exists on your disk'
        try:
            torchaudio.load(wav_path)
        except:
            print(f'ERROR!! wav file {wav_path} can not be loaded!')
            continue   
        feat = feat_func(wav_path, channel, model, args.output_norm, args.all_layers, args.device, start_time, end_time)
        feat = feat.data.cpu().numpy()
        if args.all_layers:
            feat = np.squeeze(feat, 1)
        else:
            feat = feat[0]    
        save_path = os.path.join(args.dump_dir, seg_id + '.npy')
        print(f'seg_id:{seg_id}\tfeat_shape:{feat.shape}\tsave_path:{save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        np.save(save_path, feat)




