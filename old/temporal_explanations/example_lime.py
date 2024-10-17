from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import TemporalSegmentationFactorization
from audioLIME import lime_audio
import soundfile as sf
import os

from sota_utils import prepare_config, get_predict_fn
import json
    
def process_file(wav_path, emo):
    model_path = "/home/cbolanos/experiments/iemocap_whisper/fold_2/save/CKPT+2024-10-04+15-23-48+00/model.ckpt"
    data_provider = RawAudioProvider(wav_path)
    wav, feat_len = data_provider.get_mix()
    config = prepare_config(num_classes=4, hidden_size=256, model_path=model_path, feat_len=feat_len)
    predict_fn = get_predict_fn(config)
    temporal_factorization = TemporalSegmentationFactorization(data_provider, wav, segment_duration=500, window_duration=0)
    explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)
    explanation = explainer.explain_instance(factorization=temporal_factorization,
                                             predict_fn=predict_fn,
                                             top_labels=1,
                                             num_samples="exhaustive",
                                             batch_size=32
                                            )
    label = list(explanation.local_exp.keys())[0]
    
    top_components, component_indeces = explanation.get_sorted_components(label,
                                                                          positive_components=True,
                                                                          negative_components=False,
                                                                          num_components=4,
                                                                          return_indeces=True)
    
    used_features, weights, pvals = explanation.get_components_with_stats(label)

    print(f'File: {wav_path}')
    print(f'Emotion: {emo}')
    print(f'Predicted label: {label}')
    print(f'Components: {used_features}')
    
    output_file = os.path.basename(wav_path)
    data_to_save = {
        'weights': weights.tolist(),
        'pvals': pvals.tolist(),
        'used_features': used_features.tolist(),
        'prediction_lime': float(explanation.local_pred[0]),
        'prediction_real': float(explanation.neighborhood_labels[0, label])
    }

    output_dir = f"/home/cbolanos/experiments/emotion/{output_file}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "stats.json"), 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)


    sf.write(os.path.join(output_dir, f"explanation_{emo}.wav"), sum(top_components), 16000)
    sf.write(os.path.join(output_dir, f"real_{emo}.wav"), wav, 16000)

if __name__ == '__main__':
    # List of files to process
    files_to_process = [
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro05/Ses02M_impro05_F005.wav", "emo": "neu"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro08/Ses02M_impro08_F010.wav", "emo": "neu"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro02/Ses02M_impro02_F004.wav", "emo": "sad"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro02/Ses02M_impro02_M001.wav", "emo": "sad"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro01/Ses02M_impro01_M007.wav", "emo": "ang"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02F_impro05/Ses02F_impro05_F006.wav", "emo": "ang"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02F_impro07/Ses02F_impro07_M005.wav", "emo": "exc"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro07/Ses02M_impro07_F007.wav", "emo": "exc"},
        # {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02M_impro03/Ses02M_impro03_M010.wav", "emo": "hap"}
         {"wav": "/home/cbolanos/data/2018_IEMOCAP/Session2/sentences/wav/Ses02F_script01_1/Ses02F_script01_1_M013.wav", "emo": "ang"}

    ]

    # Process each file
    for file_info in files_to_process:
        process_file(file_info["wav"], file_info["emo"])
