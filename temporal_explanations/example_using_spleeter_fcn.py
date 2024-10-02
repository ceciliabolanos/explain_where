from audioLIME.data_provider import RawAudioProvider
from audioLIME.factorization import TemporalSegmentationFactorization
from audioLIME import lime_audio
import soundfile as sf
import librosa
import os

from sota_utils import prepare_config, get_predict_fn

if __name__ == '__main__':

    wav_path = "/home/cbolanos/data/IEMOCAP/Session2/sentences/wav/Ses02F_impro07/Ses02F_impro07_F022.wav"
    model_path = "/home/cbolanos/experiments/iemocap_whisper/fold_2/save/CKPT+2024-09-29+17-11-59+00/model.ckpt"
    ### emotion: angry
    data_provider = RawAudioProvider(wav_path)
    wav, feat_len = data_provider.get_mix()

    config = prepare_config(num_classes=4,hidden_size=128, model_path=model_path, feat_len=feat_len)
    predict_fn = get_predict_fn(config)

    spleeter_factorization = TemporalSegmentationFactorization(data_provider, wav, n_temporal_segments=10)

    explainer = lime_audio.LimeAudioExplainer(verbose=True, absolute_feature_sort=False)

    explanation = explainer.explain_instance(factorization=spleeter_factorization,
                                             predict_fn=predict_fn,
                                             top_labels=1,
                                             num_samples=5000,
                                             batch_size=32
                                             )

    label = list(explanation.local_exp.keys())[0]
    top_components, component_indeces = explanation.get_sorted_components(label,
                                                                          positive_components=True,
                                                                          negative_components=False,
                                                                          num_components=3,
                                                                          return_indeces=True)

    print("predicted label:", label)
    output_file =wav_path.split("/")[-1]
    sf.write(os.path.join("output",f"explanation{output_file}"), sum(top_components), 16000)

