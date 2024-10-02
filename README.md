# setup the environment
cd examples/sb
conda create -n conda-env -f environment.yml
conda activate conda-env

# extract features
python3 speech_feature_extraction.py \
	--model_name wav2vec2-base \
	--model_path pretrained_models/wav2vec2-base \
	--dump_dir dump \
	--device cuda \
	--data data/iemocap/iemocap.json \
	--output_norm

# run training & evaluation

python3 train/train.py \
	hparams/whisper-large-v3_freeze.yaml \
	--output_folder /home/cbolanos/experiments/iemocap_whisper/fold_2 \
    --seed 1234 \
    --batch_size 32 \
    --lr 1e-4 \
    --train_annotation data/iemocap/fold_2/iemocap_train_fold_2.json \
    --valid_annotation data/iemocap/fold_2/iemocap_valid_fold_2.json \
    --test_annotation data/iemocap/fold_2/iemocap_test_fold_2.json  \
    --number_of_epochs 100 \
    --feat_dir /home/cbolanos/experiments/features_whisper \
    --label_map data/iemocap/label_map.json \
    --device cuda \
    --out_n_neurons 4 \
    --hidden_size 128 \