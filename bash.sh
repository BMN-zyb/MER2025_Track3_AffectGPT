












CUDA_VISIBLE_DEVICES=1 python -u inference_hybird.py --zeroshot --dataset='MER2025OV' --outside_user_message="please consider the following points: observe facial expressions and body movements, evaluate speech rate, tone, and volume, and analyze what the character says. Finally, infer the person's emotional state and provide your reasoning process." --cfg-path=train_configs/msa_outputhybird_bestsetup_bestfusion_frame_lz.yaml --options "inference.test_epoch=5" "inference.base_root=output/results-description"
python ovlabel_extraction.py