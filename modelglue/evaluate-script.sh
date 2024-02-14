models=("checkpoints/beyonder_1" "checkpoints/beyonder_2" "checkpoints/beyonder_3" "checkpoints/beyonder_4" "checkpoints/beyonder_5")
i=0
for model in "${models[@]}"; do
    ((i++))
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval \
        --model hf \
        --model_args pretrained=$model,dtype="bfloat16" \
        --tasks mmlu \
        --num_fewshot 5 \
        --batch_size 1 \
        --output_path ./5shot_mmlu_model_idx_$i.json >> output_model_idx_$i.out
done
