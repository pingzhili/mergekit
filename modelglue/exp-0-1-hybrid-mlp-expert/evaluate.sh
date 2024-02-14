models=("checkpoints/exp-0-1-hybrid-mlp-expert/hybrid-moe-dense-8" "checkpoints/exp-0-1-hybrid-mlp-expert/hybrid-moe-dense-24" "checkpoints/exp-0-2-hybrid-mlp-expert/hybrid-moe-dense-16")
i=0
for model in "${models[@]}"; do
    ((i++))
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks gsm8k \
      --num_fewshot 5 \
      --batch_size auto:4 \
      --output_path ./5shot_gsm8k_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks arc_challenge \
      --num_fewshot 25 \
      --batch_size auto:4 \
      --output_path ./25shot_arc_challenge_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks truthfulqa \
      --num_fewshot 0 \
      --batch_size auto:4 \
      --output_path ./0shot_truthfulqa_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks winogrande \
      --num_fewshot 5 \
      --batch_size auto:4 \
      --output_path ./5shot_winogrande_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks hellaswag \
      --num_fewshot 10 \
      --batch_size auto:4 \
      --output_path ./10shot_hellaswag_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True \
      --tasks mmlu \
      --num_fewshot 5 \
      --batch_size auto:4 \
      --output_path ./5shot_mmlu_model_idx_$i.json >> output_model_idx_$i.out
done
