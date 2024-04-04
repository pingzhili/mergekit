models=("modelglue/which-2-model/hybrid-moe-dense-0")
i=0
for model in "${models[@]}"; do
    ((i++))
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks arc_challenge \
      --num_fewshot 25 \
      --batch_size 1 \
      --output_path ./25shot_arc_challenge_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks truthfulqa \
      --num_fewshot 0 \
      --batch_size 1 \
      --output_path ./0shot_truthfulqa_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks winogrande \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_winogrande_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks hellaswag \
      --num_fewshot 10 \
      --batch_size 1 \
      --output_path ./10shot_hellaswag_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks mmlu \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_mmlu_model_idx_$i.json >> output_model_idx_$i.out
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001 ../bigcode-evaluation-harness/main.py \
      --model $model \
      --tasks humaneval \
      --max_length_generation 512 \
      --do_sample True \
      --n_samples 1 \
      --top_p 0.95 \
      --batch_size 1 \
      --temperature 0.2 \
      --trust_remote_code \
      --precision bf16 \
      --allow_code_execution \
      --use_auth_token \
      --save_generations \
      --save_generations_path humaneval_generations_model_idx$i.json \
      --metric_output_path humaneval_metric_output_model_idx$i.json
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001 ../bigcode-evaluation-harness/main.py \
      --model $model \
      --tasks mbpp \
      --max_length_generation 512 \
      --do_sample True \
      --n_samples 1 \
      --top_p 0.95 \
      --batch_size 1 \
      --temperature 0.2 \
      --trust_remote_code \
      --precision bf16 \
      --allow_code_execution \
      --use_auth_token \
      --save_generations \
      --save_generations_path mbpp_generations_model_idx$i.json \
      --metric_output_path mbpp_metric_output_model_idx$i.json
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$model,trust_remote_code=True,use_cache=False \
      --tasks gsm8k \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_gsm8k_model_idx_$i.json >> output_model_idx_$i.out
done
