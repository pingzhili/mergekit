export MODEL="checkpoints/exp-0-3-hybrid-mlp-expert/mix-moe/"
export CUDA_VISIBLE_DEVICES=0,1,2,3



#accelerate launch --main_process_port 21022 ../bigcode-evaluation-harness/main.py \
#      --model $MODEL \
#      --tasks humaneval \
#      --max_length_generation 512 \
#      --do_sample True \
#      --n_samples 1 \
#      --top_p 0.95 \
#      --batch_size 1 \
#      --temperature 0.2 \
#      --trust_remote_code \
#      --precision bf16 \
#      --allow_code_execution \
#      --use_auth_token \
#      --save_generations \
#      --save_generations_path humaneval_generations_model_moe.json \
#      --metric_output_path humaneval_metric_output_model_moe.json

accelerate launch --main_process_port 21022 ../bigcode-evaluation-harness/main.py \
      --model $MODEL \
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
      --save_generations_path mbpp_generations_model_moe.json \
      --metric_output_path mbpp_metric_output_model_moe.json


accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks gsm8k \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_gsm8k_model_moe.json >> output_model_moe.out


accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks arc_challenge \
      --num_fewshot 25 \
      --batch_size 1 \
      --output_path ./25shot_arc_challenge_model_moe.json >> output_model_moe.out

accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks truthfulqa \
      --num_fewshot 0 \
      --batch_size 1 \
      --output_path ./0shot_truthfulqa_model_moe.json >> output_model_moe.out

accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks winogrande \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_winogrande_model_moe.json >> output_model_moe.out

accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks hellaswag \
      --num_fewshot 10 \
      --batch_size 1 \
      --output_path ./10shot_hellaswag_model_moe.json >> output_model_moe.out

accelerate launch --main_process_port 21001  -m lm_eval --model hf \
      --model_args pretrained=$MODEL,trust_remote_code=True \
      --tasks mmlu \
      --num_fewshot 5 \
      --batch_size 1 \
      --output_path ./5shot_mmlu_model_moe.json >> output_model_moe.out