models=("checkpoints/exp-0-3-hybrid-mlp-expert/hybrid-moe-dense-8" "checkpoints/exp-0-3-hybrid-mlp-expert/hybrid-moe-dense-16" "checkpoints/exp-0-3-hybrid-mlp-expert/hybrid-moe-dense-24")
i=0
for model in "${models[@]}"; do
    ((i++))
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    accelerate launch --main_process_port 21022 ../bigcode-evaluation-harness/main.py \
      --model $model \
      --tasks humaneval \
      --max_length_generation 512 \
      --do_sample True \
      --n_samples 1 \
      --top_p 0.95 \
      --batch_size 1 \
      --temperature 0.2 \
      --precision bf16 \
      --allow_code_execution \
      --use_auth_token \
      --save_generations \
      --save_generations_path humaneval_generations_model_idx$i.json \
      --metric_output_path humaneval_metric_output_model_idx$i.json

    accelerate launch --main_process_port 21022 ../bigcode-evaluation-harness/main.py \
      --model $model \
      --tasks mbpp \
      --max_length_generation 512 \
      --do_sample True \
      --n_samples 1 \
      --top_p 0.95 \
      --batch_size 1 \
      --temperature 0.2 \
      --precision bf16 \
      --allow_code_execution \
      --use_auth_token \
      --save_generations \
      --save_generations_path mbpp_generations_model_idx$i.json \
      --metric_output_path mbpp_metric_output_model_idx$i.json
done
