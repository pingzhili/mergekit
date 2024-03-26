#models=("modelglue/hybrid-moe-dense-0" "modelglue/beyonder-block-hybrid/hybrid-moe-dense-8" "modelglue/beyonder-block-hybrid/hybrid-moe-dense-16" "modelglue/beyonder-block-hybrid/hybrid-moe-dense-24")
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 21001  -m lm_eval --model hf \
    --model_args pretrained="modelglue/beyonder-block-hybrid/hybrid-moe-dense-0",trust_remote_code=True,use_cache=False \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path ./5shot_gsm8k_model.json >> output_model.out