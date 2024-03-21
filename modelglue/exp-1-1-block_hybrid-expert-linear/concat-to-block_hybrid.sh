export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/exp-1-1-block_hybrid-expert-linear"
python mixmodels/concat-dense-moe-block-level.py \
  --expert1_dir="meta-llama/Llama-2-7b-chat-hf" \
  --expert2_dir="lmsys/vicuna-7b-v1.5" \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=24 \
  --out_path="$root_dir/hybrid-moe-dense-24" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-block-level.py \
  --expert1_dir="meta-llama/Llama-2-7b-chat-hf" \
  --expert2_dir="lmsys/vicuna-7b-v1.5" \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid-moe-dense-16" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-block-level.py \
  --expert1_dir="meta-llama/Llama-2-7b-chat-hf" \
  --expert2_dir="lmsys/vicuna-7b-v1.5" \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=8 \
  --out_path="$root_dir/hybrid-moe-dense-8" \
  --out_dtype="bfloat16"
