export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/which-2-model"
#python mixmodels/concat-dense-moe-model-level-hybrid.py \
#  --expert1_dir="meta-llama/Llama-2-7b-chat-hf" \
#  --expert2_dir="lmsys/vicuna-7b-v1.5" \
#  --dense_dir="$root_dir/merged_model" \
#  --moe_dir="$root_dir/mix-moe" \
#  --num_dense_layers=16 \
#  --out_path="$root_dir/hybrid-moe-dense-16" \
#  --out_dtype="bfloat16"
python mixmodels/concat-dense-moe-model-level-hybrid.py \
  --expert1_dir="meta-llama/Llama-2-7b-chat-hf" \
  --expert2_dir="lmsys/vicuna-7b-v1.5" \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid-moe-dense-0" \
  --out_dtype="bfloat16"