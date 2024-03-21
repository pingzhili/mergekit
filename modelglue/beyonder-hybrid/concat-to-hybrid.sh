export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/beyonder-hybrid"
python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=8 \
  --out_path="$root_dir/hybrid-moe-dense-8" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid-moe-dense-16" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=24 \
  --out_path="$root_dir/hybrid-moe-dense-24" \
  --out_dtype="bfloat16"