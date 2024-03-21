export PYTHONPATH="./mixmodels:$PYTHONPATH"

python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="checkpoints/beyonder-hybrid/base-model-from-dare_ties" \
  --moe_dir="checkpoints/beyonder-hybrid/mix-moe" \
  --num_dense_layers=8 \
  --out_path="checkpoints/beyonder-hybrid/hybrid-moe-dense-8" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="checkpoints/beyonder-hybrid/base-model-from-dare_ties" \
  --moe_dir="checkpoints/beyonder-hybrid/mix-moe" \
  --num_dense_layers=16 \
  --out_path="checkpoints/beyonder-hybrid/hybrid-moe-dense-16" \
  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="checkpoints/beyonder-hybrid/base-model-from-dare_ties" \
  --moe_dir="checkpoints/beyonder-hybrid/mix-moe" \
  --num_dense_layers=24 \
  --out_path="checkpoints/beyonder-hybrid/hybrid-moe-dense-24" \
  --out_dtype="bfloat16"