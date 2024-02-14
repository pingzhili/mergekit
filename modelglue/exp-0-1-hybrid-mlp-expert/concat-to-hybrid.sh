python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="checkpoints/exp-0-1-hybrid-mlp-expert/base-model-from-dare_ties" \
  --moe_dir="checkpoints/exp-0-1-hybrid-mlp-expert/mix-moe" \
  --num_dense_layers=16 \
  --out_path="checkpoints/exp-0-1-hybrid-mlp-expert/hybrid-moe-dense-16" \
  --out_dtype="bfloat16"