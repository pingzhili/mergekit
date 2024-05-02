export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/which-4"
python mixmodels/concat-dense-moe-hybrid.py \
  --dense_dir="$root_dir/linear_merged_model" \
  --moe_dir="$root_dir/ffn_mixed_moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid_ffn_lv_mixed_moe" \
  --out_dtype="bfloat16"
python mixmodels/concat-dense-moe-block-level.py \
  --expert1_dir="migtissera/Synthia-7B-v1.2" \
  --expert2_dir="neuralmagic/Llama-2-7b-evolcodealpaca" \
  --expert3_dir="PygmalionAI/pygmalion-2-7b" \
  --expert4_dir="meta-math/MetaMath-7B-V1.0" \
  --dense_dir="$root_dir/linear_merged_model" \
  --moe_dir="$root_dir/ffn_mixed_moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid_block_lv_mixed_moe" \
  --out_dtype="bfloat16"
python mixmodels/concat-dense-moe-model-level-hybrid.py \
  --expert1_dir="migtissera/Synthia-7B-v1.2" \
  --expert2_dir="neuralmagic/Llama-2-7b-evolcodealpaca" \
  --expert3_dir="PygmalionAI/pygmalion-2-7b" \
  --expert4_dir="meta-math/MetaMath-7B-V1.0" \
  --dense_dir="$root_dir/linear_merged_model" \
  --moe_dir="$root_dir/ffn_mixed_moe" \
  --num_dense_layers=16 \
  --out_path="$root_dir/hybrid_model_lv_mixed_moe" \
  --out_dtype="bfloat16"