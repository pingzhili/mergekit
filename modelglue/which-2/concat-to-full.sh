export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/which-2"
python mixmodels/concat-dense-moe-ffn-level-hybrid.py \
  --dense_dir="$root_dir/linear_merged_model" \
  --moe_dir="$root_dir/ffn_mixed_moe" \
  --num_dense_layers=0 \
  --out_path="$root_dir/full_ffn_lv_mixed_moe" \
  --out_dtype="bfloat16"
#python mixmodels/concat-dense-moe-block-level-hybrid.py \
#  --expert1_dir="migtissera/Synthia-7B-v1.2" \
#  --expert2_dir="neuralmagic/Llama-2-7b-evolcodealpaca" \
#  --expert3_dir="PygmalionAI/pygmalion-2-7b" \
#  --expert4_dir="meta-math/MetaMath-7B-V1.0" \
#  --dense_dir="$root_dir/linear_merged_model" \
#  --moe_dir="$root_dir/ffn_mixed_moe" \
#  --num_dense_layers=0 \
#  --out_path="$root_dir/full_block_lv_mixed_moe" \
#  --out_dtype="bfloat16"
#python mixmodels/concat-dense-moe-model-level-hybrid.py \
#  --expert1_dir="migtissera/Synthia-7B-v1.2" \
#  --expert2_dir="neuralmagic/Llama-2-7b-evolcodealpaca" \
#  --expert3_dir="PygmalionAI/pygmalion-2-7b" \
#  --expert4_dir="meta-math/MetaMath-7B-V1.0" \
#  --dense_dir="$root_dir/linear_merged_model" \
#  --moe_dir="$root_dir/ffn_mixed_moe" \
#  --num_dense_layers=0 \
#  --out_path="$root_dir/full_model_lv_mixed_moe" \
#  --out_dtype="bfloat16"