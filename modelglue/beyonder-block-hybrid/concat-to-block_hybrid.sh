export PYTHONPATH="./mixmodels:$PYTHONPATH"
root_dir="modelglue/beyonder-block-hybrid"
#python mixmodels/concat-dense-moe-block-level-hybrid.py \
#  --expert1_dir="openchat/openchat-3.5-1210" \
#  --expert2_dir="beowolx/CodeNinja-1.0-OpenChat-7B" \
#  --expert3_dir="maywell/PiVoT-0.1-Starling-LM-RP" \
#  --expert4_dir="WizardLM/WizardMath-7B-V1.1" \
#  --dense_dir="$root_dir/merged_model" \
#  --moe_dir="$root_dir/mix-moe" \
#  --num_dense_layers=24 \
#  --out_path="$root_dir/hybrid-moe-dense-24" \
#  --out_dtype="bfloat16"
#
#python mixmodels/concat-dense-moe-block-level-hybrid.py \
#  --expert1_dir="openchat/openchat-3.5-1210" \
#  --expert2_dir="beowolx/CodeNinja-1.0-OpenChat-7B" \
#  --expert3_dir="maywell/PiVoT-0.1-Starling-LM-RP" \
#  --expert4_dir="WizardLM/WizardMath-7B-V1.1" \
#  --dense_dir="$root_dir/merged_model" \
#  --moe_dir="$root_dir/mix-moe" \
#  --num_dense_layers=16 \
#  --out_path="$root_dir/hybrid-moe-dense-16" \
#  --out_dtype="bfloat16"
#
#python mixmodels/concat-dense-moe-block-level-hybrid.py \
#  --expert1_dir="openchat/openchat-3.5-1210" \
#  --expert2_dir="beowolx/CodeNinja-1.0-OpenChat-7B" \
#  --expert3_dir="maywell/PiVoT-0.1-Starling-LM-RP" \
#  --expert4_dir="WizardLM/WizardMath-7B-V1.1" \
#  --dense_dir="$root_dir/merged_model" \
#  --moe_dir="$root_dir/mix-moe" \
#  --num_dense_layers=8 \
#  --out_path="$root_dir/hybrid-moe-dense-8" \
#  --out_dtype="bfloat16"

python mixmodels/concat-dense-moe-block-level-hybrid.py \
  --expert1_dir="openchat/openchat-3.5-1210" \
  --expert2_dir="beowolx/CodeNinja-1.0-OpenChat-7B" \
  --expert3_dir="maywell/PiVoT-0.1-Starling-LM-RP" \
  --expert4_dir="WizardLM/WizardMath-7B-V1.1" \
  --dense_dir="$root_dir/merged_model" \
  --moe_dir="$root_dir/mix-moe" \
  --num_dense_layers=0 \
  --out_path="$root_dir/hybrid-moe-dense-0" \
  --out_dtype="bfloat16"
