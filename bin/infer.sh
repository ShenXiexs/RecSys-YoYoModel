#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}
TF_CONFIG='{}'

time_str=$2
ckpt=$(date -d "${time_str:0:8} ${time_str:8:2} - ${infer_delta} hours" +"%Y%m%d%H%M")
echo "---------------------------------main-infer-------------------------------------"
echo "ckpt_dir=${model_dir}/${ckpt}, time_str=${time_str}"
python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/${ckpt}" --mode infer --time_str "${time_str}"

