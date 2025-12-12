#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

TF_CONFIG='{}'

time_str=""
ckpt_time=""
ckpt_dir="${model_dir}/${ckpt_time}"

echo "---------------------------------main-eval-------------------------------------"
python3 ${code_dir}/main.py --ckpt_dir "${ckpt_dir}" --mode eval --time_str ${time_str} --data_path ${data_path} > ${model_dir}/logs/${time_str}.eval 2>&1
rm -rf ${model_dir}/${ckpt_time}/eval

msg=$(grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${time_str}.eval|awk -F': ' '{print $2}')
echo "task = ${MODEL_TASK}, ckpt_dir=${ckpt_dir}, time = ${time_str}, ${msg}" >> ${model_dir}/logs/eval