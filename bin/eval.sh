#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

TF_CONFIG='{}'

time_str=$2
ckpt_time=$(date -d "${time_str:0:8} ${time_str:8:2} - ${infer_delta} hours" +"%Y%m%d%H%M")

echo "---------------------------------main-eval-------------------------------------"
msg=$(grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${time_str}.eval|awk -F': ' '{print $2}')
if [ "${msg}" == "" ]; then
  python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/${ckpt_time}" --mode eval --time_str ${time_str} --data_path ${data_path} > ${model_dir}/logs/${time_str}.eval 2>&1
  rm -rf ${model_dir}/${ckpt_time}/eval
  msg=$(grep 'INFO:tensorflow:Saving dict for global step' ${model_dir}/logs/${time_str}.eval|awk -F': ' '{print $2}')
  echo "task = ${MODEL_TASK}, time = ${time_str}, ${msg}" >> ${model_dir}/logs/eval
fi
#
#metric_data_str=$(tail -n 1 ${model_dir}/logs/eval | awk -F', ' '{
#    for (i = 1; i <= NF; i++) {
#        if ($i ~ /^task = /) task = $i;
#        if ($i ~ /^time = /) time = $i;
#        if ($i ~ /^loss = /) loss = $i;
#        if ($i ~ /task1_ctcvr\/auc = /) auc = $i;
#        if ($i ~ /task1_ctcvr\/pcoc = /) pcoc = $i;
#        if ($i ~ /task1_ctr\/auc = /) auc = $i;
#        if ($i ~ /task1_ctr\/pcoc = /) pcoc = $i;
#    }
#    print task ", " time ", " loss ", " auc ", " pcoc;
#}')
##需要针对ocpc模型，输出会按ctr、cvr两塔的指标，把cvr塔指标切割出来写入模型指标表中，供dataworks的模型训练指标监控的依赖
#echo "---------------------------------save_eval_metric------------------------------------"
#python3 ${code_dir}/common/save_eval_metric.py "${metric_data_str}"


if [ "${msg}" != "" ]; then
  echo "---------------------------------save_eval_metric------------------------------------"
  #echo "task = ${MODEL_TASK}, time = ${time_str}, ${msg}" >> ${model_dir}/logs/eval
  python3 ${code_dir}/common/save_eval_metric.py --eval_path "${model_dir}/logs/eval" --model_version "${MODEL_TASK}" --dm_date "${time_str}"
else
  echo "${model_dir}/logs/${time_str}.eval not metrice date, please check code!"
fi
