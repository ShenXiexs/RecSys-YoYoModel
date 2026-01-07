#!/usr/bin/env bash
source /root/anaconda3/etc/profile.d/conda.sh
export PYTHONHASHSEED=107
###config
start_date=$2
end_date=$3
if [[ "${start_date}" == "" || "${start_date}" == "1" ]]; then
  #start_date=20250805
  start_date=$(date -d "- 1 days" +"%Y%m%d")
fi
if [[ "${end_date}" == "" || "${end_date}" == "1" ]]; then
  end_date=${start_date}
fi
if [[ "$4" == "" || "$4" == "0" ]]; then
  export IS_BATCH=0
else
  export IS_BATCH=1
  if [ "${start_date}" == "${end_date}" ]; then
    days=$(($4-1))
    start_date=$(date -d "${end_date:0:8} ${end_date:8:2} - ${days} days" +"%Y%m%d%H%M")
  fi
fi
backup=20250211
delta=24
infer_delta=24
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
export PYTHONPATH=${code_dir}
echo "code_dir=${code_dir}"
if [ "${MODEL_TASK}" == "" ]; then
  export MODEL_TASK=${1}
  if [ "${MODEL_TASK}" == "" ]; then
    export MODEL_TASK=$(basename ${code_dir})
  fi
fi
task=${MODEL_TASK}
if [ "${task}" == "" ]; then
  task=$(basename ${code_dir})
fi
#定义模型导出和模型拉取数据、模型特征表的模型分支数据
export MODEL_ROOT="/data/share/opt/model"
export DATA_ROOT="/data/share/opt/data"
model_dir="${MODEL_ROOT}/${task}"
export_dir="${model_dir}/export_dir"
data_path="${DATA_ROOT}/${task}"
#
mkdir -p ${model_dir}/logs > /dev/null
#
export TRAIN_CONFIG=config/${task}/train_config.py
echo "TRAIN_CONFIG=${TRAIN_CONFIG}"
conda activate env_gpu
#conda activate test
###
