#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

curr_export_timestamp="$(cat ${code_dir}/bin/logs/${MODEL_TASK}/export_timestamp)"
echo "curr_export_timestamp=${curr_export_timestamp}"
# 记录循环开始时间（秒级时间戳）
start_time=$(date +%s)
# 20小时的秒数（20*3600）
timeout_hour=20
timeout_seconds=$((${timeout_hour} * 3600))
#
while [ 1 ]; do
    # 获取当前时间戳
    current_time=$(date +%s)
    # 计算已运行时间
    elapsed_time=$((current_time - start_time))
    # 检查是否超时
    if [ $elapsed_time -ge $timeout_seconds ]; then
        echo "${timeout_hour}小时内未完成，脚本退出"
        exit 1
    fi
    #
    export_timestamp=$(find "${model_dir}/export_dir" -maxdepth 1 -type d ! -name "."|sort -nr|head -n 1)
    echo "export_timestamp=${export_timestamp}"
    if [ "${curr_export_timestamp}" != "${export_timestamp}" ]; then
      echo "${export_timestamp}" > ${code_dir}/bin/logs/${MODEL_TASK}/export_timestamp
      break
    else
      sleep 180
    fi
done


echo "---------------------------------upload-------------------------------------"
python3 ${code_dir}/common/upload.py
echo "---export_after ok---"
