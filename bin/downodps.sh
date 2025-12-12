#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

time_str=${start_date:0:8}

echo "--> time_str: $time_str, end_date: $end_date"
python ${code_dir}/common/downodps.py \
           --start_date "${time_str:0:8}" \
           --end_date "${end_date:0:8}" \
           --save_path "${DATA_ROOT}" \
           --max_workers 16 \
           --task "${MODEL_TASK}" \
           --is_batch ${IS_BATCH} > ${model_dir}/logs/downodps_${end_date}.log 2>&1
