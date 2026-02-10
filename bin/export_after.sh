#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

curr_export_timestamp="$(cat ${code_dir}/bin/logs/${MODEL_TASK}/export_timestamp)"
echo "curr_export_timestamp=${curr_export_timestamp}"
# Record loop start time (seconds timestamp)
start_time=$(date +%s)
# Seconds in 20 hours (20*3600)
timeout_hour=20
timeout_seconds=$((${timeout_hour} * 3600))
#
while [ 1 ]; do
    # Get current timestamp
    current_time=$(date +%s)
    # Compute elapsed time
    elapsed_time=$((current_time - start_time))
    # Check timeout
    if [ $elapsed_time -ge $timeout_seconds ]; then
        echo "Not completed within ${timeout_hour} hours, exiting script"
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
