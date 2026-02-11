#!/usr/bin/env bash
is_train=1
is_eval=1
is_export=0
is_infer=0
is_downodps=0
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}
job="chief"
id=0
return_cnt=0
export TF_CONFIG="$(python3 ${code_dir}/tfconfig.py $job $id)"
task_idx=$(python3 -c "print($id + 1) if \"$job\" == \"worker\" else print($id)")
donefile=${model_dir}/logs/donefile.${task_idx}
touch ${donefile}
############################################################
today=$(date -d "- 1 days" +"%Y%m%d")
time_str=$(date -d "${start_date:0:8} ${start_date:8:2} - ${delta} hours" +"%Y%m%d%H%M")
end_date=$(date -d "${end_date:0:8} ${end_date:8:2} - ${delta} hours" +"%Y%m%d%H%M")
while true; do
  time_str=$(date -d "${time_str:0:8} ${time_str:8:2} + ${delta} hours" +"%Y%m%d%H%M")
  end_date=$(date -d "${end_date:0:8} ${end_date:8:2} + ${delta} hours" +"%Y%m%d%H%M")
  echo "--> time_str: $time_str, end_date: $end_date"
  #调用数据下载
  #if [ "${time_str:0:8}" == "${end_date:0:8}" ]; then
  #  if [ -f "${DATA_ROOT}/${MODEL_TASK}/${time_str:0:8}/_SUCCESS" ]; then
  #    is_downodps=0
  #  fi
  #fi
  if [ ${is_downodps} -ne 0 ]; then
    wait_for_odps_process ${code_dir}
    if [ $? -ne 0 ]; then
      exit 1
    fi
    python ${code_dir}/common/downodps.py \
           --start_date "${time_str:0:8}" \
           --end_date "${end_date:0:8}" \
           --save_path "${DATA_ROOT}" \
           --max_workers 16 \
           --task "${MODEL_TASK}" \
           --is_batch ${IS_BATCH} \
           --return_cnt ${return_cnt} > ${model_dir}/logs/downodps_${end_date}.log 2>&1
    if [ $? -ne 0 ]; then
      echo "downodps error,exit"
      exit 1
    fi
  fi
  ###
  if [ ${is_train} -eq 1 ]; then
    if [ $(grep ${end_date:0:8} ${donefile} |wc -l |awk '{print $1}') -ge 1 ];then continue; fi
    echo "---------------------------------main-train-------------------------------------"
    # 批量训练，清理ckpt/
    if [ ${IS_BATCH} -ne 0 ]; then
      echo "---批量训练，清理${model_dir}/ckpt/*---"
      rm -f ${model_dir}/ckpt/*
    fi
    echo "---------------------------------main-train-------------------------------------" >> ${model_dir}/logs/main_${end_date}.log
    python3 ${code_dir}/main.py --job_name ${job}.${id} \
            --ckpt_dir ${model_dir}/ckpt \
            --time_str ${time_str} \
            --end_time_str ${end_date} \
            --data_path ${data_path} > ${model_dir}/logs/main_${end_date}.log 2>&1
    if [ $? -ne 0 ]; then
      # 由于下载问题导致模型解析dataset异常，就重新下载一次数据重新训练
      if grep -qE "UnicodeDecodeError|make_one_shot_iterator" "${model_dir}/logs/main_${end_date}.log"; then
        python ${code_dir}/common/clear_history_data.py --data_path ${DATA_ROOT} --curr_date ${time_str:0:8}
        return_cnt=$(($return_cnt+1))
        if [ ${return_cnt} -le 3 ];then
          echo "[${time_str:0:8}, ${end_date:0:8}] dataset解析异常,数据重新下载和训练,return_cnt=${return_cnt}"
          time_str=$(date -d "${time_str:0:8} ${time_str:8:2} - ${delta} hours" +"%Y%m%d%H%M")
          end_date=$(date -d "${end_date:0:8} ${end_date:8:2} - ${delta} hours" +"%Y%m%d%H%M")
          continue
        fi
      fi
      echo "model train error,exit"
      exit 1
    fi
    return_cnt=0
    if [ "$(tail -1 ${donefile} |awk '{print $1}')" != "${end_date}" ];then
      time_str=$(date -d "${time_str:0:8} ${time_str:8:2} - ${delta} hours" +"%Y%m%d%H%M")
      end_date=$(date -d "${end_date:0:8} ${end_date:8:2} - ${delta} hours" +"%Y%m%d%H%M")
      sleep 1
      continue
    fi
  fi
  ############################################################
  #backup and export serving model
  if [ ${end_date:0:8} -ge ${backup} -a "${job}" == "chief" -a ${id} -eq 0 ]; then
    if [ ${is_export} -eq 1 ]; then
      echo "---------------------------------export.sh-------------------------------------"
      cd ${model_dir}
      rm -rf ckpt/events.out.tfevents* ckpt/eval && cp -r ckpt ${end_date}
      cd ${code_dir}/bin && bash export.sh ${MODEL_TASK} ${end_date} && cd -
      if [ $? -ne 0 ]; then
        echo "export error, exit"
        exit 1
      fi
    fi
    if [ ${is_eval} -eq 1 ]; then
      echo "eval data:${end_date}"
      echo "---------------------------------eval.sh-------------------------------------"
      cd ${code_dir}/bin && bash eval.sh ${MODEL_TASK} ${end_date} && cd -
      if [ $? -ne 0 ]; then
        echo "eval error, exit"
        exit 1
      fi
    fi
    if [ ${is_infer} -eq 1 ]; then
      echo "infer data:${end_date}"
      echo "---------------------------------infer.sh-------------------------------------"
      cd ${code_dir}/bin && bash infer.sh ${MODEL_TASK} ${end_date} && cd -
      if [ $? -ne 0 ]; then
        echo "infer error, exit"
        exit 1
      fi
    fi
  fi
  echo "----------------------------clear history data--------------------------------"
  cd ${code_dir}/bin && bash clear.sh ${MODEL_TASK} ${end_date} && cd -
  # Optional: stop window after finishing one day's work.
  # Enable by setting STOP_AFTER_HM (e.g. 0230). Optional STOP_RESUME_HM (default 1000).
  if [[ -n "${STOP_AFTER_HM}" ]]; then
    now_hm=$(date +%H%M)
    stop_start=${STOP_AFTER_HM}
    stop_end=${STOP_RESUME_HM:-1000}
    if ((10#${now_hm} >= 10#${stop_start} && 10#${now_hm} < 10#${stop_end})); then
      echo "Time window closed (${stop_start}-${stop_end}). Stop after finishing ${end_date:0:8}."
      break
    fi
  fi
  if [[ "${end_date:0:8}" == "${today}" ]]; then
    break
  fi
  #if [ "${end_date:0:8}" == "20251101" ]; then
  #  break
  #fi
  #if [ "${end_date:0:8}" == "20251101" ]; then
  #  is_eval=1
  #  is_infer=1
  #fi
  #break
done
