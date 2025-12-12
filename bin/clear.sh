#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh

time_str=$2
# ckpt、log日志等数据保留最近14天
delete_date=$(date -d "${time_str:0:8} ${time_str:8:2} - 14days " +"%Y%m%d")
# 改成了train_config中del_days控制，默认保留30天
# data_del_date=$(date -d "${time_str:0:8} ${time_str:8:2} - 90days " +"%Y%m%d")
python ${code_dir}/common/clear_history_data.py --data_path ${DATA_ROOT} --curr_date "${time_str:0:8}"
#
rm -rf ${model_dir}/${delete_date}0000
rm -f ${model_dir}/logs/*${delete_date}*
rm -f ${code_dir}/bin/${MODEL_TASK}/nohup_*${delete_date}*.log

data_dir=$model_dir/export_dir
for item in "$data_dir"/*; do
    if [ -d "$item" ]; then  # 检查是否为文件
        model_ts="$(basename "$item")"  # 打印文件名
        if (( ${#model_ts}==13 ));then
            model_ts=$(( model_ts/1000  ))
        fi
        if (( ${#model_ts}==10  ));then
            current_timestamp=$(date +%s)
            diff_seconds=$((current_timestamp - model_ts))
            diff_days=$((diff_seconds /86400))
            if (( diff_days>10  ));then

                echo " $item diff days $diff_days "
                rm -rf $item
                #rm -f $item/saved_model.pb
                #rm -f $item/variables/variables*
                #rm -f $item/assets.extra/tf_serving_warmup_requests
            fi
        fi
    fi
done
