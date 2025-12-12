#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}
#加载conf.sh配置文件,获取model_dir、export_dir、binning_table_name等配置信息

TF_CONFIG='{}'
time_str=$(awk '{print $1}' ${model_dir}/logs/donefile.0|tail -1)
echo "---------------------------------main-export-------------------------------------"
export CUDA_VISIBLE_DEVICES='-1'
python3 ${code_dir}/main.py --ckpt_dir "${model_dir}/${time_str}" --export_dir ${model_dir}/export_dir --mode export --time_str ${time_str}
if [ $? -ne 0 ]; then
  echo "export error, exit"
  exit 1
fi
day=${time_str:0:8}
#生成body.json文件,取分桶后表某一天数据在generate_body.py代码中 通过schema.conf与features字段对齐，features字段已经按照adx_dmp.ctr_cvr_selected_fea_cols_conf的selected_fea_cols对齐
echo "-------------------------------generate_body---------------------------------------"
python3 ${code_dir}/common/generate_body.py --day "${day}"
echo "==================warmup.py:  export_dir:${export_dir} ===================="
#使用binning_table_name分桶且特征选择数据生成warmup文件
python ${code_dir}/common/warmup.py --model_dir="${export_dir}"
exit 0


#############迁移到export_after.sh脚本中执行(解决warmip脚本中在调用sklearn包异常问题)#############
echo "==================warmup.py:  export_dir:${export_dir} ===================="
. ${code_dir}/bin/conf.sh && cd ${model_dir}
#使用binning_table_name分桶且特征选择数据生成warmup文件
for ((i=1;i<=4;i++)); do
  echo "========warmup.py run ${i}==========="
  python ${code_dir}/common/warmup.py --model_dir="${export_dir}"
  if [ $? -ne 0 ]; then
    conda activate test
    sleep 300
    continue
  else
    break
  fi
done

day=${time_str:0:8}
#生成body.json文件,取分桶后表某一天数据在generate_body.py代码中 通过schema.conf与features字段对齐，features字段已经按照adx_dmp.ctr_cvr_selected_fea_cols_conf的selected_fea_cols对齐
echo "-------------------------------generate_body---------------------------------------"
python3 ${code_dir}/common/generate_body.py --day "${day}"
echo "---------------------------------upload-------------------------------------"
python3 ${code_dir}/common/upload.py

