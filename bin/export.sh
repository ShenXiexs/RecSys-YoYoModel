#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}
# Load conf.sh to get model_dir, export_dir, binning_table_name, etc.

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
# Generate body.json using one day of binned-table data in generate_body.py. It aligns with schema.conf
# and the features field has already been aligned with selected_fea_cols in adx_dmp.ctr_cvr_selected_fea_cols_conf.
echo "-------------------------------generate_body---------------------------------------"
python3 ${code_dir}/common/generate_body.py --day "${day}"
echo "==================warmup.py:  export_dir:${export_dir} ===================="
# Generate warmup file using binning_table_name and selected features
python ${code_dir}/common/warmup.py --model_dir="${export_dir}"
exit 0


############# Moved to export_after.sh (to fix sklearn import errors in warmup script) #############
echo "==================warmup.py:  export_dir:${export_dir} ===================="
. ${code_dir}/bin/conf.sh && cd ${model_dir}
# Generate warmup file using binning_table_name and selected features
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
# Generate body.json using one day of binned-table data in generate_body.py. It aligns with schema.conf
# and the features field has already been aligned with selected_fea_cols in adx_dmp.ctr_cvr_selected_fea_cols_conf.
echo "-------------------------------generate_body---------------------------------------"
python3 ${code_dir}/common/generate_body.py --day "${day}"
echo "---------------------------------upload-------------------------------------"
python3 ${code_dir}/common/upload.py
