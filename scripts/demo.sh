#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh
cd ${code_dir}

# Download data
nohup python common/downodps.py --start_date "20250825" --end_date "20250826"> "logs/downodps_${MODEL_TASK}.log" 2>&1 &
# Test dataset
python dataset/dataset_cvr.py
# Generate body.json
python generate_body.py
