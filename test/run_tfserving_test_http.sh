#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh
cd ${code_dir}

#tfserving
#本地服务器上启动模型服务命令：
docker run -t --rm -p 8501:8501 -v "${MODEL_ROOT}/${MODEL_TASK}/export_dir:/models/model" -e MODEL_NAME=model registry.cn-beijing.aliyuncs.com/yoyohulian/tfra:serving &
sleep 10
echo "$(pwd)"
python ${code_dir}/test/tf_serving_test_http.py
