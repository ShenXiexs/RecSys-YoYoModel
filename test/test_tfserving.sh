#本地服务器上启动模型服务命令：
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh
cd ${code_dir}

#docker run -t --rm -p 8501:8501 -v "${MODEL_ROOT}/${MODEL_TASK}/export_dir:/models/model" -e MODEL_NAME=model registry.cn-beijing.aliyuncs.com/yoyohulian/tfra:serving &
#sleep 10

python ${code_dir}/test/tf_serving_test_http.py

#docker ps
#docker kill 367ded3e0a56

#bash test_tfserving.sh O31_v3
#{'outputs': [[0.700230479, 0.695258856, 0.0492721796], [0.61874634, 0.610923886, 0.0790759623]]}

#bash test_tfserving.sh O31_v6
#{'outputs': [[0.883045435, 0.87458849, 0.0783210099], [0.842889965, 0.834553182, 0.0851900578]]}
