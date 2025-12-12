#!/usr/bin/env python
# coding=utf-8
import os, json
import tensorflow as tf
from tensorflow_serving.apis import prediction_log_pb2
import tensorflow_serving.apis.predict_pb2
import tensorflow_serving.apis.prediction_service_pb2_grpc
from absl import app, flags

try:
    from common.utils import train_config, schema_fea2idx_dict
except:
    from utils import train_config, schema_fea2idx_dict

flags.DEFINE_string('body_file', train_config.body_json_name, 'body.json文件路径')
flags.DEFINE_string('model_dir', '', '模型路径')
FLAGS = flags.FLAGS

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
basename = os.path.basename(dirname)


def parse_int(x):
    try:
        return int(x)
    except:
        return -1


def main(argv):
    # /opt/model/{basename}/export_dir
    model_dir = FLAGS.model_dir
    print(f'model_dir:{model_dir}')
    print(f"body_file:{FLAGS.body_file}")

    files = sorted(map(parse_int, os.listdir(model_dir)))
    print(f"files: {files}")
    # f"{dirname}/common/body.json"
    with open(FLAGS.body_file) as f:
        data = json.load(f)["inputs"]

    request = tensorflow_serving.apis.predict_pb2.PredictRequest()
    request.model_spec.name = 'model'  # 替换为你的模型名称
    request.model_spec.signature_name = 'serving_default'  # 使用默认签名函数
    for k in data:
        k_idx = schema_fea2idx_dict[k]
        request.inputs[k_idx].CopyFrom(tf.make_tensor_proto(data[k], tf.string))  # 将图片数据添加到输入张量中

    example = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))

    warmup_path = f"{model_dir}/{files[-1]}/assets.extra"
    os.makedirs(warmup_path, exist_ok=True)
    with tf.io.TFRecordWriter(f"{warmup_path}/tf_serving_warmup_requests") as writer:
        for _ in range(300):
            writer.write(example.SerializeToString())


if __name__ == "__main__":
    # 基于body.json生成的数据，生成tf_serving_warmup_requests文件到模型导出目录下
    app.run(main)

