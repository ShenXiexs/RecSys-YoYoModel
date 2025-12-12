# -*- coding: utf-8 -*-
# @Time : 2025/8/29 15:56
# @Author : huangmian
# @File : dataset.py
import os
import time
import tensorflow.compat.v1 as tf
try:
    from common.utils import *
except:
    from utils import *


logger = tf.compat.v1.logging
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

feat_len = len(train_config.data_schema)
features_index = train_config.data_schema.index("features")
label_schema = getattr(train_config, "label_schema", {"is_click": "ctr_label", "is_conversion": "ctcvr_label"})
features_keys = [label_schema.get(col, col) for col in train_config.data_schema]
compression_type = getattr(train_config, "compression_type", "GZIP")
features_sep = getattr(train_config, "features_sep", "\002")
field_sep = getattr(train_config, "field_sep", "\t")

def mapper(x):
    lst = tf.split(x, [1] * feat_len)
    dataset_row = [tf.squeeze(x) for x in lst]
    dataset_row[features_index] = tf.strings.split(dataset_row[features_index], features_sep)
    dataset_row[features_index] = tf.gather(dataset_row[features_index], select_index, axis=0)
    for idx, col in enumerate(train_config.data_schema):
        if col in label_schema:
            dataset_row[idx] = tf.strings.to_number(dataset_row[idx], tf.float32)
    return dataset_row

def input_fn(filenames, model_dir, task_number=1, task_idx=0, shuffle=True, epochs=1, batch_size=1024):
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    logger.info(f"filenames={filenames[:5]}")
    logger.info((">>>", model_dir, task_number, task_idx, shuffle, epochs, batch_size))
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if task_number > 1:
        dataset = dataset.shard(task_number, task_idx)
    #
    dataset = tf.data.TextLineDataset(dataset, compression_type=compression_type)
    #
    global field_sep
    if "/O35_mutil_cvr/" in filenames[0] and filenames[0].split("/")[-2] <= '20251118':
        field_sep = '\t'
    if "/O35_mutil_cvr/" in filenames[0] and filenames[0].split("/")[-2] == '20251115':
        field_sep = "\003"
    dataset = dataset.map(lambda x: tf.strings.split(x, field_sep))\
                     .filter(lambda x: tf.math.equal(tf.shape(x)[0], feat_len))
    #验证features的字段个数是和slot的字段个数对齐，不对齐就过滤
    dataset = dataset.map(mapper).filter(lambda *x: tf.math.equal(tf.shape(x[features_index])[0], dense_feature_num))
    #
    dataset = dataset.repeat(epochs).prefetch(batch_size * 100)
    # Randomizes input using a window of 256 elements (read into memory)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 20)
    # epochs from blending together.
    elements = tf.compat.v1.data.make_one_shot_iterator(dataset.batch(batch_size)).get_next()
    #
    features = dict(zip(features_keys, elements))
    return features


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    t0 = time.time()
    import glob
    from common.utils import train_config
    data_root = os.environ.get("DATA_ROOT", "/data/share/opt/data") + f"/{train_config.model_version}"
    dt = max([i for i in os.listdir(data_root) if i.startswith("2")])
    filenames = glob.glob(f"{data_root}/{dt}/part*.gz")
    #print(f"filenames[:5]={filenames[:5]}")
    for filename in filenames:
        print(f">>>>>>>>>>>{filename}<<<<<<<<<<<<")
        features = input_fn([filename], "")
        print(f"{features=}")
        for key in features_keys:
            print(f"{key} --> {features[key]}")
