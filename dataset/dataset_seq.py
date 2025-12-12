# -*- coding: utf-8 -*-
# @Time : 2025/8/29 15:56
# @Author : huangmian
# @File : dataset.py
import os
import time
import logging
import tensorflow.compat.v1 as tf
try:
    from common.utils import *
    logging.info('--------------common.utils---------------')
except:
    from utils import *
    logging.info('--------------dataset.utils---------------')


logger = tf.compat.v1.logging
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logging.info(f">>>------ select_index.len={len(select_index)}, dense_feature_num={dense_feature_num}-----")
def mapper(x):
    lst = tf.split(x, [1] * feat_len)
    dataset_row = [tf.squeeze(x) for x in lst]
    dataset_row[features_index] = tf.strings.split(dataset_row[features_index], features_sep)
    dataset_row[features_index] = tf.gather(dataset_row[features_index], select_index, axis=0)
    for idx, col in enumerate(train_config.data_schema):
        if col in label_schema:
            dataset_row[idx] = tf.strings.to_number(dataset_row[idx], tf.float32)
    if seq_features_index:
        dataset_row[seq_features_index] = tf.strings.split(dataset_row[seq_features_index], features_sep)
        #dataset_row[seq_features_index] = dict(zip(seq_feat_names, tf.split(dataset_row[seq_features_index],
        #                                                                    seq_length_list)
        #                                           ))
        dataset_row[seq_features_index] = tf.concat(tf.split(dataset_row[seq_features_index], seq_length_list), axis=0)
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


