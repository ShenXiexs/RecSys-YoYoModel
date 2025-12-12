#!/usr/bin/env python
# coding=utf-8
import os
import time
import tensorflow.compat.v1 as tf
# try:
#     from common.utils import select_feature, select_index
# except:
#     from utils import select_feature, select_index
try:
    from common.utils import *
except:
    from utils import *

logger = tf.compat.v1.logging
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(dirname)

def mapper(x):
    lst = tf.split(x, [1] * 7)
    user_id, requestid, combination_un_id, adslot_id_type, is_click, is_conversion, features = [tf.squeeze(x) for x in lst]
    features = tf.strings.split(features, "\002")
    features = tf.gather(features, select_index, axis=0)
    return tf.strings.to_number(is_click, tf.float32),tf.strings.to_number(is_conversion, tf.float32), features,user_id,requestid,combination_un_id

def input_fn(filenames, model_dir, task_number=1, task_idx=0, shuffle=True, epochs=1, batch_size=1024):
    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if task_number > 1:
        dataset = dataset.shard(task_number, task_idx)
    #
    dataset = tf.data.TextLineDataset(dataset, compression_type='GZIP')
    # ['user_id','requestid','combination_un_id','adslot_id_type','is_click','is_conversion','features']
    dataset = dataset.map(lambda x: tf.strings.split(x, "\t"))\
                     .filter(lambda x: tf.math.equal(tf.shape(x)[0], 7))
    #返回dataset列为is_click,is_conversion,features,并验证features的字段个数是和slot的字段个数对齐，不对齐就过滤
    dataset = dataset.map(mapper).filter(lambda *x: tf.math.equal(tf.shape(x[2])[0],
                                                                                         len(select_feature)))
    #
    dataset = dataset.repeat(epochs).prefetch(batch_size * 100)
    # Randomizes input using a window of 256 elements (read into memory)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 20)
    # epochs from blending together.
    elements = tf.compat.v1.data.make_one_shot_iterator(dataset.batch(batch_size)).get_next()
    #
    features = dict(zip(["ctr_label", "ctcvr_label", "features","user_id","requestid","combination_un_id"], elements))
    return features


if __name__ == "__main__":
    # tf.disable_v2_behavior()
    t0 = time.time()
    import glob
    from common.utils import train_config
    data_root = os.environ.get("DATA_ROOT", "/data/share/opt/data")
    dt = max(os.listdir(data_root))
    filenames = glob.glob(f"{data_root}/{dt}/part*.gz")
    features = input_fn(filenames, "")
    print(features,'-------')
    batch_size = tf.shape(features["features"])[0]
    ctr_labels = []
    if 'ctr_label' in features:
        print('----------ctr_label')
        ctr_labels.append(tf.cast(features['ctr_label'], tf.float32))
    else:
        ctr_labels.append(tf.zeros((batch_size,)))

    ctcvr_labels = []
    if 'ctcvr_label' in features:
        print('----------ctcvr_label')
        ctcvr_labels.append(tf.cast(features['ctcvr_label'], tf.float32))
    else:
        ctcvr_labels.append(tf.zeros((batch_size,)))

    print("ctr_labels---",ctr_labels)
    print("ctcvr_labels----",ctcvr_labels)
    print("features:---",features['features'])