#!/usr/bin/env python
# coding=utf-8
import os, json, sys, re
import datetime
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from collections import defaultdict
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
tf_config = json.loads(os.environ.get("TF_CONFIG") or '{}')
tc_module, _, _ = os.environ.get("TRAIN_CONFIG").replace("/", ".").rpartition('.')
train_config = getattr(import_module(tc_module), "TrainConfig")

with open(train_config.schema_path) as f:
    schema = [l.strip("\n") for l in f if not (l.startswith("#") or l.startswith("label"))]
with open(train_config.slot_path) as f:
    orig_slots_dict = {l.strip("\n"):idx for idx,l in enumerate(f) if not l.startswith("#")}
    orig_select_feature, orig_select_index = list(orig_slots_dict.keys()), list(orig_slots_dict.values())
    #
    sel_features = []
    if os.path.exists(train_config.sel_feat_path):
        with open(train_config.sel_feat_path) as f1:
            sel_features = [l.strip("\n") for l in f1 if not (l.startswith("#") or l.startswith("label"))]
    slots_dict = {l:idx for l,idx in orig_slots_dict.items() if (not sel_features or l in sel_features)}
    select_feature, select_index = list(slots_dict.keys()), list(slots_dict.values())
with open(train_config.boundaries_map_path) as f:
    boundaries_map = json.load(f)

with open(train_config.fg_path, "r") as json_f:
    schema_fea2idx_dict = {v['feature_name']: ("f" + str(idx + 1)) for idx, v in
                           enumerate(json.load(json_f)['features'])}


def write_donefile(time_str, donefile):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(donefile):
        open(donefile, "w").close()
    with open(donefile) as f:
        txt = list(map(lambda x: x.strip("\n"), f.readlines()))
    ###########################################
    txt.append(f"{time_str}\t{now}")
    with open(donefile, "w") as wf:
        wf.write("\n".join(txt))


def _parse_type(type_str):
    type_str = type_str.upper()
    if type_str == 'STRING':
        return tf.string
    elif type_str == 'FLOAT':
        return tf.float32
    elif type_str == 'LONG':
        return tf.int64
    else:
        arr_re = re.compile("ARRAY<(.*)>\(?(\d*)\)?")
        t = arr_re.findall(type_str)
        if len(t) == 1 and isinstance(t[0], tuple) and len(t[0]) == 2:
            return _parse_type(t[0][0]), 50 if t[0][1] == "" else int(t[0][1])
        raise TypeError("Unsupport type", type_str)


def get_exists_schema():
    schema = list()
    with open(train_config.schema_path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("label"):
                continue
            name = re.split(" +", line.strip("\n"))[0]
            schema.append(name)
    return schema


def get_example_fmt():
    example_fmt = {}
    with open(train_config.schema_path) as f:
        for line in f:
            name = re.split(" +", line.strip("\n"))[0]
            if line.startswith("#"):
                continue
            if line.startswith("label"):
                example_fmt[name] = tf.io.FixedLenFeature((), tf.float32)
            elif "ARRAY" in line:
                type_str = re.split(" +", line.strip("\n"))[1]
                dtype, length = _parse_type(type_str)
                example_fmt[name] = tf.io.FixedLenFeature([length], dtype)
            else:
                # 单特征默认tf.string
                example_fmt[name] = tf.io.FixedLenFeature((), tf.string)
    if "label" not in example_fmt:
        example_fmt["label"] = tf.io.FixedLenFeature((), tf.float32)
    return example_fmt


def serving_input_receiver_dense_fn():
    tf.compat.v1.disable_eager_execution()

    inps = {}
    lst = []
    for i, k in enumerate(schema):
        k_idx = schema_fea2idx_dict[k]
        if k_idx is None:
            print(f'=========none index:{k}')
        inp = tf.compat.v1.placeholder(tf.string, shape=(None,), name=k)
        inps[k_idx] = inp
        if k not in slots_dict:
            continue

        if k in boundaries_map:
            boundary = boundaries_map[k]
            inp = tf.searchsorted(tf.constant(boundary, tf.float32), tf.strings.to_number(inp), "right")
            inp = tf.strings.join(["%d" % i, tf.as_string(inp)], ":")
            lst.append(tf.expand_dims(inp, 1))
            print(f'feature:{k} in boundaries_map, boundary idx:{inp}')
        elif (k.startswith("user__") or k.startswith("item__") or k.startswith("doc__")) and "_div_" not in k:
            boundary = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
            inp = tf.searchsorted(tf.constant(boundary, tf.float32), tf.strings.to_number(inp), "right")
            inp = tf.strings.join(["%d" % i, tf.as_string(inp)], ":")
            lst.append(tf.expand_dims(inp, 1))
            print(f'user or item feature:{k} use default boundary:{boundary} , boundary idx:{inp}')
        elif (k.startswith("user__") or k.startswith("item__") or k.startswith("doc__")) and "_div_" in k:
            boundary = [0.0, 0.0061, 0.0068, 0.0074, 0.0078, 0.0081, 0.0085, 0.0089, 0.0093, 0.0096, 0.0098, 0.0101,
                        0.0107, 0.0112, 0.0119, 0.0126, 0.0132, 0.0134, 0.0139, 0.0146, 0.0152, 0.0154, 0.0156, 0.016,
                        0.0164, 0.017, 0.0181, 0.0195, 0.0199, 0.0212, 0.0225, 0.0235, 0.0242, 0.0251, 0.0265, 0.0276,
                        0.0286, 0.0297, 0.0305, 0.0317, 0.0324, 0.0331, 0.0338, 0.0347, 0.0355, 0.0369, 0.0376, 0.0385,
                        0.0397, 0.0408, 0.0429, 0.0443, 0.0456, 0.0471, 0.0512, 0.0554, 0.0584, 0.0603, 0.0656, 0.0699,
                        0.0725, 0.0734, 0.0748, 0.0768, 0.0786, 0.0803, 0.0816, 0.0824, 0.0842, 0.0889, 0.0958, 0.1,
                        0.1036, 0.1113, 0.1192, 0.2, 0.2497, 0.328, 0.3625, 0.4145, 0.5467]
            inp = tf.searchsorted(tf.constant(boundary, tf.float32), tf.strings.to_number(inp), "right")
            inp = tf.strings.join(["%d" % i, tf.as_string(inp)], ":")
            lst.append(tf.expand_dims(inp, 1))
            print(f'user or item cross stat feature:{k} use default boundary:{boundary},boundary idx:{inp}')
        # elif k != 'user_id':
        else:
            inp = tf.strings.join(["%d" % i, inp], ":")
            lst.append(tf.expand_dims(inp, 1))
            print(f'defalut feature:{inp}')

    features = {
        "features": tf.concat(lst, axis=1),
    }
    print(f"features: {features} tensors: {len(inps)}")
    return tf.compat.v1.estimator.export.ServingInputReceiver(features, inps)


def _gauc(user_id, label, prob):
    uid_label_map = defaultdict(list)
    uid_prob_map = defaultdict(list)
    for i in range(len(user_id)):
        uid_label_map[user_id[i]].append(label[i])
        uid_prob_map[user_id[i]].append(prob[i])

    total_imp = 0
    total_auc = 0
    for uid in uid_label_map:
        try:
            imp = len(uid_label_map[uid])
            if imp < 10:
                continue
            auc = imp * metrics.roc_auc_score(uid_label_map[uid], uid_prob_map[uid])
            total_imp += imp
            total_auc += auc
        except:
            pass
    return total_auc / (total_imp + 1e-10)


def get_metrics(df):
    metric_map = {}
    user_id = df.index
    y = df.iloc[:, :df.shape[1] // 2].values
    p = df.iloc[:, df.shape[1] // 2:].values
    assert y.shape == p.shape
    for i in range(y.shape[1]):
        auc = metrics.roc_auc_score(y[:, i], p[:, i])
        pcoc = p[:, i].mean() / (y[:, i].mean() + 1e-10)
        gauc = _gauc(user_id, y[:, i], p[:, i])
        mae = metrics.mean_absolute_error(y[:, i], p[:, i])
        real_ctr = y[:, i].mean()
        prob = p[:, i].mean()

        _metric_names = map(lambda x: "%s_%s" % (x, i), "auc, pcoc, gauc, mae, real_ctr, prob".split(", "))
        _metric_map = dict(zip(_metric_names, [auc, pcoc, gauc, mae, real_ctr, prob]))
        metric_map.update(_metric_map)

    return metric_map
