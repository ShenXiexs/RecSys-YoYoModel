#!/usr/bin/env python
# coding=utf-8
import os
import sys
import tensorflow as tf

import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from layers.esmm import ESMM
from common.metrics import evaluate
from collections import OrderedDict
from common.utils import *

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 假设config在项目根目录下
sys.path.insert(0, project_root)
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def model_fn(features, labels, mode, params):
    device = params.get("device", "CPU")
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.info(f"------ mode: {mode} strategy is {tf.compat.v1.distribute.get_strategy()} -------")
    logger.info(f"------ features: {features} -------")
    logger.info(f">>>------ train_config.__dict__: {train_config.__dict__}")
    ######################devide################################
    if is_training:
        # 训练支持多卡
        devices_info = ["/job:localhost/replica:0/task:{}/{}:{}".format(i, device, i) for i in
                        range(params["ps_num"])]  # ps_num也就是GPU训练使用的个数等同gpu_num
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
        # eval和infer都是单卡，使用GPU:0
        devices_info = ["/job:localhost/replica:0/task:{}/{}:0".format(0, device) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    else:
        # export的时候用CPU
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0: devices_info = "/job:localhost/replica:0/task:0/CPU:0"
    logger.info("------ dynamic_embedding devices_info is {} -------".format(devices_info))
    ##################################################################
    # 合并普通特征和序列特征一起进行hash和embedding lookup
    batch_size = tf.shape(features["features"])[0]
    # 处理序列特征: [batch_size, num_sequences, seq_len] -> [batch_size, num_sequences * seq_len]
    #sequence_features = tf.stack(list(features["seq_features"].values()), axis=1)  # [None, 26, 50] for each sample
    num_sequences = len(seq_idxs)  #len(features["seq_features"])  # 序列特征的数量
    seq_len = seq_length_list[0]  # 每个序列的长度
    # 重塑序列特征 [batch_size, 26, 50] -> [batch_size, 26*50]
    seq_features_flat = features["seq_features"]  #tf.reshape(sequence_features, [batch_size, num_sequences * seq_len])

    # 合并所有特征 [batch_size, (num_regular_features + num_sequences * seq_len)]
    all_features = tf.concat([features["features"], seq_features_flat], axis=1)

    # 统一进行hash和embedding lookup
    feat_val, id_idx = tf.unique(tf.reshape(all_features, (-1,)))
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])

    ######################emb_de################################
    groups = []
    embeddings = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=devices_info,
        trainable=is_training,
        initializer=initializer)
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings)
    update_tstp_op = policy.apply_update(id_val)
    restrict_op = policy.apply_restriction(int(1e8))
    groups.append(update_tstp_op)
    if params["restrict"]: groups.append(restrict_op)

    ######################lookup################################
    sparse_weights, trainable_wrapper = de.embedding_lookup(embeddings, id_val, return_trainable=True, name="lookup")
    # tf.compat.v1.add_to_collections(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, embeddings)
    weights = tf.gather(sparse_weights, id_idx)  # (None * 150, 9)
    ######################lookup end################################
    num_regular_features = len(select_feature)
    total_features = num_regular_features + num_sequences * seq_len
    # 重塑所有特征的embedding [batch_size, total_features, 9]
    all_embeddings = tf.reshape(weights, [batch_size, total_features, embedding_size])
    # 分离普通特征和序列特征的embedding
    # 普通特征embedding: [batch_size, num_regular_features, 9] -> [batch_size, num_regular_features * 9]
    regular_emb = tf.reshape(all_embeddings[:, :num_regular_features, :],
                             [batch_size, num_regular_features * embedding_size])

    # 序列特征embedding: [batch_size, num_sequences * seq_len, 9] -> [batch_size, num_sequences, seq_len, 9]
    seq_emb_reshaped = tf.reshape(all_embeddings[:, num_regular_features:, :],
                                  [batch_size, num_sequences, seq_len, embedding_size])

    # 创建序列mask (填充位置为0，其他为1)
    seq_mask = tf.not_equal(seq_features_flat, "0")  # [batch_size, num_sequences * seq_len]
    seq_mask = tf.reshape(seq_mask, [batch_size, num_sequences, seq_len])  # [batch_size, num_sequences, seq_len]
    seq_mask_float = tf.cast(seq_mask, tf.float32)  # [batch_size, num_sequences, seq_len]
    seq_mask_expanded = tf.expand_dims(seq_mask_float, -1)  # [batch_size, num_sequences, seq_len, 1]

    # 应用mask并计算均值pooling
    seq_emb_masked = seq_emb_reshaped * seq_mask_expanded  # [batch_size, num_sequences, seq_len, 9]
    seq_sum = tf.reduce_sum(seq_emb_masked, axis=2)  # [batch_size, num_sequences, 9]
    seq_count = tf.reduce_sum(seq_mask_float, axis=2, keepdims=True)  # [batch_size, num_sequences, 1]
    seq_count = tf.maximum(seq_count, 1.0)  # 避免除0
    seq_emb_pooled = seq_sum / seq_count  # [batch_size, num_sequences, 9]

    # 展平序列特征embedding [batch_size, num_sequences * 9]
    seq_emb_flat = tf.reshape(seq_emb_pooled, [batch_size, num_sequences * embedding_size])

    # 合并普通特征和序列特征的embedding
    total_emb = tf.concat([regular_emb, seq_emb_flat], axis=1)

    logger.info(f"------embeddings: {embeddings} emb_lookuped: {total_emb} -------")
    with tf.name_scope("esmm"):
        model = ESMM(features, total_emb, is_training)
    # model.predictions = {"value": feat_val, "sign": id_val, "emb": sparse_weights}
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings.size(),
        "losses": model.losses,
        "ctr_losses": model.ctr_losses,
        "ctcvr_losses": model.ctcvr_losses
    })
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()
        for i, (ctr_label, ctr_prob) in enumerate(zip(model.ctr_labels, model.ctr_probs), start=1):
            evaluate(ctr_label, ctr_prob, f"task{i}_ctr", eval_metric_ops)

        for i, (ctcvr_label, ctcvr_prob) in enumerate(zip(model.ctcvr_labels, model.ctcvr_probs), start=1):
            print("model.ctr_labels shape:", tf.shape(model.ctr_labels))
            print("model.ctr_probs shape:", tf.shape(model.ctr_probs))
            print("model.ctcvr_labels shape:", tf.shape(model.ctcvr_labels))
            print("model.ctcvr_probs shape:", tf.shape(model.ctcvr_probs))
            print("---model.ctcvr_label shape:", tf.shape(ctcvr_label))
            print("---model.ctcvr_prob shape:", tf.shape(ctcvr_prob))
            print("model.ctcvr_prob data:", ctcvr_prob)
            print("model.ctcvr_label data:", ctcvr_label)

            evaluate(ctcvr_label, ctcvr_prob, f"task{i}_ctcvr", eval_metric_ops)

        for k in eval_metric_ops:
            loggings[k] = eval_metric_ops[k][0]
            groups.append(eval_metric_ops[k][1])
        if params["slot"]:
            eval_metric_ops[f'slot_{params["slot"]}'] = eval_metric_ops[k]

    ######################train################################
    if mode == tf.estimator.ModeKeys.TRAIN:
        trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        dense_vars = [var for var in trainable_variables if not var.name.startswith("lookup")]
        sparse_vars = [var for var in trainable_variables if var.name.startswith("lookup")]
        global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        logger.debug(f"trainable_variables: {trainable_variables}")
        logger.debug(f"dense_vars: {dense_vars}")
        logger.debug(f"sparse_vars: {sparse_vars}")
        logger.debug(f"global_vars: {global_variables}")
        logger.debug(f"l2_loss_vars: {[v for v in trainable_variables if 'bias' not in v.name]}")
        logger.debug("#" * 100)
        ######################l2_regularization################################
        l2_regularization = 1e-06
        l2_loss = l2_regularization * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in trainable_variables if 'bias' not in v.name])
        model.losses += l2_loss
        loggings["l2_loss"] = l2_loss

        optimizer = tf.compat.v1.train.AdamOptimizer()
        # optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=0.001)
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

        dense_op = optimizer.minimize(model.losses, global_step=global_step)  # , var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
        ######################WarmStartHook################################
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            loss=model.losses,  # loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None  # [sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput(model.outputs)
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            export_outputs=export_outputs)  # export_outputs:exported to`SavedModel` and used during serving
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            loss=model.losses,
            eval_metric_ops=eval_metric_ops)
    else:
        return None
