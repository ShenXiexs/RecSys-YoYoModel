#!/usr/bin/env python
# coding=utf-8
import os
import json
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de

from layers.base import MLP
from common.metrics import evaluate
from collections import OrderedDict
from common.utils import select_feature, slots_dict
from common.utils import train_config as TrainConfig

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def get_mask(tensor, padding="_"):
    # mask = tf.where(tensor == padding, tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32),
                    tf.ones_like(tensor, dtype=tf.float32))
    return mask


def masked_pooling(inputs, inputs_emb, mask_value, pooling_mode="mean"):
    """
    带 mask 的 pooling 操作，支持 mean/sum/target/concat 三种模式

    Args:
        inputs: 原始输入张量，用于生成 mask（形状：[batch_size, seq_len, ...]）
        inputs_emb: 待 pooling 的嵌入张量（形状：[batch_size, seq_len, emb_dim]）
        mask_value: 用于标记 mask 位置的数值（如 padding 对应的 0）
        pooling_mode: pooling 模式，可选 "mean"（默认）、"sum"、"target"、"concate"

    Returns:
        pooled_emb: 经过 mask 处理后的 pooling 结果（形状：[batch_size, emb_dim]）
    """
    # 1. 生成 mask 矩阵（True 表示有效位置，False 表示 mask 位置）
    mask = get_mask(inputs, mask_value)

    # 扩展 mask 维度以匹配 inputs_emb（从 [batch_size, seq_len] → [batch_size, seq_len, 1]）
    mask = tf.expand_dims(mask, axis=-1)

    # 2. 根据 pooling 模式执行对应操作
    if pooling_mode == "mean":
        # 平均池化：先对有效元素求和，再除以有效元素个数
        sum_emb = tf.reduce_sum(inputs_emb * mask, axis=1)
        # # 避免除以 0, + 1.e-12
        valid_count = tf.reduce_sum(mask, axis=1) + 1.e-12
        return sum_emb / valid_count
    elif pooling_mode == "sum":
        # 求和池化：直接对有效元素求和
        return tf.reduce_sum(inputs_emb * mask, axis=1)
    elif pooling_mode == "target":
        return inputs_emb[:, -1, :]
    elif pooling_mode == "concat":
        return tf.reshape(inputs_emb, sum(tf.shape(inputs_emb)[1:]))
    else:
        raise ValueError(f"不支持的 pooling 模式：{pooling_mode}，可选模式为 avg/max/sum")


def _get_embedding(feas, shape, embedding_table, policy, lookup_name='lookup'):
    '''

    :param feas: 输入的原始特征，shape:[batch_size,fea_size]
    :param shape: 输出的tensorflow的维度
    :param embedding_table: 动态Emb table
    :param lookup_name:
    :return:
    '''
    print(f"_get_embedding lookup_name:{lookup_name} feas shape:{tf.shape(feas)} feas:{feas}")
    #
    feat_val, id_idx = tf.unique(
        tf.reshape(feas, (-1,)))  # feat_val shape:(batch_size*fea_size) id_idx shape:(batch_size*fea_size)
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])
    #
    update_tstp_op = policy.apply_update(id_val)
    restrict_op = policy.apply_restriction(int(1e8))
    #
    sparse_weights, trainable_wrapper = de.embedding_lookup(embedding_table, id_val, return_trainable=True,
                                                            name=lookup_name)
    weights = tf.gather(sparse_weights, id_idx)  # (None * 150, 9)
    emb_lookuped = tf.reshape(weights, shape)
    return emb_lookuped, update_tstp_op, restrict_op


def get_feas_embedding(feas, slots_dict, embedding_table):
    feas_idx = [slots_dict[fea.strip("\n")] for fea in feas]
    cross_feas_embed = []
    for idx in feas_idx:
        cross_feas_embed.append(embedding_table[:, idx])
    feas_embed = tf.concat(cross_feas_embed, axis=1)
    return feas_embed


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.info(f"------ mode: {mode} strategy is {tf.compat.v1.distribute.get_strategy()} -------")
    logger.info(f"------ features: {features} -------")
    ######################devide################################
    device = params.get("device", "CPU").upper()
    if is_training:
        devices_info = ["/job:ps/replica:0/task:{}/{}:0".format(i, device) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for _ in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0: devices_info = None
    logger.info("------ dynamic_embedding devices_info is {} -------".format(devices_info))
    #
    groups = []
    batch_size = tf.shape(features["features"])[0]
    fea_size = len(select_feature)
    #
    embeddings_table = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=devices_info,
        trainable=is_training,
        initializer=initializer)
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings_table)
    #
    other_emb_lookuped, other_update_tstp_op, other_restrict_op = _get_embedding(features["features"],
                                                                                 [batch_size,
                                                                                  fea_size * embeddings_table.dim],
                                                                                 embeddings_table, policy, 'lookup')
    groups.append(other_update_tstp_op)
    logger.info(f"other_emb_lookuped={other_emb_lookuped}")
    if params["restrict"]:
        groups.append(other_restrict_op)
    # user_seq feature
    for seq_col, seq_len in TrainConfig.seq_length.items():
        if seq_col in features:
            seq_feat = tf.strings.split(features[seq_col], "\002").to_tensor()  # [B]->[B, L]
            seq_feat = tf.reshape(seq_feat, [batch_size, seq_len])
            emb_lookuped, update_tstp_op, restrict_op = _get_embedding(seq_feat,
                                                                       [batch_size, seq_len, embeddings_table.dim],
                                                                       embeddings_table,
                                                                       policy,
                                                                       seq_col)  # [B, L]->[B, L, D]
            emb_lookuped = masked_pooling(seq_feat, emb_lookuped, "0", "mean")
            other_emb_lookuped = tf.concat([other_emb_lookuped, emb_lookuped], axis=1)
    #
    losses = 0
    with tf.name_scope("dnn"):
        logger.info(f"---------init MLP-----------")
        DNN = MLP(**params["dnn_config"])
        logger.info(f"---------run MLP-----------")
        logger.info(f"other_emb_lookuped2={other_emb_lookuped}")
        ctr_prob = DNN(other_emb_lookuped, training=is_training)
        logger.info(f"---------ctr_prob={ctr_prob}-----------")
        ctr_label = tf.reshape(features['click_label'] if 'click_label' in features \
                                   else tf.zeros((batch_size,)), [-1, 1])
        logger.info(f"---------ctr_label={ctr_label}-----------")
        ctr_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=ctr_label, output=ctr_prob))
        losses += ctr_loss

        predictions = {
            "requestid": features["requestid"] if "requestid" in features else tf.as_string(
                tf.zeros((batch_size,), tf.int16)),
            "combination_un_id": features["combination_un_id"] if "combination_un_id" in features else tf.as_string(
                tf.zeros((batch_size,), tf.int16)),
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in [ctr_label, ctr_prob]], axis=1),
                             axis=1)
        }
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings_table.size(),  # embeddings_other.size()+embeddings_uid.size(),
        "ctr_losses": ctr_loss,
    })
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()
        evaluate(ctr_label, ctr_prob, f"task_ctr", eval_metric_ops)
        for k in eval_metric_ops:
            loggings[k] = eval_metric_ops[k][0]
            groups.append(eval_metric_ops[k][1])
        if params["slot"]:
            eval_metric_ops[f'slot_{params["slot"]}'] = eval_metric_ops[k]

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

        l2_regularization = 1e-06
        l2_loss = l2_regularization * tf.add_n(
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in trainable_variables if 'bias' not in v.name])
        losses += l2_loss
        loggings["l2_loss"] = l2_loss
        loggings['losses'] = losses

        optimize_config = params.get("optimize_config", {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        })
        optimizer = tf.compat.v1.train.AdamOptimizer(**optimize_config)
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

        dense_op = optimizer.minimize(losses, global_step=global_step)  # , var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
        ######################WarmStartHook################################
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=losses,  # loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None  # [sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput({"ctr_output": ctr_prob})
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)  # export_outputs:exported to`SavedModel` and used during serving
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=losses,
            eval_metric_ops=eval_metric_ops)
    else:
        None
