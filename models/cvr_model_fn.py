#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict
from layers.esmm import ESMM
from common.metrics import evaluate
from common.utils import select_feature

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def mask_tensor(tensor, padding="_"):
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32),
                    tf.ones_like(tensor, dtype=tf.float32))
    return mask


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.info(f"------ mode: {mode} strategy is {tf.compat.v1.distribute.get_strategy()} -------")
    logger.info(f"------ features: {features} -------")
    ######################devide################################
    if is_training:
        devices_info = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for i in range(params["ps_num"])]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0: devices_info = None
    logger.info("------ dynamic_embedding devices_info is {} -------".format(devices_info))
    ##################################################################
    feat_val, id_idx = tf.unique(tf.reshape(features["features"], (-1,)))
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])
    ######################dnn################################
    # with tf.name_scope("embedding"):
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
    # id_val, id_idx = tf.unique(tf.reshape(ids, (-1,)))
    sparse_weights, trainable_wrapper = de.embedding_lookup(embeddings, id_val, return_trainable=True, name="lookup")
    # tf.compat.v1.add_to_collections(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, embeddings)
    weights = tf.gather(sparse_weights, id_idx)  # (None * 150, 9)
    ######################lookup end################################
    batch_size = tf.shape(features["features"])[0]
    features_num = len(select_feature)
    emb_lookuped = tf.reshape(weights, [batch_size, features_num * embedding_size])
    #
    logger.info(f"------embeddings: {embeddings} emb_lookuped: {emb_lookuped} -------")
    with tf.name_scope("dnn"):
        model = ESMM(features, emb_lookuped, is_training)
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
        None
