#!/usr/bin/env python
# coding=utf-8
import os
import copy
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict
from layers.mmoe import mmoe_layer
from layers.base import DNN, MLP
from layers.se_net import ContextGating
from common.metrics import evaluate
from common.utils import select_feature, train_config

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def mask_tensor(tensor, padding="_"):
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32),
                    tf.ones_like(tensor, dtype=tf.float32))
    return mask

def ctcvr_fusion(hidden_inputs, logits_inputs, type="multiply", hidden_units=None, name=''):
    logger.info(f"------{name}-ctcvr_fusion.type={type}-----------")
    if type == 'mlp':
        fusion_inputs = tf.concat(hidden_inputs, axis=-1)
        mlp = DNN(hidden_units, output_dim=1, output_activation="sigmoid")
        pctcvr = mlp(fusion_inputs)
    else:
        pctcvr = logits_inputs[0]
        for cvr in logits_inputs[1:]:
            pctcvr *= cvr
    return tf.reshape(pctcvr, [-1])


def model_fn(features: dict, labels, mode, params: dict):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    ######################print features################################
    logger.info(f"------ mode: {mode} strategy is {tf.compat.v1.distribute.get_strategy()} -------")
    logger.info(f"------ features: {features} -------")
    logger.info(f"------ params: {params} -------")
    logger.info(f"------ train_config.__dict__: {train_config.__dict__}")
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
    # senet
    if params.get("use_senet", False):
        senet_layer = ContextGating(bn=True)
        emb_lookuped = senet_layer(emb_lookuped, emb_lookuped, is_training)
        logger.info(f"------senet out embedding: {emb_lookuped} -------")
    #
    label_schema = copy.deepcopy(getattr(train_config, "label_schema", {}))
    label_weights = {}
    for k in ['is_valid_shouden', 'is_valid_linghongbao', 'is_valid_yemianfangwen']:
        try:
            k_ = label_schema.pop(k)
            bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
            label_weights[k_.split("_")[0]] = (tf.reshape(features.get(k_, tf.ones((batch_size,))), [batch_size, 1]),
                                               bce_loss)
        except Exception as e:
            pass
    tower_ctcvr = []
    all_labels = []
    losses_dict = {}
    losses = 0
    #
    with tf.name_scope("middle_tower"):
        pred_dict = {}
        for label_nm in label_schema.values():
            label = features.get(label_nm, tf.zeros((batch_size,)))
            label_nm = label_nm.split("_")[0]
            label_weight, bce_loss = label_weights.get(label_nm, (None, tf.keras.losses.BinaryCrossentropy(from_logits=False)))
            mlp = DNN(**params.get("mlp_config", {}))
            hidden_output, logits = mlp(emb_lookuped, is_training, return_hidden=True)
            pred_dict[label_nm] = hidden_output, logits
            if label_nm == "ctr":
                logits = tf.sigmoid(tf.clip_by_value(tf.reshape(logits, (-1,)), -15, 15))
                ctcvr = logits
            elif label_nm == 'awake':
                ctr_hidden_output, ctr_logits = pred_dict["ctr"]
                ctcvr = ctcvr_fusion([ctr_hidden_output, hidden_output],
                                      [ctr_logits, logits],
                                      params.get("fusion_type", "multiply"),
                                      params.get("fusion_hidden_units", None),
                                      name=label_nm)
            else:
                ctr_hidden_output, ctr_logits = pred_dict["ctr"]
                awake_hidden_output, awake_logits = pred_dict["awake"]
                ctcvr = ctcvr_fusion([ctr_hidden_output, awake_hidden_output, hidden_output],
                                          [ctr_logits, awake_logits, logits],
                                          params.get("fusion_type", "multiply"),
                                          params.get("fusion_hidden_units", None),
                                          name=label_nm)
            loss = bce_loss(label, ctcvr, sample_weight=label_weight)
            losses_dict[label_nm + "_loss"] = loss
            loss_weight = params.get(label_nm+"_weight", 1)
            logger.info(f"---{label_nm} label={label}, logits={logits}, ctcvr={ctcvr}, loss_weight={loss_weight}---")
            losses += loss * loss_weight
            all_labels.append(label)
            tower_ctcvr.append(ctcvr)
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings.size(),
        "losses": losses
    })
    loggings.update(losses_dict)
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()
        for label_, task_pred, label_nm in zip(all_labels, tower_ctcvr, label_schema.values()):
            label_weight, _ = label_weights.get(label_nm.split("_")[0], (tf.ones(batch_size), None))
            label_weight = tf.reshape(label_weight, [batch_size, ])
            evaluate(tf.boolean_mask(label_, label_weight),
                     tf.boolean_mask(task_pred, label_weight),
                     f"{label_nm.split('_')[0]}", eval_metric_ops)
        for k in eval_metric_ops:
            loggings[k] = eval_metric_ops[k][0]
            groups.append(eval_metric_ops[k][1])
        if params["slot"]:
            eval_metric_ops[f'slot_{params["slot"]}'] = eval_metric_ops[k]
    ###predictions###
    predictions = {
        "requestid": features["requestid"] if "requestid" in features else tf.as_string(
            tf.zeros((batch_size,), tf.int16)),
        "combination_un_id": features["combination_un_id"] if "combination_un_id" in features else tf.as_string(
            tf.zeros((batch_size,), tf.int16)),
        "out": tf.concat([tf.reshape(x, [-1, 1]) for x in
                          all_labels + tower_ctcvr + \
                          [v[0] for v in label_weights.values()]
                          ], axis=1)
    }
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
        losses += l2_loss
        loggings["l2_loss"] = l2_loss
        #
        optimizer = tf.compat.v1.train.AdamOptimizer(**params['optimize_config'])
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)
        #
        dense_op = optimizer.minimize(losses, global_step=global_step)  # , var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)
        #
        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
        #
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=losses,  # loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None  # [sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        outputs = tf.concat(
            tf.concat([tf.reshape(x, [-1, 1]) for x in tower_ctcvr], axis=1), axis=1)
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput(outputs)
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
