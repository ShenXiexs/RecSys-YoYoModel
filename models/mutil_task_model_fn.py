#!/usr/bin/env python
# coding=utf-8
import os
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict
from layers.mmoe import mmoe_layer
from layers.base import MLP, DNN
from common.metrics import evaluate
from common.utils import select_feature, train_config

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def mask_tensor(tensor, padding="_"):
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32),
                    tf.ones_like(tensor, dtype=tf.float32))
    return mask


def focal_loss_sigmoid(labels, logits, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification
    Args:
        labels: true labels
        logits: model logits (before sigmoid)
        alpha: balancing parameter
        gamma: focusing parameter
    """
    # 计算概率
    probabilities = logits  # tf.sigmoid(logits)
    # 计算p_t
    p_t = tf.where(tf.equal(labels, 1.0), probabilities, 1 - probabilities)
    # 计算alpha_t
    alpha_t = tf.where(tf.equal(labels, 1.0), alpha, 1 - alpha)
    # Focal Loss
    focal_loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-8, 1.0))
    return tf.reduce_mean(focal_loss)


def model_fn(features, labels, mode, params):
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
    with tf.name_scope("mmoe"):
        label_schema = getattr(train_config, "label_schema", {})
        if len(label_schema) == 0:
            ValueError("params.label_schema is emtpy, please check inputs!")
        mmoe_task_outlist = mmoe_layer(emb_lookuped,
                                       num_domains=params.get("num_domains", len(label_schema)),
                                       num_experts=params.get("num_experts", 4),
                                       exprt_units=params.get("exprt_units", [128, 64, 128]))
        if params.get("final_layer_type", "mlp"):
            tower_layers = [MLP(output_dim=1,
                                hidden_units=params.get("hidden_units", [128, 64, 128]),
                                hidden_activations=params.get("hidden_activations", 'relu'),
                                output_activation=None,
                                dropout_rates=params.get("dropout_rates", 0),
                                batch_norm=params.get("batch_norm", False),
                                output_kernel_initializer=params.get("output_kernel_initializer", None),
                                kernel_initializer=params.get("kernel_initializer", None),
                                bias_initializer=params.get("bias_initializer", None),
                                use_bias=params.get("use_bias", False))
                     for _ in range(len(label_schema))]
        else:
            tower_layers = [DNN(output_dim=1,
                                hidden_units=params.get("hidden_units", [128, 64, 128]),
                                hidden_activations=params.get("hidden_activations", 'relu'),
                                output_activation=None,
                                dropout_rates=params.get("dropout_rates", 0),
                                batch_norm=params.get("batch_norm", False),
                                output_kernel_initializer=params.get("output_kernel_initializer", None),
                                kernel_initializer=params.get("kernel_initializer", None),
                                bias_initializer=params.get("bias_initializer", None),
                                use_bias=params.get("use_bias", False))
                            for _ in range(len(label_schema))]
        losses = 0
        tower_logits = []
        all_labels = []
        losses_dict = {}
        for i, (task_pred, label_nm) in enumerate(zip(mmoe_task_outlist, label_schema.values())):
            label_ = tf.cast(features[label_nm], tf.float32) if label_nm in features else tf.zeros((batch_size, ))
            all_labels.append(label_)
            task_logits = tf.sigmoid(tf.clip_by_value(tf.reshape(tower_layers[i](task_pred), (-1,)),
                                                      -15, 15))
            tower_logits.append(task_logits)
            logger.info(f"---label_{i}:{label_}, task_pred_{i}:{task_pred}, task_logits_{i}:{task_logits}---")
            if label_nm == "order_label" and params.get("use_focal_loss", False):
                order_alpha = params.get('order_alpha', 0.75)  # 正样本权重
                order_gamma = params.get('order_gamma', 2.0)  # 聚焦参数
                loss = focal_loss_sigmoid(label_, task_logits, order_alpha, order_gamma)
            else:
                loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=label_,
                                                                       output=task_logits))
            task_weight = params.get(f'{label_nm}_weight', 1.0)
            loss = loss * task_weight
            losses_dict[label_nm.split("_")[0] + "_loss"] = loss
            losses += loss
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
        for label_, task_pred, label_nm in zip(all_labels, tower_logits, label_schema.values()):
            evaluate(label_, task_pred, f"task_{label_nm}", eval_metric_ops)
        #
        # evaluate(tf.reshape(tf.concat(all_labels, axis=1), [-1]),
        #          tf.reshape(tf.concat(tower_logits, axis=1), [-1]),
        #          f"mmoe_tower",
        #          eval_metric_ops)
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
        "out": tf.concat([tf.reshape(x, [-1, 1]) for x in all_labels + tower_logits], axis=1)
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

        lr = params.get("lr", 0.001)
        optimizer = tf.compat.v1.train.AdamOptimizer(lr)
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)

        dense_op = optimizer.minimize(losses, global_step=global_step)  # , var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=losses,  # loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None  # [sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        outputs = tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in tower_logits], axis=1), axis=1)
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
