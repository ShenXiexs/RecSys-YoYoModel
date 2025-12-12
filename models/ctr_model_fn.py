#!/usr/bin/env python
# coding=utf-8
import os
import json
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from models.ctr_mmoe_dcn import DNN
from common.metrics import evaluate
from collections import OrderedDict
from common.utils import select_feature, slots_dict
from common.utils import train_config as TrainConfig

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = tf.compat.v1.logging


def mask_tensor(tensor, padding="_"):
    # mask = tf.where(tensor == padding, tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    padding = tf.constant(padding, dtype=tensor.dtype)
    mask = tf.where(tf.equal(tensor, padding), tf.zeros_like(tensor, dtype=tf.float32), tf.ones_like(tensor, dtype=tf.float32))
    return mask


def _get_embedding(feas,batch_size,fea_size,embedding_table,lookup_name='lookup'):
    '''

    :param feas: 输入的原始特征，shape:[batch_size,fea_size]
    :param embedding_table: tfra.dynamic_embedding.get_variable
    :param lookup_name:
    :return:
    '''
    print(f"_get_embedding lookup_name:{lookup_name} feas shape:{tf.shape(feas)} feas:{feas}")

    feat_val, id_idx = tf.unique(tf.reshape(feas, (-1,))) # feat_val shape:(batch_size*fea_size) id_idx shape:(batch_size*fea_size)
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])

    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embedding_table)
    update_tstp_op = policy.apply_update(id_val)
    restrict_op = policy.apply_restriction(int(1e8))

    sparse_weights, trainable_wrapper = de.embedding_lookup(embedding_table, id_val, return_trainable=True, name=lookup_name)
    weights = tf.gather(sparse_weights, id_idx)  #(None * 150, 9)

    embedding_size = embedding_table.dim
    emb_lookuped = tf.reshape(weights, [batch_size, fea_size * embedding_size])
    return emb_lookuped,update_tstp_op,restrict_op


def get_feas_embedding(feas,slots_dict,embedding_table):
    feas_idx = [slots_dict[fea.strip("\n")] for fea in feas]
    cross_feas_embed = []
    for idx in feas_idx:
        cross_feas_embed.append(embedding_table[:,idx])
    feas_embed = tf.concat(cross_feas_embed,axis=1)
    return feas_embed


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

    #
    groups = []
    batch_size = tf.shape(features["features"])[0]
    fea_size = len(select_feature)

    embeddings_other = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=devices_info,
        trainable=is_training,
        initializer=initializer)

    other_emb_lookuped,other_update_tstp_op,other_restrict_op = _get_embedding(features["features"],batch_size,fea_size,embeddings_other,'lookup')
    groups.append(other_update_tstp_op)
    if params["restrict"]:
        groups.append(other_restrict_op)
    #
    emb_lookuped = other_emb_lookuped
    #
    with open(TrainConfig.feature_config_path, mode='r') as f:
        datas = f.read()
        feature_conf_dict = json.loads(datas)
    #
    other_emb_tb = tf.reshape(other_emb_lookuped, [batch_size, fea_size, embedding_size])
    #
    ctr_dcn_fea_cols = feature_conf_dict['ctr_dcn']
    ctr_dcn_embed = get_feas_embedding(ctr_dcn_fea_cols, slots_dict, other_emb_tb)
    #
    ctr_key_feature_cols = feature_conf_dict['ctr_key_feature']
    ctr_key_fea_embed = get_feas_embedding(ctr_key_feature_cols, slots_dict, other_emb_tb)
    #
    logger.info(f"==========>ctr_dcn_fea_cols:{ctr_dcn_fea_cols}<============")
    logger.info(f"==========>ctr_key_feature_cols:{ctr_key_feature_cols}<============")
    logger.info(f"==========>ctr_dcn_embed shape:{tf.shape(ctr_dcn_embed)}<============")
    logger.info(f"==========>ctr_key_fea_embed shape:{tf.shape(ctr_key_fea_embed)}<============")


    with tf.name_scope("dnn"):
        model = DNN(features,
                     mmoe_input=emb_lookuped,
                     dcn_input={"ctr_dcn":ctr_dcn_embed},
                     key_fea_input={"ctr_key_feature":ctr_key_fea_embed},
                     is_training=is_training)
    ######################metrics################################
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings_other.size(), #embeddings_other.size()+embeddings_uid.size(),
        "ctr_losses":model.ctr_losses,
    })
    with tf.name_scope('metrics'):
        eval_metric_ops = OrderedDict()

        for i, (ctr_label,ctr_prob) in enumerate(zip(model.ctr_labels,model.ctr_probs), start=1):
            evaluate(ctr_label, ctr_prob, f"task{i}_ctr", eval_metric_ops)

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
        l2_loss = l2_regularization * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in trainable_variables if 'bias' not in v.name])
        model.ctr_losses += l2_loss
        loggings["l2_loss"] = l2_loss

        optimizer = tf.compat.v1.train.AdamOptimizer()
        optimizer = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(optimizer)


        dense_op = optimizer.minimize(model.ctr_losses, global_step=global_step)#, var_list=dense_vars)
        train_op = tf.group(dense_op, *groups)

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
        ######################WarmStartHook################################
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=model.predictions,
            loss=model.ctr_losses, # loss: Training loss `Tensor`. Must be either scalar, or with shape `[1]`
            train_op=train_op,
            training_hooks=[log_hook, ],
            training_chief_hooks=None#[sync_replicas_hook]
        )
    elif mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            "serving_default": tf.compat.v1.estimator.export.PredictOutput(model.outputs)
        }
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=model.predictions,
                export_outputs=export_outputs) # export_outputs:exported to`SavedModel` and used during serving
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=model.predictions,
                loss=model.ctr_losses,
                eval_metric_ops=eval_metric_ops)
    else:
        None

