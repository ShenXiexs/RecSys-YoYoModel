# -*- coding: utf-8 -*-
# @Time : 2025/11/19 16:29
# @Author : huangmian
# @File : embeddings.py
import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de


def create_lookup_emb_tabel(id_features, embedding_size, devices_info,
                            is_training, initializer, params, name=""):
    groups = []
    embeddings = tfra.dynamic_embedding.get_variable(
        name="_".join([name, "embeddings"]).lstrip("_"),
        dim=embedding_size,
        devices=devices_info,
        trainable=is_training,
        initializer=initializer)
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings)
    restrict_op = policy.apply_restriction(int(1e8))
    if params["restrict"]: groups.append(restrict_op)
    emb_lookuped = lookup_emb_table(id_features, groups, embeddings, policy)
    return emb_lookuped, groups, embeddings, policy

def lookup_emb_table(id_features, groups, emb_table, policy):
    feat_val, id_idx = tf.unique(tf.reshape(id_features, (-1,)))
    id_val = tf.strings.to_hash_bucket_strong(feat_val, 2 ** 63 - 1, [1, 2])
    #
    update_tstp_op = policy.apply_update(id_val)
    groups.append(update_tstp_op)
    # lookup
    sparse_weights, trainable_wrapper = de.embedding_lookup(emb_table, id_val, return_trainable=True, name="lookup")
    weights = tf.gather(sparse_weights, id_idx)
    # reshape
    shape_list = id_features.get_shape().as_list()
    if len(shape_list) == 2:
        _, features_num = shape_list
        emb_lookuped = tf.reshape(weights, [-1, features_num * emb_table.dim])
    elif len(shape_list) == 3:
        _, features_num, seq_len = shape_list
        emb_lookuped = tf.reshape(weights, [-1, features_num, seq_len, emb_table.dim])
    return emb_lookuped
