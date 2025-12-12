#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

from tensorflow.python.framework import dtypes, ops
from tensorflow import dtypes
from tensorflow.python.ops import math_ops, array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.eager import context
from tensorflow.python.ops.metrics_impl import _aggregate_across_replicas, metric_variable, _remove_squeezable_dimensions


def sum(values,
        dtype=dtypes.int64,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name=None):

  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean is not supported when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'sum', (values, weights)):
    values = math_ops.cast(values, dtype)

    total = metric_variable([], dtype, name='total')

    if weights is not None:
        values, _, weights = _remove_squeezable_dimensions(predictions=values, labels=None, weights=weights)
        weights = weights_broadcast_ops.broadcast_weights(math_ops.cast(weights, dtype), values)
        values = math_ops.multiply(values, weights)

    update_op = state_ops.assign_add(total, math_ops.reduce_sum(values))

    def compute_sum(_, t):
        return t

    sum_t = _aggregate_across_replicas(metrics_collections, compute_sum, total)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return sum_t, update_op


tf.compat.v1.metrics.sum = sum
def evaluate(label, pred, task, eval_metric_ops):
    with tf.name_scope(f"metrics/{task}"):
        eval_metric_ops[f'{task}/auc'] = tf.compat.v1.metrics.auc(label, pred, num_thresholds=4068)
        ctr, ctr_op = tf.compat.v1.metrics.mean(label)
        prob, prob_op = tf.compat.v1.metrics.mean(pred)

        eval_metric_ops[f'{task}/pcoc'] = tf.compat.v1.div_no_nan(prob, ctr), tf.compat.v1.div_no_nan(prob_op, ctr_op)
        eval_metric_ops[f'{task}/mae'] = tf.compat.v1.metrics.mean_absolute_error(label, pred)
        eval_metric_ops[f'{task}/ctr'] = ctr, ctr_op
        eval_metric_ops[f'{task}/prob'] = prob, prob_op
        eval_metric_ops[f'{task}/cnt'] = tf.compat.v1.metrics.sum(tf.ones_like(label, dtype=tf.int64))




