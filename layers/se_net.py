# -*- coding: utf-8 -*-
# @Time : 2025/10/11 10:08
# @Author : huangmian
# @File : se_net.py
import math
import tensorflow as tf

from layers.base import DNN
from layers.activation import get_activation
from layers.normalization import bn_layer


class SeNet:
    def __init__(self, num_fields, reduction_ratio=3, excitation_activation="ReLU"):
        reduced_size = max(1, int(num_fields / reduction_ratio))
        excitation = [tf.keras.layers.Dense(reduced_size, use_bias=False),
                      get_activation('relu'),
                      tf.keras.layers.Dense(num_fields, use_bias=False)]
        if excitation_activation.lower() == "relu":
            excitation.append(get_activation('relu'))
        elif excitation_activation.lower() == "sigmoid":
            excitation.append(get_activation('sigmoid'))
        else:
            raise NotImplementedError
        self.excitation = excitation

    def __call__(self, feature_emb, training=False, **kwargs):
        A = tf.reduce_mean(feature_emb, axis=-1)
        for layer in self.excitation:
            A = layer(A, training=training)
        if kwargs.get("return_weight", False):
            return A
        V = feature_emb * tf.expand_dims(A, axis=-1)
        return V

class ContextGating:
    def __init__(self, bn=True, name="context_gating"):
        self.bn = bn
        self.name = name

    def __call__(self, input_layer, product_layer, training=False):
        """
        Variant of se_net: context_gating applies vector weighting on the last dim of product_layer.
        Args:
            input_layer: build gate weights from input_layer, [B, iD]
            product_layer: vector to be weighted, [B, oD]
            bn: whether to apply BN on gates or add bias, False
            training: training phase flag, False
            name:
        Returns:
            weighted product_layer
        """
        input_dim = input_layer.get_shape().as_list()[-1]
        # Input dim matches output dim because it multiplies input_gate.
        output_dim = product_layer.get_shape().as_list()[-1]
        # Output dim must match product_layer because they are multiplied.
        gating_weights = tf.compat.v1.get_variable("_".join((self.name, "weights")),
                                                   [input_dim, output_dim],
                                                   initializer=tf.compat.v1.truncated_normal_initializer(
                                                       stddev=math.sqrt(2.0 / (input_dim + output_dim))),
                                                   # Larger dims yield more variance, so use smaller stddev for stability.
                                                   dtype=input_layer.dtype)  # [iD, oD]
        # Compute cost is huge: e.g., dim=3200 => 1024*3200*3200 ~ 1e10 multiplications.
        gates = tf.matmul(input_layer, gating_weights)  # [B, iD]->[B, oD]
        if self.bn:
            # After gate_weights, apply BN for gate_net stability before weighting the original network.
            gates = bn_layer(gates, training, name="_".join((self.name, "bn")))
        else:
            gating_biases = tf.compat.v1.get_variable("_".join((self.name, "biased")),
                                                      shape=[output_dim],
                                                      initializer=tf.compat.v1.truncated_normal_initializer(
                                                          stddev=1.0 / math.sqrt(output_dim)),
                                                      dtype=input_layer.dtype)
            gates += gating_biases
        # After weighting, values expand; compress back to probability space to unify scales.
        gates = tf.sigmoid(gates)  # [B, oD]
        new_product_layer = tf.multiply(product_layer, gates)  # [B, oD]
        return new_product_layer

class TransformGate:
    def __init__(self, input_layer, product_layer, sigmoid_factor=1, bBN=True):
        input_dim = input_layer.get_shape().as_list()[-1]
        output_dim = output_dim = product_layer.get_shape().as_list()[-1]
        self.input_layer = input_layer
        self.product_layer = product_layer
        self.sigmoid_factor = sigmoid_factor
        self.bBN = bBN
        # Equivalent to a 2-layer MLP: widen then restore; first layer uses dice activation.
        middle_dim = 2 * input_dim  # Widen the gate to increase weighting dims; compute cost rises, evaluate benefit.
        self.gate_layer = DNN(hidden_units=[middle_dim],
                              hidden_activations="dice",
                              output_dim=output_dim,
                              output_activation=None,
                              dropout_rates=0.0,
                              batch_norm=False,
                              bn_only_once=False,  # Set True for inference speed up
                              output_kernel_initializer=None,
                              kernel_initializer=None,
                              bias_initializer=None,
                              use_bias=False
                              )

    def __call__(self, training=False, **kwargs):
        gates = self.gate_layer(self.input_layer, training=training)
        if self.bBN:
            gates = bn_layer(gates, training=training)
        gates = self.sigmoid_factor * tf.sigmoid(gates)
        activation = tf.multiply(self.product_layer, gates)
        return activation
