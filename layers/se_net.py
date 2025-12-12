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
        se_net的变种，区别在于context_gating是针对product_layer的最后一维做向量加权，从而实现对特征向量加权
        Args:
            input_layer: 基于input_layer构建gate_weight, [B, iD]
            product_layer: 待加权的向量, [B, oD]
            bn: gates是过BN还是添加bias, False
            training: 是否训练阶段, False
            name:
        Returns:
            product_layer加权后的值
        """
        input_dim = input_layer.get_shape().as_list()[-1]
        # 因为要和input_gate 矩阵乘，所以输入是他的输出维
        output_dim = product_layer.get_shape().as_list()[-1]
        # 因为最后要和product_layer乘，所以输出维度和他的输出维度要保持一致
        gating_weights = tf.compat.v1.get_variable("_".join((self.name, "weights")),
                                                   [input_dim, output_dim],
                                                   initializer=tf.compat.v1.truncated_normal_initializer(
                                                       stddev=math.sqrt(2.0 / (input_dim + output_dim))),
                                                   # 维度越大，数据越容易分布差异大，因此需要保持更小标准差，以求数据更稳定
                                                   dtype=input_layer.dtype)  # [iD, oD]
        # 这里的计算量非常大，例如dim=3200，一次context_gating 100亿次乘法，1024*3200*3200=10的10次方，100亿次乘法
        gates = tf.matmul(input_layer, gating_weights)  # [B, iD]->[B, oD]
        if self.bn:
            # 通过gate_weights权重网络得到gate后，为了保证gate_net的参数稳定，也需要过一层BN，稳定的gate_net参数网络，再给原始网络加权
            gates = bn_layer(gates, training, name="_".join((self.name, "bn")))
        else:
            gating_biases = tf.compat.v1.get_variable("_".join((self.name, "biased")),
                                                      shape=[output_dim],
                                                      initializer=tf.compat.v1.truncated_normal_initializer(
                                                          stddev=1.0 / math.sqrt(output_dim)),
                                                      dtype=input_layer.dtype)
            gates += gating_biases
        # 加权相乘后，数据膨胀，再次压缩回概率空间，统一权重的量纲
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
        # 等于通过两层的MLP，先加宽再还原，第一层会过dice激活函数
        middle_dim = 2 * input_dim  # 加宽了gate，实际上是增加了权重加权的维度，计算量增加很多，需要评估收益比
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
