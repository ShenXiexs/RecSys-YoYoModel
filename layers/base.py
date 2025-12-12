import tensorflow as tf
from tensorflow.keras.layers import Layer
from layers.activation import get_activation
logger = tf.compat.v1.logging

def dnn(units, prefix, activation="relu"):
    layers = []
    for i, unit in enumerate(units):
        name = 'dnn_hidden_%s_%d' % (prefix, i)
        layer = tf.keras.layers.Dense(
            units=unit, activation=get_activation(activation),
            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
            name=name
        )
        layers.append(layer)
        # if i < max(len(units) - 2, 0):
        #    dropout = tf.keras.layers.Dropout(0.2)
        #    layers.append(dropout)
    return tf.keras.Sequential(layers)


class MLP(Layer):
    def __init__(self,
                 hidden_units=None,
                 hidden_activations="relu",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 bn_only_once=False,  # Set True for inference speed up
                 output_kernel_initializer=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 use_bias=True):
        super().__init__()
        self.output_layer = None
        self.output_activation = None
        if hidden_units is None or len(hidden_units) == 0:
            raise ValueError("hidden_units 不能为空，请在 dnn_config 中配置至少一个隐藏层维度")
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        if batch_norm and bn_only_once:
            dense_layers.append(tf.keras.layers.BatchNormalization())
        for idx in range(len(hidden_units)):
            dense_layers.append(tf.keras.layers.Dense(hidden_units[idx],
                                                      use_bias=use_bias,
                                                      kernel_initializer=kernel_initializer,
                                                      bias_initializer=bias_initializer))
            if batch_norm and not bn_only_once:
                dense_layers.append(tf.keras.layers.BatchNormalization())
            if hidden_activations[idx]:
                dense_layers.append(tf.keras.layers.Activation(hidden_activations[idx]))
            if dropout_rates[idx] > 0:
                dense_layers.append(tf.keras.layers.Dropout(dropout_rates[idx]))
        if output_dim is not None:
            dense_layers.append(tf.keras.layers.Dense(output_dim,
                                                      use_bias=use_bias,
                                                      kernel_initializer=output_kernel_initializer,
                                                      bias_initializer=bias_initializer))
        if output_activation is not None:
            dense_layers.append(tf.keras.layers.Activation(output_activation))
        # self.mlp = tf.keras.Sequential(dense_layers)  # * used to unpack list
        self.mlp = dense_layers

    def call(self, inputs, training=False, **kwargs):
        # return self.mlp(inputs)
        for i, layer in enumerate(self.mlp):
            logger.info(f"mlp-{i}-inputs={inputs}")
            inputs = layer(inputs, training=training)
        return inputs

class DNN:
    def __init__(self,
                 hidden_units=None,
                 hidden_activations="relu",
                 output_dim=None,
                 output_activation=None,
                 dropout_rates=0.0,
                 batch_norm=False,
                 bn_only_once=False,  # Set True for inference speed up
                 output_kernel_initializer=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 use_bias=True):
        self.output_layer = None
        self.output_activation = None
        if hidden_units is None or len(hidden_units) == 0:
            raise ValueError("hidden_units 不能为空，请在 dnn_config 中配置至少一个隐藏层维度")
        dense_layers = []
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        if batch_norm and bn_only_once:
            dense_layers.append(tf.keras.layers.BatchNormalization())
        for idx in range(len(hidden_units)):
            dense_layers.append(tf.keras.layers.Dense(hidden_units[idx],
                                                      use_bias=use_bias,
                                                      kernel_initializer=kernel_initializer,
                                                      bias_initializer=bias_initializer,
                                                      activation=get_activation(hidden_activations[idx])))
            if batch_norm and not bn_only_once:
                dense_layers.append(tf.keras.layers.BatchNormalization())
            if dropout_rates[idx] > 0:
                dense_layers.append(tf.keras.layers.Dropout(dropout_rates[idx]))
        if output_dim is not None:
            self.output_layer = tf.keras.layers.Dense(output_dim,
                                                      use_bias=use_bias,
                                                      kernel_initializer=output_kernel_initializer,
                                                      bias_initializer=bias_initializer)
        if output_activation is not None:
            self.output_activation = tf.keras.layers.Activation(output_activation)
        # self.mlp = tf.keras.Sequential(dense_layers)  # * used to unpack list
        self.mlp = dense_layers

    def __call__(self, inputs, training=False, **kwargs):
        # return self.mlp(inputs)
        for i, layer in enumerate(self.mlp):
            logger.info(f"mlp-{i}-inputs={inputs}")
            inputs = layer(inputs, training=training)
        hidden_output = inputs
        if self.output_layer:
            inputs = self.output_layer(inputs, training=training)
        if self.output_activation:
            inputs = self.output_activation(inputs, training=training)
        if kwargs.get("return_hidden", False):
            return hidden_output, inputs
        return inputs
