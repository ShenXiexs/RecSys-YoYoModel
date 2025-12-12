# -*- coding: utf-8 -*-
# @Time : 2025/10/11 10:18
# @Author : huangmian
# @File : ple.py
import tensorflow as tf
from layers.activation import get_activation
from layers.base import DNN
logger = tf.compat.v1.logging


class CGCLayer:
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks,
                 expert_hidden_units, gate_hidden_units, hidden_activations,
                 net_dropout, batch_norm):
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = [DNN(hidden_units=expert_hidden_units,
                                   hidden_activations=hidden_activations,
                                   output_activation=None,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm) for _ in range(self.num_shared_experts)]
        self.specific_experts = [
            [DNN(hidden_units=expert_hidden_units,
                 hidden_activations=hidden_activations,
                 output_activation=None,
                 dropout_rates=net_dropout,
                 batch_norm=batch_norm) for _ in range(self.num_specific_experts)] for _ in range(num_tasks)]
        self.gate = [
            DNN(output_dim=num_specific_experts + num_shared_experts if i < num_tasks else num_shared_experts,
                hidden_units=gate_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm) for i in range(self.num_tasks + 1)]
        self.gate_activation = get_activation('softmax')


    def __call__(self, x, require_gate=False, training=False, **kwargs):
        """
        x: list, len(x)==num_tasks+1
        """
        specific_expert_outputs = []
        shared_expert_outputs = []
        # specific experts
        for i in range(self.num_tasks):
            task_expert_outputs = []
            for j in range(self.num_specific_experts):
                task_expert_outputs.append(self.specific_experts[i][j](x[i]))
            specific_expert_outputs.append(task_expert_outputs)
        # shared experts
        for i in range(self.num_shared_experts):
            shared_expert_outputs.append(self.shared_experts[i](x[-1]))
        logger.info(f"CGC.specific_expert_outputs={specific_expert_outputs}, shared_expert_outputs={shared_expert_outputs}")
        # gate
        cgc_outputs = []
        gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # for specific experts
                # gate_input: (?, num_specific_experts+num_shared_experts, dim)
                gate_input = tf.stack(specific_expert_outputs[i] + shared_expert_outputs, axis=1)
                gate = self.gate_activation(self.gate[i](x[i]))  # (?, num_specific_experts+num_shared_experts)
                gates.append(tf.reduce_mean(gate, axis=0))
                cgc_output = tf.reduce_sum(tf.expand_dims(gate, axis=-1) * gate_input, axis=1)  # (?, dim)
                cgc_outputs.append(cgc_output)
            else:
                # for shared experts
                gate_input = tf.stack(shared_expert_outputs, axis=1)  # (?, num_shared_experts, dim)
                gate = self.gate_activation(self.gate[i](x[-1]))  # (?, num_shared_experts)
                gates.append(tf.reduce_mean(gate, axis=0))
                cgc_output = tf.reduce_sum(tf.expand_dims(gate, axis=-1) * gate_input, axis=1)  # (?, dim)
                cgc_outputs.append(cgc_output)
            logger.info(f"CGC.task_{i}.gate_input={gate_input}, gate={gate}, cgc_output={cgc_output}")
        if require_gate:
            return cgc_outputs, gates
        else:
            return cgc_outputs


class PLE:
    def __init__(self,
                 num_tasks=1,
                 num_layers=1,
                 num_shared_experts=1,
                 num_specific_experts=1,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 use_tower_layers=False,
                 **kwargs):
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.use_tower_layers = use_tower_layers
        self.cgc_layers = [
            CGCLayer(num_shared_experts,
                     num_specific_experts,
                     num_tasks,
                     expert_hidden_units=expert_hidden_units,
                     gate_hidden_units=gate_hidden_units,
                     hidden_activations=hidden_activations,
                     net_dropout=net_dropout,
                     batch_norm=batch_norm) for _ in range(self.num_layers)]
        if self.use_tower_layers:
            self.tower = [DNN(
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=hidden_activations,
                output_activation=None,
                dropout_rates=net_dropout,
                batch_norm=batch_norm)
                for _ in range(num_tasks)]

    def __call__(self, feature_emb, training=False, **kwargs):
        cgc_inputs = [feature_emb for _ in range(self.num_tasks + 1)]
        for i in range(self.num_layers):
            logger.info(f"PLE.cgc_inputs_{i}={cgc_inputs}")
            cgc_outputs = self.cgc_layers[i](cgc_inputs)
            cgc_inputs = cgc_outputs
        logger.info(f"PLE.cgc_outputs={cgc_inputs}")
        if self.use_tower_layers:
            tower_output = [self.tower[i](cgc_outputs[i]) for i in range(self.num_tasks)]
        else:
            tower_output = cgc_outputs[:self.num_tasks]
        logger.info(f"PLE.tower_output={tower_output}")
        return tower_output
