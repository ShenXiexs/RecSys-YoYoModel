# -*- coding: utf-8 -*-
# @Time : 2025/11/13 18:11
# @Author : huangmian
# @File : stem.py
import tensorflow as tf
from layers.base import DNN
from layers.activation import get_activation


class StemLayer:
    def __init__(self, num_shared_experts, num_specific_experts, num_tasks, expert_hidden_units,
                 gate_hidden_units, hidden_activations='relu', net_dropout=0, batch_norm=False):
        self.num_shared_experts = num_shared_experts
        self.num_specific_experts = num_specific_experts
        self.num_tasks = num_tasks
        self.shared_experts = [DNN(hidden_units=expert_hidden_units,
                                   hidden_activations=hidden_activations,
                                   output_activation=None,
                                   dropout_rates=net_dropout,
                                   batch_norm=batch_norm) for _ in range(self.num_shared_experts)]
        self.specific_experts = [[DNN(hidden_units=expert_hidden_units,
                                      hidden_activations=hidden_activations,
                                      output_activation=None,
                                      dropout_rates=net_dropout,
                                      batch_norm=batch_norm) for _ in
                                  range(self.num_specific_experts)] for _ in range(num_tasks)]
        self.gate = [DNN(output_dim=num_specific_experts * num_tasks + num_shared_experts,
                         hidden_units=gate_hidden_units,
                         hidden_activations=hidden_activations,
                         output_activation=None,
                         dropout_rates=net_dropout,
                         batch_norm=batch_norm) for _ in range(self.num_tasks + 1)]
        self.gate_activation = get_activation('softmax')

    def __call__(self, x, return_gate=False):
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
        # gate 
        stem_outputs = []
        stem_gates = []
        for i in range(self.num_tasks + 1):
            if i < self.num_tasks:
                # for specific experts
                gate_input = []
                for j in range(self.num_tasks):
                    if j == i:
                        gate_input.extend(specific_expert_outputs[j])
                    else:
                        specific_expert_outputs_j = specific_expert_outputs[j]
                        specific_expert_outputs_j = [tf.stop_gradient(out) for out in specific_expert_outputs_j]
                        gate_input.extend(specific_expert_outputs_j)
                gate_input.extend(shared_expert_outputs)
                # (?, num_specific_experts*num_tasks+num_shared_experts, axis)
                gate_input = tf.stack(gate_input, axis=1)
                # (?, num_specific_experts*num_tasks+num_shared_experts)
                gate = self.gate_activation(self.gate[i](x[i] + x[-1]))
                if return_gate:
                    specific_gate = tf.reduce_mean(gate[:, :self.num_specific_experts * self.num_tasks], axis=0)
                    task_gate = tf.split(specific_gate, self.num_tasks)
                    specific_gate_list = []
                    for tg in task_gate:
                        specific_gate_list.append(tf.reduce_sum(tg))
                    shared_gate = tf.reduce_sum(tf.reduce_mean(gate[:, -self.num_shared_experts:], axis=0))
                    # (num_task+1,1)
                    target_task_gate = tf.reshape(tf.stack(specific_gate_list + [shared_gate], axis=0), [-1, 1])
                    assert len(target_task_gate) == self.num_tasks + 1
                    stem_gates.append(target_task_gate)
                stem_output = tf.reduce_sum(tf.expand_dims(gate, -1) * gate_input, axis=1)  # (?, axis)
                stem_outputs.append(stem_output)
            else:
                # for shared experts 
                gate_input = []
                for j in range(self.num_tasks):
                    gate_input.extend(specific_expert_outputs[j])
                gate_input.extend(shared_expert_outputs)
                # (?, num_specific_experts*num_tasks+num_shared_experts, axis)
                gate_input = tf.stack(gate_input, axis=1)
                # (?, num_specific_experts*num_tasks+num_shared_experts)
                gate = self.gate_activation(self.gate[i](x[-1]))
                stem_output = tf.reduce_sum(tf.expand_dims(gate, -1) * gate_input, axis=1)  # (?, axis)
                stem_outputs.append(stem_output)
        #
        if return_gate:
            return stem_outputs, stem_gates
        else:
            return stem_outputs
