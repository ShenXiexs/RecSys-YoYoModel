import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

import tensorflow as tf
from layers.base import dnn, DNN


def gate_layer(unit, deep_feas, name):
    '''

    :param unit: number of expert networks
    :param deep_feas: features fed into the gating network
    :param name: gating network variable name
    :return:
    '''
    fea = tf.compat.v1.layers.dense(inputs=deep_feas, units=unit, name=name)
    return tf.nn.softmax(fea, axis=1)  # Last gate layer size matches expert count; apply softmax to blend experts


def expert_dnn(units, deep_feas, name, activation='relu'):
    mlp = dnn(units=units, prefix=name, activation=activation)
    return mlp(deep_feas)


def mmoe_layer(inputs, num_domains, num_experts, exprt_units):
    '''

    :param inputs: input embedding features
    :param num_domains: number of tasks
    :param num_experts: number of experts
    :param exprt_units: expert network structure
    :return: per-task inputs produced by the MMoE network
    '''

    expert_outlist = []
    for expert_id in range(num_experts):
        # Feed inputs into the i-th expert network
        expert_output = expert_dnn(exprt_units, inputs, name=f'expert_{expert_id}')  # (batch_size,expert_out_dim)
        expert_outlist.append(expert_output)
    expert_feas = tf.stack(expert_outlist, axis=1)  # (batch_size,num_experts,expert_out_dim)
    print('expert_outlist----', expert_feas)

    domain_input_list = []
    gate_units = num_experts
    for task_id in range(num_domains):
        # Feed inputs into the gating network
        gate_i = gate_layer(gate_units, inputs, f'gate_{task_id}')  # shape:(batch_size,num_experts)
        gate_i = tf.expand_dims(gate_i, -1)  # (batch_size,num_experts,1)
        print('gate_i--', gate_i)
        # Gate i weights expert outputs by element-wise multiplication
        domain_input = tf.multiply(expert_feas, gate_i)  # (batch_size,num_experts,expert_out_dim)
        # Sum gated expert outputs to get per-task inputs
        domain_input = tf.reduce_sum(domain_input, axis=1)  # (batch_size,expert_out_dim)
        print('--domain_input', domain_input)
        domain_input_list.append(domain_input)
    return domain_input_list


class MMoE:
    def __init__(self, num_experts, num_domains, expert_hidden_units, gate_hidden_units,
                 hidden_activations='relu', net_dropout=0, batch_norm=False):
        self.num_experts = num_experts
        self.num_tasks = num_domains
        self.experts = tf.keras.Sequential([DNN(hidden_units=expert_hidden_units,
                                                hidden_activations=hidden_activations,
                                                output_activation=None,
                                                dropout_rates=net_dropout,
                                                batch_norm=batch_norm) for _ in range(self.num_experts)])
        self.gate = tf.keras.Sequential([DNN(hidden_units=gate_hidden_units,
                                             output_dim=num_experts,
                                             hidden_activations=hidden_activations,
                                             output_activation=None,
                                             dropout_rates=net_dropout,
                                             batch_norm=batch_norm) for _ in range(self.num_tasks)])
        self.gate_activation = 'softmax'

    def __call__(self, x):
        # (?, num_experts, dim)
        experts_output = tf.stack([self.experts[i](x) for i in range(self.num_experts)], axis=1)
        mmoe_output = []
        for i in range(self.num_tasks):
            gate_output = self.gate[i](x)
            if self.gate_activation is not None:
                gate_output = tf.nn.softmax(gate_output, axis=1)  # (?, num_experts)
            mmoe_output.append(tf.reduce_sum(tf.multiply(tf.expand_dims(gate_output, -1), experts_output), axis=1))
        return mmoe_output


if __name__ == "__main__":
    batch_input = tf.random.uniform(shape=[32, 64], minval=0, maxval=1)
    num_tasks = 2
    exprt_units = [32, 16, 10]
    feat_val, id_idx = tf.unique(tf.reshape(batch_input, (-1,)))
    print('===feat_val:', feat_val)
    print("===fea_idx:", id_idx)

    domain_input_list = mmoe_layer(batch_input, num_domains=num_tasks, num_experts=3, exprt_units=exprt_units)
    print('domain_input_list:', domain_input_list)
