import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

import tensorflow as tf
from layers.base import dnn, DNN


def gate_layer(unit, deep_feas, name):
    '''

    :param unit:对应专家网络数量
    :param deep_feas: 输入门控网络的特征
    :param name: 门控网络变量名
    :return:
    '''
    fea = tf.compat.v1.layers.dense(inputs=deep_feas, units=unit, name=name)
    return tf.nn.softmax(fea, axis=1)  # 门控网络最后一层神经元数量与专家数量保持一致,过一层softmax对专家进行融合选择


def expert_dnn(units, deep_feas, name, activation='relu'):
    mlp = dnn(units=units, prefix=name, activation=activation)
    return mlp(deep_feas)


def mmoe_layer(inputs, num_domains, num_experts, exprt_units):
    '''

    :param inputs:输入的embedding特征
    :param num_domains: 任务数量
    :param num_experts: 专家数量
    :param exprt_units: 专家网络结构
    :return: 返回mmoe网络对应的后续每个任务的输入
    '''

    expert_outlist = []
    for expert_id in range(num_experts):
        # inputs输入到第i个专家网络中
        expert_output = expert_dnn(exprt_units, inputs, name=f'expert_{expert_id}')  # (batch_size,expert_out_dim)
        expert_outlist.append(expert_output)
    expert_feas = tf.stack(expert_outlist, axis=1)  # (batch_size,num_experts,expert_out_dim)
    print('expert_outlist----', expert_feas)

    domain_input_list = []
    gate_units = num_experts
    for task_id in range(num_domains):
        # inputs输入到门控网络中
        gate_i = gate_layer(gate_units, inputs, f'gate_{task_id}')  # shape:(batch_size,num_experts)
        gate_i = tf.expand_dims(gate_i, -1)  # (batch_size,num_experts,1)
        print('gate_i--', gate_i)
        # 门控i对专家j输出进行加权，即将门控网络i与所有专家输出进行点乘
        domain_input = tf.multiply(expert_feas, gate_i)  # (batch_size,num_experts,expert_out_dim)
        # 用门控对所有专家加权后，进行求和得到每个任务的输入
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
