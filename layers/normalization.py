import random
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros, Constant
import tensorflow as tf


class PartitionedNormalization(Layer):
    def __init__(self,domain_types,name=None):
        super(PartitionedNormalization,self).__init__(name=name)
        self.domain_types = domain_types
        self.bn_list = {
            dtype:tf.keras.layers.BatchNormalization(center=False, scale=False, name=f"bn_{dtype}")
            for dtype in self.domain_types
        }


    def build(self, input_shape):
        '''
        创建模型参数权重，调用call时被自动调用
        # input_shape {'features': TensorShape([batch_size, feature_size]), 'domain_types': TensorShape([batch_size, 1])}
        :return:
        '''
        self.global_gamma = self.add_weight(name='global_normal_gamma',shape=[1],initializer=Constant(0.5))
        self.global_beta = self.add_weight(name='global_normal_beta',shape=[1],initializer=Zeros())

        self.domain_gammas = {
            dtype: self.add_weight(name=f'domain_{dtype}_normal_gamma',shape=[1],initializer=Constant(0.5))
            for dtype in self.domain_types
        }
        self.domain_betas = {
            dtype: self.add_weight(name=f'domain_{dtype}_normal_beta', shape=[1], initializer=Zeros())
            for dtype in self.domain_types
        }

    def call(self,inputs,training=None):
        '''
        定义层的前向传播
        :param inputs:字典类型，包含输入的特征和每个样本对应的domain类型；
            获取输入特征features = inputs['features']
            获取对应的domain = inputs['domain_types']
        :param training:
        :return:
        '''
        # 获取对应domain对应的样本
        features = inputs['features']
        batch_datas = tf.zeros(shape=features.shape)
        for dtype in self.domain_types:
            # 获取domain对应的样本index
            domain_sample_idxs = tf.reshape(tf.equal(inputs['domain_types'], dtype), shape=[-1])
            # 获取domain对应的样本
            domain_data = tf.boolean_mask(features, domain_sample_idxs)
            bn = self.bn_list[dtype]
            print(f'domain {dtype} pn process,domain_data shape:{domain_data.shape},original feature_data shape:{features.shape}')
            domain_data = bn(domain_data,training=training)
            # 获取gamma和beta，对归一化后的数据进行放缩
            domain_data = (self.global_gamma * self.domain_gammas[dtype])*domain_data+self.global_beta+self.domain_betas[dtype]
            # 将数据放回到原来位置
            domain_data = tf.scatter_nd(tf.where(domain_sample_idxs), domain_data, features.shape)
            batch_datas += domain_data
        return batch_datas


def bn_layer(bn_in, training, name="bn", **kv):
    # 对这个batch内的参数做归一化，避免参数波动过大导致模型的loss难以收敛
    # 看了下，我们在处理过程中，无论是context_gating出来还是fc出来后先过一层BN
    # 保证参数的稳定性
    layer = tf.keras.layers.BatchNormalization(name=name, **kv)
    res = layer(bn_in, training=training)
    return res

if __name__ == "__main__":
    batch_size = 10
    feature_size = 3
    domain_types = ['search','rec']
    features = tf.random.uniform(shape=[batch_size,feature_size],minval=0,maxval=1,dtype=tf.float16)
    domain_idxs = tf.constant([[random.choice(domain_types)] for _ in range(batch_size)],shape=(batch_size,1),dtype=tf.string)
    inputs = {
        "features":features,
        "domain_types":domain_idxs
    }
    pn = PartitionedNormalization(domain_types)
    outs = pn(inputs,training=True)
    print('--------outs:',outs)
