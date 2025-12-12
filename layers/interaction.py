import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, root_dir)

from layers.activation import get_activation
from layers.normalization import PartitionedNormalization
from tensorflow import Tensor
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.keras.initializers import Zeros, Constant
from layers.base import dnn

def dcn_cross_layer(x_0,x_i,name):
    '''
    dcn中交叉层具体实现
    :param x_0: 输入的数据 shape:(batch_size,input_size)
    :param x_i: 上一层的输出 shape:(batch_size,input_size)
    :param name:
    :return:
    '''
    input_dim = x_0.shape[1]
    xw_i = tf.compat.v1.layers.dense(x_i, units=1, name=f'cross_layer_{name}') # sum(xi*wi)
    bias_i = tf.compat.v1.get_variable(shape=(input_dim), name=f'corss_bias_{name}') # 偏置项
    cross_v = tf.add(x_0 * xw_i, bias_i)  # x0*xw + bias
    x_i = tf.add(cross_v, x_i)
    return x_i


def dcn_cross_v1(feas_embed:Tensor,num_cross_layers,task_name):
    '''

    :param feas_embed: shape:(batch_size,input_size)
    :param num_cross_layers: 交叉层数
    :return:
    '''
    input_dim = feas_embed.shape[-1] # shape:(batch_size,input_dim)
    x0 = feas_embed
    x_i = feas_embed
    for i in range(num_cross_layers):
        # xw_i = tf.compat.v1.layers.dense(x_i,units=1,name=f'cross_layer_{i}')
        # bias_i = tf.compat.v1.get_variable(shape=(input_dim),name=f'corss_bias_{i}')
        # cross_v = tf.add(x0*xw_i,bias_i) # x0*xw + bias
        # x_i = tf.add(cross_v,x_i)
        x_i = dcn_cross_layer(x0,x_i,f'{task_name}_{i}')
    return x_i


def regulation_module(inputs,temperature=1):
    '''
   Input shape
       - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

     Output shape
       - 2D tensor with shape: ``(batch_size,field_size * embedding_size)``.
    '''
    embed_size = inputs.shape[2]
    field_size = inputs.shape[1]
    gw = tf.compat.v1.get_variable(shape=(1, field_size, 1), initializer=tf.ones_initializer(), name='regulation_module_field_weight',dtype=tf.float32) # 创建每个特征域对应的权重参数
    # 使用softmax进行归一化，统一权重量纲
    field_gating_score = tf.nn.softmax(1/temperature*gw,axis=1) # shape:(1,field_size,1)
    # 对每个特征域的embedding向量进行用feild_gating_score进行加权
    E = field_gating_score * inputs # shape:(batch_size,field_size,embed_size)

    return tf.reshape(E, [-1, field_size * embed_size]) # shape:(batch_size,field_size * embed_size)


def bridge_module(cross_x,hidden_x):
    '''
    EDCN模型中的桥接模块
    :param cross_x:shape:(batch_size,field_size*embedding_size)
    :param hidden_x:(batch_size,field_size*embedding_size)
    :return:
    '''
    # 桥接方式采用hadamard product
    return tf.multiply(cross_x,hidden_x)


class StarTopologyFCN(Layer):
    def __init__(self,domain_types,hidden_units,activation_name='relu',dropout_rate=0.,
                 l2_reg=0,use_pn=False,is_traing=True):
        super(StarTopologyFCN,self).__init__()
        self.domain_types = domain_types
        self.hidden_units = hidden_units
        self.activation_name = activation_name
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_pn = use_pn
        self.is_training = is_traing
        self.activation_dict = {}
        self.dropout_dict = {}
        for dtype in self.domain_types:
            self.activation_dict[dtype] = [get_activation(self.activation_name) for i in range(len(self.hidden_units))]
            self.dropout_dict[dtype] = [tf.keras.layers.Dropout(self.dropout_rate) for i in range(len(self.hidden_units))]

        if self.use_pn:
            self.pn_list = [
                PartitionedNormalization(self.domain_types, name=f'pn_star_layer_{i}_topology_fcn') for i in range(len(self.hidden_units))
            ]

    def build(self,input_shape):
        fea_size = input_shape['features'][-1]
        tmp_hidden_units = self.hidden_units.copy()
        # 连接输入层的参数
        tmp_hidden_units.insert(0,fea_size)
        # 连接第i层和i+1层神经元的参数矩阵
        self.share_w_list = [ self.add_weight(name=f'domain_share_{i}_neural_w',
                                              shape=[tmp_hidden_units[i],tmp_hidden_units[i+1]],
                                              initializer='glorot_uniform',
                                              regularizer=l2(self.l2_reg),
                                              trainable=True
                            ) for i in range(len(tmp_hidden_units)-1)
        ]
        self.share_bias_list = [
            self.add_weight(name=f'domain_share_{i}_neural_bias',
                            shape=[tmp_hidden_units[i+1]],
                            initializer=Zeros(),
                            trainable=True
            ) for i in range(len(tmp_hidden_units)-1)
        ]

        self.domain_w_list = {}
        self.domain_bias_list = {}

        for dtype in self.domain_types:
            self.domain_w_list[dtype] = [self.add_weight(name=f'domain_spec_{i}_neural_w',
                            shape=[tmp_hidden_units[i], tmp_hidden_units[i + 1]],
                            initializer='glorot_uniform',
                            regularizer=l2(self.l2_reg),
                            trainable=True
                            ) for i in range(len(tmp_hidden_units) - 1)
            ]
            self.domain_bias_list[dtype] = [
                self.add_weight(name=f'domain_spec_{i}_neural_bias',
                                shape=[tmp_hidden_units[i + 1]],
                                initializer=Zeros(),
                                trainable=True
                                ) for i in range(len(tmp_hidden_units) - 1)
            ]

    def call(self,inputs):
        features = inputs['features']
        domain_outpus = {}
        for dtype in self.domain_types:
            output = features
            print(f'start process domain:{dtype}')
            for i in range(len(self.hidden_units)):
                # 获取共享domain对应的权重
                share_w = self.share_w_list[i]
                share_bias = self.share_bias_list[i]
                # 获取domain对应的权重
                domain_w = self.domain_w_list[dtype][i]
                domain_bias = self.domain_bias_list[dtype][i]
                # w_shre*w_domain*x + bias_share+bias_domain
                print(f'=====> domain {dtype},layer:{i} domain_w shape:{domain_w.shape}')
                print(f'=====> domain {dtype},layer:{i} share_w shape:{share_w.shape}')
                print(f'=====> domain {dtype},layer:{i} domain_bias shape:{domain_bias.shape}')
                print(f'=====> domain {dtype},layer:{i} share_bias shape:{share_bias.shape}')
                print(f"=====> domain {dtype},layer:{i} inputs shape:{output.shape}")
                print(f'=====> domain {dtype},layer:{i} weight matul shape:{tf.matmul(output,domain_w*share_w).shape}')
                output = tf.matmul(output,domain_w*share_w) + domain_bias+share_bias
                # pn->activation->dropout
                if self.use_pn:
                    print(f'=====> domain {dtype},layer:{i} pn features shape:{output.shape}, domain_types shape:{inputs["domain_types"].shape} ')
                    output = self.pn_list[i]({'features':output,'domain_types':inputs['domain_types']},self.is_training)
                output = self.activation_dict[dtype][i](output)
                output = self.dropout_dict[dtype][i](output)

            domain_outpus[dtype] = output
        return domain_outpus



if __name__ == '__main__':

    # 测试dcn_cross_v1
    x:Tensor = tf.constant([[0.2,0.3,1.2],[0.1,0.12,1.5]],dtype=tf.float32) # (batch_size,input_dim)
    with tf.compat.v1.Session() as sess:
        print(f"x shape:{x.shape},x_value:{sess.run(x)}")
        num_cross_layer = 4
        cross_logit = dcn_cross_v1(x,num_cross_layer,'dcn')
        dnnlogit = dnn([256, 128, 64, 32, 1], "task1")(x)
        print(f"dcn cross shape:{cross_logit.shape} ")
        print(f"dnnlogit shape:{dnnlogit.shape} ")

        final_logit = tf.concat([cross_logit,dnnlogit],axis=1)
        print(f"final_logit1 shape:{final_logit.shape} ")

        final_logit = tf.compat.v1.layers.dense(final_logit,1)
        print(f"final_logit2 shape:{final_logit.shape} ")

        # print(f"dcn cross shape:{cross_logit.shape}, cross value:{sess.run(cross_logit)} ")
        #print(f"dcn cross shape:{dnnlogit.shape}, cross value:{sess.run(dnnlogit)} ")

        #tf.concat([cross_logit,dnnlogit])


        bridge_x = bridge_module(x,x)
        print(f"bridge_x shape:{bridge_x.shape}")

        reg_module = tf.compat.v1.get_variable(shape=(1, 4, 1),initializer=tf.ones_initializer() ,name='regulation_module_field_weight',dtype=tf.float32)

        feild_gating_score = tf.nn.softmax(reg_module * 1, 1) # 归一化，统一量纲
        print('feild_gating_score',feild_gating_score)

        x = tf.constant([0.1,0.3,0.9,1.2, 5,6,4,7, 5,0,1.4,4.7]) # 2*2*3
        x = tf.reshape(x,[2,2,3])


