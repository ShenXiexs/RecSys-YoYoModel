#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from layers.mmoe import mmoe_layer
from layers.interaction import dcn_cross_v1


class DNN:
    def __init__(self, features, mmoe_input,dcn_input,key_fea_input, is_training=True):
        super(DNN, self).__init__()
        self.is_training = is_training
        ######################_label################################
        batch_size = tf.shape(features["features"])[0]
        self.ctr_labels = [] # (batch_i,batch_size,1)

        _label = tf.cast(features['ctr_label'], tf.float32) if 'ctr_label' in features else tf.zeros((batch_size, ))
        self.ctr_labels.append(_label)


        # mmoe layer,广告场景分类：联盟，信息流，推荐，搜索
        mmoe_task_outlist = mmoe_layer(mmoe_input, num_domains=1, num_experts=4, exprt_units=[128, 64, 128])

        ctr_logits = self.build_tower('ctr',mmoe_task_outlist[0],dcn_input['ctr_dcn'],key_fea_input['ctr_key_feature'])

        # ctcvr计算 pctcvr=pctr*pcvr
        # total loss  L(ctr,cvr)=loss(ctr_label,probs_ctr) + loss(ctcvr_label,probs_ctr*probs_cvr)
        self.ctr_losses = 0
        self.outputs = {}
        self.ctr_probs = []

        for i, (ctr_label, ctr_logit) in enumerate(zip(self.ctr_labels,ctr_logits ), start=1):
            ctr_preds = tf.sigmoid(ctr_logit)

            print(f"8888888888:  ctr_preds shape:{tf.shape(ctr_preds)}")

            ctr_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=ctr_label,output=ctr_preds))

            loss = ctr_loss
            self.ctr_losses += ctr_loss
            tf.summary.scalar("ctr_loss%s" % i, ctr_loss)
            self.outputs["output%s" % i] =  tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in [ctr_preds]], axis=1), axis=1)
            self.ctr_probs.append(ctr_preds)

        self.predictions = {
            "requestid": features["requestid"] if "requestid" in features else tf.as_string(tf.zeros((batch_size,), tf.int16)),
            "combination_un_id": features["combination_un_id"] if "combination_un_id" in features else tf.as_string(tf.zeros((batch_size,), tf.int16)),
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.ctr_labels  + self.ctr_probs], axis=1), axis=1)
        }
        print("======",self.ctr_labels)

    def build_layers(self, units, prefix, activation=None):
        act = "relu"
        layers = []
        for i, unit in enumerate(units):
            name = 'dnn_hidden_%s_%d' % (prefix, i)
            if i == len(units) - 1:
                act = activation
            layer = tf.keras.layers.Dense(
                units=unit, activation=act,
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name=name
            )
            layers.append(layer)
            #if i < max(len(units) - 2, 0):
            #    dropout = tf.keras.layers.Dropout(0.2)
            #    layers.append(dropout)
        return tf.keras.Sequential(layers)

    def build_tower(self,task_type,mmoe_out,dcn_input,key_fea_input):
        # dcn部分
        dcn_out = dcn_cross_v1(dcn_input, num_cross_layers=3,task_name=f'{task_type}_dcn_cross')
        # dnn
        layers = self.build_layers([64, 32, 16], f'{task_type}_task')
        task_out = layers(mmoe_out)
        # 将ctr dcn,ctr任务，ctr关键特征拼接到一起
        final_input = tf.concat([task_out, dcn_out, key_fea_input],axis=1)

        print(f'{task_type} build_esmm_tower =====>final_input shape:{tf.shape(final_input)}')
        logits = tf.compat.v1.layers.dense(inputs=final_input, units=1, name=f'{task_type}_final_input')
        logits = [tf.clip_by_value(tf.reshape(logits, (-1,)), -15, 15)]
        return logits


