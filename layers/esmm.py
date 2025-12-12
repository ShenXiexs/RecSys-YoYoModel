#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class ESMM:
    def __init__(self, features, emb_lookuped, is_training=True):
        super(ESMM, self).__init__()
        self.is_training = is_training
        ######################_label################################
        batch_size = tf.shape(features["features"])[0]
        self.ctr_labels = [] # (batch_i,batch_size,1)
        self.ctcvr_labels = []

        _label = tf.cast(features['ctr_label'], tf.float32) if 'ctr_label' in features else tf.zeros((batch_size, ))
        self.ctr_labels.append(_label)

        _label = tf.cast(features['ctcvr_label'], tf.float32) if 'ctcvr_label' in features else tf.zeros((batch_size,))
        self.ctcvr_labels.append(_label)

        # ctr tower
        ctr_layers = self.build_layers([256, 128, 64, 32, 1],'ctr_tower')
        ctr_logits = self.build_task_tower(emb_lookuped,ctr_layers)

        # cvr tower
        cvr_layers = self.build_layers([256, 128, 64, 32, 1],'cvr_tower')
        cvr_logits = self.build_task_tower(emb_lookuped,cvr_layers)

        # ctcvr计算 pctcvr=pctr*pcvr
        # total loss  L(ctr,cvr)=loss(ctr_label,probs_ctr) + loss(ctcvr_label,probs_ctr*probs_cvr)
        self.losses = 0
        self.ctr_losses = 0
        self.ctcvr_losses = 0
        self.outputs = {}
        self.ctr_probs = []
        self.cvr_probs = []
        self.ctcvr_probs = []
        for i, (ctr_label, ctr_logit,ctcvr_label,cvr_logit) in enumerate(zip(self.ctr_labels,ctr_logits,self.ctcvr_labels,cvr_logits ), start=1):
            ctr_preds = tf.sigmoid(ctr_logit)
            cvr_preds = tf.sigmoid(cvr_logit)

            ctcvr_preds = tf.multiply(ctr_preds,cvr_preds)
            print(f"8888888888:  ctr_preds shape:{tf.shape(ctr_preds)}, cvr_preds shape:{cvr_preds} ctcvr_preds shape:{tf.shape(ctcvr_preds)}")

            ctr_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=ctr_label,output=ctr_preds))
            ctcvr_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(target=ctcvr_label,output=ctcvr_preds))

            loss = ctr_loss + ctcvr_loss
            self.ctr_losses += ctr_loss
            self.ctcvr_losses += ctcvr_loss
            self.losses += loss
            tf.summary.scalar("total_loss%s" % i, loss)
            tf.summary.scalar("ctr_loss%s" % i, ctr_loss)
            tf.summary.scalar("ctcvr_loss%s" % i, ctcvr_loss)
            self.outputs["output%s" % i] =  tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in [ctr_preds,cvr_preds,ctcvr_preds]], axis=1), axis=1)
            #self.outputs["output%s" % i] = [ctr_preds,cvr_preds,ctcvr_preds]
            self.ctr_probs.append(ctr_preds)
            self.cvr_probs.append(cvr_preds)
            self.ctcvr_probs.append(ctcvr_preds)

        self.predictions = {
            "requestid": features["requestid"] if "requestid" in features else tf.as_string(tf.zeros((batch_size,), tf.int16)),
            "combination_un_id": features["combination_un_id"] if "combination_un_id" in features else tf.as_string(tf.zeros((batch_size,), tf.int16)),
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.ctr_labels + self.ctcvr_labels + self.ctr_probs + self.cvr_probs+ self.ctcvr_probs], axis=1), axis=1)
        } # a = tf.reshape(x, [-1, 1]) shape(batch_size,1) tf.concat(a,axis=1) shape:(batch_size,2)
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
            if i < max(len(units) - 2, 0):
               dropout = tf.keras.layers.Dropout(0.2)
               layers.append(dropout)
        return tf.keras.Sequential(layers)

    def build_task_tower(self,features,task_layer):
        outs = [task_layer(features)]
        logits = []
        for i, logit in enumerate(outs, start=1):
            logit = tf.clip_by_value(tf.reshape(logit, (-1,)), -15, 15)
            logits.append(logit)
        return logits



