# models/rankmixer_main.py
# RankMixer Estimator: 重写版，参考 RankMixer 论文 + HSTU/TOP5 输入处理
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict
import math

from models.ctr_hstu_seq import LayerNorm
from common.utils import select_feature, train_config as TrainConfig, seq_features_config
from common.metrics import evaluate

logger = tf.compat.v1.logging


# 若 TrainConfig 内未显式指定 seq_length，则根据 seq_features_config 自动生成
if not hasattr(TrainConfig, "seq_length"):
    TrainConfig.seq_length = OrderedDict(
        (cfg["name"], cfg["length"])
        for cfg in seq_features_config
        if cfg.get("is_download", 1) == 1
    )


def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3))))


class MultiHeadTokenMixer(tf.layers.Layer):
    """可学习 Token Mixing（非论文原文）。"""
    def __init__(self, num_tokens, d_model, num_heads=8, dropout=0.0, name=None):
        super(MultiHeadTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能整除 num_heads, got d_model=%d num_heads=%d" %
                             (self.d_model, self.num_heads))
        self.d_head = self.d_model // self.num_heads
        init = tf.variance_scaling_initializer(scale=2.0)
        self.mix_weight = self.add_weight(
            "mix_weight",
            shape=[self.num_heads, self.num_tokens, self.num_tokens],
            initializer=init
        )
        super(MultiHeadTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        # x: [B,T,D]
        B = tf.shape(x)[0]
        T = self.num_tokens
        h = self.num_heads
        d = self.d_head
        x_heads = tf.reshape(x, [B, T, h, d])
        x_heads = tf.transpose(x_heads, [0, 2, 1, 3])  # [B,h,T,d]
        mixed = tf.einsum("bhtd,hkt->bhkd", x_heads, self.mix_weight)
        mixed = tf.transpose(mixed, [0, 2, 1, 3])      # [B,T,h,d]
        mixed = tf.reshape(mixed, [B, T, h * d])
        if self.dropout and training:
            mixed = tf.nn.dropout(mixed, keep_prob=1.0 - self.dropout)
        return mixed


class PaperMultiHeadTokenMixer(tf.layers.Layer):
    """
    论文原文 Multi-Head Token Mixing：参数无关的“Split + Shuffle + Merge”。

    设输入 X ∈ R^{B×T×D}，令 H = T 且 D % H == 0。
    - SplitHead:  X -> [B, T, H, D/H]
    - Shuffle:    transpose -> [B, H, T, D/H]
    - Merge:      reshape -> [B, H(=T), T*(D/H)=D]

    这样可保持 token 数不变，方便残差连接。
    """
    def __init__(self, num_tokens, d_model, dropout=0.0, name=None):
        super(PaperMultiHeadTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.d_model % self.num_tokens != 0:
            raise ValueError(
                "论文 TokenMixing 需要 d_model % num_tokens == 0, got d_model=%d num_tokens=%d" %
                (self.d_model, self.num_tokens)
            )
        self.d_head = self.d_model // self.num_tokens
        super(PaperMultiHeadTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        # x: [B,T,D]
        B = tf.shape(x)[0]
        T = self.num_tokens
        d = self.d_head
        x4 = tf.reshape(x, [B, T, T, d])          # [B, T(tokens), H(=T), d]
        x4 = tf.transpose(x4, [0, 2, 1, 3])       # [B, H, T, d]
        y = tf.reshape(x4, [B, T, T * d])         # [B, T, D]
        if self.dropout and training:
            y = tf.nn.dropout(y, keep_prob=1.0 - self.dropout)
        return y


class PerTokenFFN(tf.layers.Layer):
    """每个 token 拥有独立 FFN，建模异构 slot。"""
    def __init__(self, num_tokens, d_model, mult=4, dropout=0.0, name=None):
        super(PerTokenFFN, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.dropout = float(dropout)

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight("W1", [self.num_tokens, self.d_model, hidden_dim], initializer=init)
        self.b1 = self.add_weight("b1", [self.num_tokens, hidden_dim], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight("W2", [self.num_tokens, hidden_dim, self.d_model], initializer=init)
        self.b2 = self.add_weight("b2", [self.num_tokens, self.d_model], initializer=tf.zeros_initializer())
        super(PerTokenFFN, self).build(input_shape)

    def call(self, x, training=False):
        h = tf.einsum("btd,tdh->bth", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        y = tf.einsum("bth,thd->btd", h, self.W2) + self.b2
        if self.dropout and training:
            y = tf.nn.dropout(y, keep_prob=1.0 - self.dropout)
        return y


class RankMixerBlock(tf.layers.Layer):
    """标准 RankMixer Block: LN -> TokenMixer -> Residual -> LN -> PerTokenFFN -> Residual。"""
    def __init__(self, num_tokens, d_model, num_heads, ffn_mult, token_mixer_type="paper",
                 token_dp=0.0, ffn_dp=0.0, name=None):
        super(RankMixerBlock, self).__init__(name=name)
        self.ln1 = LayerNorm(name="ln1")
        token_mixer_type = str(token_mixer_type).lower()
        if token_mixer_type in ("paper", "shuffle", "rankmixer"):
            self.token_mixer = PaperMultiHeadTokenMixer(
                num_tokens=num_tokens, d_model=d_model, dropout=token_dp, name="token_mixer"
            )
        elif token_mixer_type in ("learned", "param", "mlp"):
            self.token_mixer = MultiHeadTokenMixer(
                num_tokens=num_tokens, d_model=d_model, num_heads=num_heads, dropout=token_dp, name="token_mixer"
            )
        else:
            raise ValueError("Unknown token_mixer_type: %s" % token_mixer_type)
        self.ln2 = LayerNorm(name="ln2")
        self.per_token_ffn = PerTokenFFN(num_tokens=num_tokens, d_model=d_model,
                                         mult=ffn_mult, dropout=ffn_dp, name="per_token_ffn")

    def call(self, x, training=False):
        y = self.ln1(x)
        y = self.token_mixer(y, training=training)
        x = x + y
        z = self.ln2(x)
        z = self.per_token_ffn(z, training=training)
        return x + z


class RankMixerEncoder(tf.layers.Layer):
    """堆叠 RankMixerBlock。"""
    def __init__(self, num_layers, num_tokens, d_model, num_heads, ffn_mult,
                 token_mixer_type="paper", token_dp=0.0, ffn_dp=0.0, name=None):
        super(RankMixerEncoder, self).__init__(name=name)
        self.blocks = [
            RankMixerBlock(num_tokens=num_tokens, d_model=d_model, num_heads=num_heads,
                           ffn_mult=ffn_mult, token_mixer_type=token_mixer_type,
                           token_dp=token_dp, ffn_dp=ffn_dp, name="block_%d" % i)
            for i in range(num_layers)
        ]
        self.final_ln = LayerNorm(name="encoder_ln")

    def call(self, x, training=False):
        out = x
        for blk in self.blocks:
            out = blk(out, training=training)
        return self.final_ln(out)


def _dense_if_sparse(x, default_value=""):
    return tf.sparse.to_dense(x, default_value=default_value) if isinstance(x, tf.SparseTensor) else x


def _pad_trunc_to_length(tokens_dense, L):
    T = tf.shape(tokens_dense)[1]
    tokens_cut = tokens_dense[:, :tf.minimum(T, L)]
    pad_len = tf.maximum(0, L - tf.shape(tokens_cut)[1])
    tokens_fix = tf.pad(tokens_cut, paddings=[[0, 0], [0, pad_len]])
    tokens_fix.set_shape([None, L])
    return tokens_fix


def _get_seq_embedding(tokens_2d, embeddings_table, policy, name="seq_lookup"):
    B = tf.shape(tokens_2d)[0]
    L = tf.shape(tokens_2d)[1]
    flat = tf.reshape(tokens_2d, [-1])
    uniq, idx = tf.unique(flat)
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])
    update_tstp_op = policy.apply_update(ids)
    restrict_op = policy.apply_restriction(int(1e8))
    emb_u, _ = de.embedding_lookup(embeddings_table, ids, return_trainable=True, name=name)
    gathered = tf.gather(emb_u, idx)
    seq_emb = tf.reshape(gathered, [B, L, embeddings_table.dim])
    return seq_emb, update_tstp_op, restrict_op


def _get_dense_emb_from_features(features, embeddings_table, policy):
    x = features["features"]
    x = _dense_if_sparse(x, default_value="")
    B = tf.shape(x)[0]
    fea_size = len(select_feature)
    flat = tf.reshape(x, [-1])
    uniq, idx = tf.unique(flat)
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])
    update_tstp_op = policy.apply_update(ids)
    restrict_op = policy.apply_restriction(int(1e8))
    emb_u, _ = de.embedding_lookup(embeddings_table, ids, return_trainable=True, name="features_lookup")
    gathered = tf.gather(emb_u, idx)
    dense_emb = tf.reshape(gathered, [B, fea_size * embeddings_table.dim])
    return dense_emb, update_tstp_op, restrict_op


def _sequence_pool(seq_emb, tokens, mode="mean"):
    """支持 mean/max/target 等模式的 pooling。"""
    mode = str(mode).lower()
    pad_mask = tf.logical_or(tf.equal(tokens, ""), tf.equal(tokens, "0"))
    valid = tf.cast(tf.logical_not(pad_mask), tf.float32)
    denom = tf.reduce_sum(valid, axis=1, keepdims=True) + 1e-6
    if mode == "mean":
        summed = tf.reduce_sum(seq_emb * tf.expand_dims(valid, -1), axis=1)
        return summed / denom
    if mode == "max":
        neg_inf = tf.cast(-1e9, seq_emb.dtype)
        masked = tf.where(tf.expand_dims(pad_mask, -1), tf.fill(tf.shape(seq_emb), neg_inf), seq_emb)
        max_val = tf.reduce_max(masked, axis=1)
        has_valid = tf.reduce_any(tf.logical_not(pad_mask), axis=1, keepdims=True)
        return tf.where(has_valid, max_val, tf.zeros_like(max_val))
    if mode == "target":
        counts = tf.reduce_sum(valid, axis=1)
        last_idx = tf.maximum(tf.cast(counts, tf.int32) - 1, 0)
        batch_idx = tf.range(tf.shape(tokens)[0])
        gather_idx = tf.stack([batch_idx, last_idx], axis=1)
        gathered = tf.gather_nd(seq_emb, gather_idx)
        gathered = tf.where(tf.expand_dims(counts > 0, -1), gathered, tf.zeros_like(gathered))
        return gathered
    raise ValueError("Unsupported seq_pool mode: %s" % mode)


def _parse_pool_modes(value, default="mean"):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip().lower() for v in value.split(",") if v.strip()]
    else:
        items = []
    return items or [default]


def _prepare_seq_tokens(features, embeddings_table, policy, seq_cfg, pool_modes,
                        restrict, update_ops):
    if not seq_cfg:
        return None, 0
    if "seq_features" not in features:
        logger.warning("features 中缺少 seq_features，RankMixer 仅使用非序列特征。")
        return None, 0
    seq_features_flat = _dense_if_sparse(features["seq_features"], default_value="0")
    start = 0
    seq_tokens = []
    for seq_col, L in seq_cfg.items():
        tokens_slice = seq_features_flat[:, start:start + L]
        start += L
        mask = tf.equal(tokens_slice, tf.constant("0", dtype=tf.string))
        empty = tf.fill(tf.shape(tokens_slice), "")
        tokens_slice = tf.where(mask, empty, tokens_slice)
        tokens = _pad_trunc_to_length(tokens_slice, L)
        seq_emb, up_s, rs_s = _get_seq_embedding(tokens, embeddings_table, policy, name="%s_lookup" % seq_col)
        update_ops.append(up_s)
        if restrict:
            update_ops.append(rs_s)
        for pool_mode in pool_modes:
            pooled = _sequence_pool(seq_emb, tokens, mode=pool_mode)
            seq_tokens.append(pooled)
    if not seq_tokens:
        return None, 0
    seq_stack = tf.stack(seq_tokens, axis=1)
    return seq_stack, len(seq_tokens)


def _binary_cross_entropy_from_probs(labels, probs, eps=1e-7):
    labels = tf.cast(labels, tf.float32)
    probs = tf.clip_by_value(probs, eps, 1.0 - eps)
    loss = -labels * tf.math.log(probs) - (1.0 - labels) * tf.math.log(1.0 - probs)
    return tf.reduce_mean(loss)


def _compute_bce_from_logits(logits, labels):
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(tf.cast(logits, tf.float32), [-1, 1])
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(ce)
    prob = tf.nn.sigmoid(logits)
    return loss, prob


def _pick_label(source_features, source_labels, candidates):
    for key in candidates:
        if key in source_features:
            return source_features[key]
    if source_labels is not None:
        if isinstance(source_labels, dict):
            for key in candidates:
                if key in source_labels:
                    return source_labels[key]
        else:
            return source_labels
    return None


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    rank_cfg = params.get("rankmixer", {}) if params else {}
    d_model = int(rank_cfg.get("d_model", 128))
    num_layers = int(rank_cfg.get("num_layers", 4))
    num_heads_req = int(rank_cfg.get("num_heads", 8))
    num_heads = num_heads_req
    ffn_mult = int(rank_cfg.get("ffn_mult", 4))
    token_dp = float(rank_cfg.get("token_mixing_dropout", 0.0))
    ffn_dp = float(rank_cfg.get("ffn_dropout", 0.0))
    use_other = bool(rank_cfg.get("use_other_features", True))
    seq_pool_modes = _parse_pool_modes(rank_cfg.get("seq_pool", ["mean"]))
    dense_group_size = int(rank_cfg.get("dense_token_group_size", 0))
    dense_group_pool = str(rank_cfg.get("dense_token_pool", "mean")).lower()
    embedding_size = int(rank_cfg.get("embedding_size", 9))
    add_cls_token = bool(rank_cfg.get("add_cls_token", True))
    input_dropout = float(rank_cfg.get("input_dropout", 0.0))
    head_dropout = float(rank_cfg.get("head_dropout", 0.0))
    tokenization = str(rank_cfg.get("tokenization", "paper")).lower()
    num_tokens = int(rank_cfg.get("num_tokens", 64))
    token_mixer_type = str(rank_cfg.get("token_mixer_type", "paper")).lower()
    pooling = str(rank_cfg.get("pooling", "mean")).lower()

    ps_num = int(params.get("ps_num", 0)) if params else 0
    restrict = bool(params.get("restrict", False)) if params else False

    device = params.get("device", "CPU") if params else "CPU"
    if is_training:
        devices_info = ["/job:localhost/replica:0/task:{}/{}:{}".format(i, device, i) for i in range(ps_num)]
        initializer = tf.compat.v1.random_normal_initializer(-1, 1)
    elif mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
        devices_info = ["/job:localhost/replica:0/task:{}/{}:0".format(0, device) for _ in range(ps_num)]
        initializer = tf.compat.v1.zeros_initializer()
    else:
        devices_info = ["/job:localhost/replica:0/task:{}/CPU:0".format(0) for _ in range(ps_num)]
        initializer = tf.compat.v1.zeros_initializer()
    if len(devices_info) == 0:
        devices_info = "/job:localhost/replica:0/task:0/CPU:0"

    embeddings_table = tfra.dynamic_embedding.get_variable(
        name="embeddings", dim=embedding_size, devices=devices_info,
        trainable=is_training, initializer=initializer)

    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings_table)
    update_tstp_op = policy.apply_update(tf.constant([0], dtype=tf.int64))
    restrict_op = policy.apply_restriction(int(1e8))
    groups = [update_tstp_op]
    if restrict:
        groups.append(restrict_op)

    seq_cfg = TrainConfig.seq_length
    update_ops = []

    other_emb = None
    if use_other:
        other_emb, up_t, rs_t = _get_dense_emb_from_features(features, embeddings_table, policy)
        update_ops.append(up_t)
        if restrict:
            update_ops.append(rs_t)
    seq_tokens, seq_token_count = _prepare_seq_tokens(features, embeddings_table, policy, seq_cfg,
                                                      seq_pool_modes, restrict, update_ops)

    if tokenization in ("paper", "flat", "flat_slice"):
        # 论文 tokenization：将不同来源 embedding 拼成 e_input，再按固定 token 数切片 + Proj
        raw_chunks = []
        if use_other and other_emb is not None:
            raw_chunks.append(other_emb)  # [B, F*E]
        if seq_tokens is not None and seq_token_count > 0:
            raw_chunks.append(tf.reshape(seq_tokens, [tf.shape(seq_tokens)[0], -1]))  # [B, S*E]
        if not raw_chunks:
            raise ValueError("RankMixer 无可用输入，请确认 use_other_features 或 seq_features 配置。")
        e_input = tf.concat(raw_chunks, axis=1)  # [B, total_dim]

        # 固定 token 数：pad 到 num_tokens * d_in
        # Dense layer 需要最后一维静态已知；因此这里用静态 shape 计算 d_in。
        T = int(num_tokens)
        total_dim_static = e_input.get_shape().as_list()[1]
        if total_dim_static is None:
            # 尝试从组成部分静态推断（other_emb / seq_tokens 都应有静态维度）
            total_dim_static = 0
            if use_other and other_emb is not None and other_emb.get_shape().as_list()[1] is not None:
                total_dim_static += int(other_emb.get_shape().as_list()[1])
            if seq_tokens is not None:
                s_shape = seq_tokens.get_shape().as_list()
                if len(s_shape) == 3 and s_shape[1] is not None and s_shape[2] is not None:
                    total_dim_static += int(s_shape[1]) * int(s_shape[2])
            if total_dim_static <= 0:
                raise ValueError(
                    "paper tokenization 需要 e_input 的静态维度（Dense 要求输入最后一维已知）。"
                    "请检查 dataset 输出是否导致 shape 丢失，或将 rankmixer.tokenization 改为 legacy。"
                )
            e_input.set_shape([None, total_dim_static])

        d_in = int(math.ceil(float(total_dim_static) / float(T)))
        pad_len = int(T * d_in - total_dim_static)
        if pad_len > 0:
            e_input = tf.pad(e_input, paddings=[[0, 0], [0, pad_len]])
        tokens = tf.reshape(e_input, [-1, T, d_in])
        tokens.set_shape([None, T, d_in])
        tokens = tf.compat.v1.layers.dense(tokens, units=d_model, activation=None, name="token_projection")
        token_count = T

        # 论文 TokenMixing 推荐 H=T，要求 D % T == 0；若不满足则降级到 learned mixer
        if token_mixer_type in ("paper", "shuffle", "rankmixer") and (d_model % token_count != 0):
            logger.warning("d_model=%d 不能整除 num_tokens=%d，token_mixer_type=paper 将降级为 learned。",
                           d_model, token_count)
            token_mixer_type = "learned"

        # 论文里一般 H=T；此处默认跟随 token_count
        if token_mixer_type in ("paper", "shuffle", "rankmixer"):
            num_heads = token_count

    else:
        # 兼容旧实现：每个特征/序列汇聚向量作为一个 token，可选做 group pooling
        token_chunks = []
        token_count = 0
        dense_token_count = 0
        seq_token_count = 0

        if use_other and other_emb is not None:
            fea_size = len(select_feature)
            dense_tokens = tf.reshape(other_emb, [-1, fea_size, embedding_size])
            fea_token_count = fea_size
            if dense_group_size and dense_group_size > 1:
                group_count = (fea_size + dense_group_size - 1) // dense_group_size
                pad = group_count * dense_group_size - fea_size
                if pad > 0:
                    pad_tensor = tf.zeros([tf.shape(dense_tokens)[0], pad, embedding_size])
                    dense_tokens = tf.concat([dense_tokens, pad_tensor], axis=1)
                dense_tokens = tf.reshape(dense_tokens, [-1, group_count, dense_group_size, embedding_size])
                if dense_group_pool == "max":
                    dense_tokens = tf.reduce_max(dense_tokens, axis=2)
                else:
                    dense_tokens = tf.reduce_mean(dense_tokens, axis=2)
                fea_token_count = group_count
            token_chunks.append(dense_tokens)
            token_count += fea_token_count
            dense_token_count = fea_token_count

        if seq_tokens is not None:
            token_chunks.append(seq_tokens)
            token_count += seq_token_count

        if not token_chunks:
            raise ValueError("RankMixer 无可用 token，请确认 use_other_features 或 seq_features 配置。")

        tokens = tf.concat(token_chunks, axis=1)
        tokens.set_shape([None, token_count, embedding_size])
        if embedding_size != d_model:
            tokens = tf.compat.v1.layers.dense(tokens, units=d_model, activation=None, name="token_projection")
        tokens.set_shape([None, token_count, d_model])

        if token_mixer_type in ("paper", "shuffle", "rankmixer") and (d_model % token_count != 0):
            logger.warning("d_model=%d 不能整除 token_count=%d，token_mixer_type=paper 将降级为 learned。",
                           d_model, token_count)
            token_mixer_type = "learned"

    # 论文实现默认不需要 CLS token；如果你想保留，仍可打开
    if add_cls_token:
        cls_embed = tf.compat.v1.get_variable(
            "rankmixer_cls_token", shape=[1, 1, d_model],
            initializer=tf.random_normal_initializer(stddev=0.02))
        cls_token = tf.tile(cls_embed, [tf.shape(tokens)[0], 1, 1])
        tokens = tf.concat([cls_token, tokens], axis=1)
        token_count += 1
        if token_mixer_type in ("paper", "shuffle", "rankmixer") and (d_model % token_count != 0):
            logger.warning("CLS token 导致 token_count=%d 不再整除 d_model=%d，token_mixer_type=paper 将降级为 learned。",
                           token_count, d_model)
            token_mixer_type = "learned"
            num_heads = num_heads_req
        elif token_mixer_type in ("paper", "shuffle", "rankmixer"):
            num_heads = token_count

    input_ln = LayerNorm(name="input_ln")
    tokens = input_ln(tokens)
    if input_dropout and is_training:
        tokens = tf.nn.dropout(tokens, keep_prob=1.0 - input_dropout)

    encoder = RankMixerEncoder(num_layers=num_layers, num_tokens=token_count, d_model=d_model,
                               num_heads=num_heads, ffn_mult=ffn_mult,
                               token_mixer_type=token_mixer_type, token_dp=token_dp, ffn_dp=ffn_dp,
                               name="rankmixer_encoder")
    encoded = encoder(tokens, training=is_training)
    encoded.set_shape([None, token_count, d_model])

    # 论文里 output pooling 使用 mean pooling
    if pooling in ("mean", "avg"):
        head_input = tf.reduce_mean(encoded, axis=1)
    elif pooling == "cls":
        head_input = encoded[:, 0, :]
    elif pooling == "mean_max":
        head_input = tf.concat([tf.reduce_mean(encoded, axis=1), tf.reduce_max(encoded, axis=1)], axis=1)
    else:
        raise ValueError("Unknown pooling: %s" % pooling)

    if head_dropout and is_training:
        head_input = tf.nn.dropout(head_input, keep_prob=1.0 - head_dropout)
    head_hidden = tf.compat.v1.layers.dense(head_input, units=d_model * 2, activation=gelu,
                                            name="rankmixer_head_dense1")
    if head_dropout and is_training:
        head_hidden = tf.nn.dropout(head_hidden, keep_prob=1.0 - head_dropout)
    head_hidden = tf.compat.v1.layers.dense(head_hidden, units=d_model, activation=gelu,
                                            name="rankmixer_head_dense2")

    ctr_logits = tf.compat.v1.layers.dense(head_hidden, units=1, activation=None, name="ctr_logit")
    cvr_logits = tf.compat.v1.layers.dense(head_hidden, units=1, activation=None, name="cvr_logit")

    ctr_label_raw = _pick_label(features, labels, ["click_label", "ctr_label", "is_click"])
    if ctr_label_raw is None:
        ctr_label_raw = tf.zeros([tf.shape(ctr_logits)[0]], tf.float32)
    ctr_label = tf.reshape(tf.cast(ctr_label_raw, tf.float32), [-1, 1])

    ctcvr_label_raw = _pick_label(features, labels, ["ctcvr_label", "is_conversion"])
    has_ctcvr = ctcvr_label_raw is not None
    if ctcvr_label_raw is None:
        ctcvr_label_raw = tf.zeros([tf.shape(ctr_logits)[0]], tf.float32)
    ctcvr_label = tf.reshape(tf.cast(ctcvr_label_raw, tf.float32), [-1, 1])

    ctr_loss, ctr_prob = _compute_bce_from_logits(ctr_logits, ctr_label)
    cvr_prob = tf.nn.sigmoid(cvr_logits)
    ctcvr_prob = ctr_prob * cvr_prob
    ctcvr_loss = _binary_cross_entropy_from_probs(ctcvr_label, ctcvr_prob) if has_ctcvr else tf.constant(0.0)

    # 可选：点击条件下的 CVR 辅助损失（有助于提升 cvr_auc/cvr_pcoc，缓解 ESMM 分解漂移）
    cvr_loss_weight = float(rank_cfg.get("cvr_loss_weight", params.get("cvr_loss_weight", 0.0) if params else 0.0))
    cvr_loss = tf.constant(0.0, dtype=tf.float32)
    if cvr_loss_weight and cvr_loss_weight > 0:
        click_mask = tf.reshape(tf.cast(ctr_label, tf.float32), [-1, 1])  # [B,1] 0/1
        cvr_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=ctcvr_label, logits=cvr_logits)  # [B,1]
        denom = tf.reduce_sum(click_mask) + 1e-6
        cvr_loss = tf.reduce_sum(cvr_ce * click_mask) / denom

    ctr_w = float(rank_cfg.get("ctr_loss_weight", 1.0))
    ctcvr_w = float(rank_cfg.get("ctcvr_loss_weight", 1.0))
    total_loss = ctr_w * ctr_loss + ctcvr_w * ctcvr_loss + cvr_loss_weight * cvr_loss

    eval_metric_ops = OrderedDict()
    evaluate(ctr_label, ctr_prob, "task1_ctr", eval_metric_ops)
    evaluate(ctcvr_label, ctcvr_prob, "task1_ctcvr", eval_metric_ops)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings_table.size(),
        "losses": total_loss,
        "ctr_losses": ctr_loss,
        "ctcvr_losses": ctcvr_loss,
        "rankmixer_tokens": tf.constant(int(token_count), dtype=tf.int32),
        "rankmixer_heads": tf.constant(int(num_heads), dtype=tf.int32),
    })
    if cvr_loss_weight and cvr_loss_weight > 0:
        loggings["cvr_losses"] = cvr_loss
    for k, (val, up_op) in eval_metric_ops.items():
        loggings[k] = val
        groups.append(up_op)
    groups.extend(update_ops)

    batch_size = tf.shape(ctr_prob)[0]
    out_tensor = tf.concat([ctr_label, ctcvr_label, ctr_prob, cvr_prob, ctcvr_prob], axis=1)
    predictions = {
        "requestid": features.get("requestid", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "combination_un_id": features.get("combination_un_id", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "out": out_tensor
    }

    class _ModelView(object):
        pass

    model = _ModelView()
    model.losses = total_loss
    model.ctr_losses = ctr_loss
    model.ctcvr_losses = ctcvr_loss
    model.ctr_labels = [ctr_label]
    model.ctr_probs = [ctr_prob]
    model.cvr_probs = [cvr_prob]
    model.ctcvr_labels = [ctcvr_label]
    model.ctcvr_probs = [ctcvr_prob]
    model.predictions = predictions
    model.outputs = {
        "out": out_tensor,
        "ctr_output": ctr_prob,
        "cvr_output": cvr_prob,
        "ctcvr_output": ctcvr_prob
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        predict_outputs = model.outputs
        export_outputs = {"serving_default": tf.compat.v1.estimator.export.PredictOutput(predict_outputs)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                          loss=model.losses, eval_metric_ops=eval_metric_ops)

    trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    l2_reg = float(params.get("l2_reg", 1e-6)) if params else 1e-6
    if l2_reg > 0:
        l2_loss = l2_reg * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                     for v in trainable_variables if 'bias' not in v.name])
        model.losses = model.losses + l2_loss
        loggings["l2_loss"] = l2_loss

    opt_cfg = params.get("optimize_config", {}) if params else {}
    learning_rate = float(opt_cfg.get("learning_rate", 1e-3))
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.999))
    epsilon = float(opt_cfg.get("epsilon", 1e-8))

    # 可选：warmup / decay / grad clip（训练更稳，loss 不必强求单调下降）
    warmup_steps = int(opt_cfg.get("warmup_steps", 0))
    decay_steps = int(opt_cfg.get("decay_steps", 0))
    decay_type = str(opt_cfg.get("decay_type", "none")).lower()
    min_learning_rate = float(opt_cfg.get("min_learning_rate", 0.0))
    grad_clip_norm = float(opt_cfg.get("grad_clip_norm", 0.0))

    lr = tf.constant(learning_rate, dtype=tf.float32)
    if decay_steps and decay_steps > 0 and decay_type in ("cosine", "poly", "polynomial"):
        gs = tf.cast(global_step, tf.int32)
        if decay_type == "cosine":
            lr_decay = tf.compat.v1.train.cosine_decay(learning_rate=learning_rate, global_step=gs,
                                                       decay_steps=decay_steps, alpha=min_learning_rate / learning_rate
                                                       if learning_rate > 0 else 0.0)
        else:
            lr_decay = tf.compat.v1.train.polynomial_decay(learning_rate=learning_rate, global_step=gs,
                                                           decay_steps=decay_steps, end_learning_rate=min_learning_rate,
                                                           power=1.0)
        lr = lr_decay
    if warmup_steps and warmup_steps > 0:
        warm = tf.cast(tf.minimum(global_step + 1, warmup_steps), tf.float32) / float(warmup_steps)
        lr = tf.where(global_step < warmup_steps, lr * warm, lr)

    loggings["lr"] = lr

    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
    opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(opt)
    grads_and_vars = opt.compute_gradients(model.losses)
    grads = [g for g, _ in grads_and_vars if g is not None]
    grad_norm = tf.linalg.global_norm(grads) if grads else tf.constant(0.0, tf.float32)
    loggings["grad_norm"] = grad_norm
    if grad_clip_norm and grad_clip_norm > 0 and grads:
        clipped_grads, _ = tf.clip_by_global_norm(grads, clip_norm=grad_clip_norm)
        it = iter(clipped_grads)
        clipped_gv = []
        for g, v in grads_and_vars:
            if g is None:
                clipped_gv.append((g, v))
            else:
                clipped_gv.append((next(it), v))
        grads_and_vars = clipped_gv
    dense_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
    train_op = tf.group(dense_op, *groups)

    log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                      loss=model.losses, train_op=train_op, training_hooks=[log_hook])
