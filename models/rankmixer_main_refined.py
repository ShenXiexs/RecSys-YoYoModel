# models/rankmixer_main.py
# RankMixer Estimator: 重写版，参考 RankMixer 论文 + HSTU/TOP5 输入处理
import math
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict

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


def _sanitize_group_name(name):
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return safe or "group"


DEFAULT_SEMANTIC_GROUP_RULES = [
    {"name": "core_id", "patterns": [r"^combination_un_id$"]},
    {"name": "seq", "patterns": [r"^seq::", r"^seq_"]},
    {"name": "dpa", "patterns": [r"^dpa_"]},
    {"name": "item_meta", "patterns": [
        r"^brand_name$", r"^first_category$", r"^second_category$", r"^annual_vol$",
        r"^shop_id$", r"^shop_name$", r"^shop_source$"
    ]},
    {"name": "price", "patterns": [
        r"^reserve_price$", r"^final_promotion_price$", r"^commission$", r"^commission_rate$"
    ]},
    {"name": "semantics", "patterns": [r"^title_sem_id$", r"^image_sem_id$"]},
    {"name": "adslot", "patterns": [
        r"^adx_adslot_id$", r"^ssp_adslot_id$", r"^adslot_id$", r"^channel_id$",
        r"^adslot_id_type$", r"^source_adslot_type$", r"^bid_floor$",
        r"^ad_idea_id$", r"^ad_unit_id$", r"^template_id$", r"^template_type$",
        r"^promotion_type$", r"^target_type$"
    ]},
    {"name": "app", "patterns": [
        r"^app_pkg_src$", r"^app_pkg$", r"^app_src_", r"^package_name$", r"^app_first_type$", r"^app_second_type$"
    ]},
    {"name": "device", "patterns": [
        r"^device_", r"^network$", r"^ip_region$", r"^ip_city$", r"^device_size$", r"^city_level$"
    ]},
    {"name": "strategy", "patterns": [
        r"^model_type$", r"^dispatch_center_id$", r"^rta_type$", r"^crowd_type$", r"^is_new_item$"
    ]},
    {"name": "time", "patterns": [r"^day_h$"]},
    {"name": "user_stat", "patterns": [r"^user__"]},
    {"name": "item_stat", "patterns": [r"^item__"]},
    {"name": "skuid_key_one", "patterns": [r"^skuid__key_one__"]},
    {"name": "skuid_key_two", "patterns": [r"^skuid__key_two__"]},
    {"name": "skuid_key_three", "patterns": [r"^skuid__key_three__"]},
    {"name": "skuid_key_four", "patterns": [r"^skuid__key_four__"]},
    {"name": "skuid_key_five", "patterns": [r"^skuid__key_five__"]},
    {"name": "skuid_stat", "patterns": [r"^skuid__"]},
    {"name": "tsd_stat", "patterns": [r"^tsd__"]},
    {"name": "isd_stat", "patterns": [r"^isd__"]},
]


class MultiHeadTokenMixer(tf.layers.Layer):
    """可学习权重的多头 Token Mixing。"""
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


class ParameterFreeTokenMixer(tf.layers.Layer):
    """参数无关的多头 Token Mixing（论文版本，通常 H=T）。"""
    def __init__(self, num_tokens, d_model, num_heads=None, dropout=0.0, name=None):
        super(ParameterFreeTokenMixer, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.num_heads = int(num_heads) if num_heads else int(num_tokens)
        self.dropout = float(dropout)

    def build(self, input_shape):
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model 必须能整除 num_heads, got d_model=%d num_heads=%d" %
                             (self.d_model, self.num_heads))
        self.d_head = self.d_model // self.num_heads
        super(ParameterFreeTokenMixer, self).build(input_shape)

    def call(self, x, training=False):
        # x: [B,T,D]
        B = tf.shape(x)[0]
        T = self.num_tokens
        h = self.num_heads
        d = self.d_head
        x_heads = tf.reshape(x, [B, T, h, d])
        x_heads = tf.transpose(x_heads, [0, 2, 1, 3])  # [B,h,T,d]
        mixed = tf.reshape(x_heads, [B, h, T * d])
        if self.dropout and training:
            mixed = tf.nn.dropout(mixed, keep_prob=1.0 - self.dropout)
        return mixed


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


class PerTokenSparseMoE(tf.layers.Layer):
    """Per-token Sparse-MoE（ReLU routing + DTSI）。"""
    def __init__(self, num_tokens, d_model, mult=4, num_experts=4, dropout=0.0,
                 l1_coef=0.0, use_dtsi=True, name=None):
        super(PerTokenSparseMoE, self).__init__(name=name)
        self.num_tokens = int(num_tokens)
        self.d_model = int(d_model)
        self.mult = int(mult)
        self.num_experts = int(num_experts)
        self.dropout = float(dropout)
        self.l1_coef = float(l1_coef)
        self.use_dtsi = bool(use_dtsi)

    def build(self, input_shape):
        hidden_dim = self.d_model * self.mult
        init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight("W1", [self.num_tokens, self.num_experts, self.d_model, hidden_dim], initializer=init)
        self.b1 = self.add_weight("b1", [self.num_tokens, self.num_experts, hidden_dim], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight("W2", [self.num_tokens, self.num_experts, hidden_dim, self.d_model], initializer=init)
        self.b2 = self.add_weight("b2", [self.num_tokens, self.num_experts, self.d_model], initializer=tf.zeros_initializer())
        self.gate_w_train = self.add_weight("gate_w_train", [self.num_tokens, self.d_model, self.num_experts],
                                            initializer=init)
        self.gate_b_train = self.add_weight("gate_b_train", [self.num_tokens, self.num_experts],
                                            initializer=tf.zeros_initializer())
        if self.use_dtsi:
            self.gate_w_infer = self.add_weight("gate_w_infer", [self.num_tokens, self.d_model, self.num_experts],
                                                initializer=init)
            self.gate_b_infer = self.add_weight("gate_b_infer", [self.num_tokens, self.num_experts],
                                                initializer=tf.zeros_initializer())
        super(PerTokenSparseMoE, self).build(input_shape)

    def _router(self, x, w, b):
        gate = tf.einsum("btd,tde->bte", x, w) + b
        return tf.nn.relu(gate)

    def call(self, x, training=False):
        h = tf.einsum("btd,tedh->bteh", x, self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training:
            h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        expert_out = tf.einsum("bteh,tehd->bted", h, self.W2) + self.b2
        if self.dropout and training:
            expert_out = tf.nn.dropout(expert_out, keep_prob=1.0 - self.dropout)

        gate_train = self._router(x, self.gate_w_train, self.gate_b_train)
        if self.use_dtsi:
            gate_infer = self._router(x, self.gate_w_infer, self.gate_b_infer)
        else:
            gate_infer = gate_train

        gate = gate_train if training else gate_infer
        y = tf.reduce_sum(expert_out * tf.expand_dims(gate, -1), axis=2)

        if self.l1_coef > 0.0:
            l1_loss = self.l1_coef * tf.reduce_mean(tf.reduce_sum(gate_infer, axis=-1))
        else:
            l1_loss = tf.constant(0.0)
        return y, l1_loss


class RankMixerBlock(tf.layers.Layer):
    """RankMixer Block: TokenMixing + Per-token FFN，支持 Post-LN 与 Sparse-MoE。"""
    def __init__(self, num_tokens, d_model, num_heads, ffn_mult, token_dp=0.0, ffn_dp=0.0,
                 token_mixing_type="learned", ln_style="pre",
                 use_moe=False, moe_experts=4, moe_l1_coef=0.0, moe_use_dtsi=True,
                 name=None):
        super(RankMixerBlock, self).__init__(name=name)
        self.ln1 = LayerNorm(name="ln1")
        self.ln2 = LayerNorm(name="ln2")
        self.ln_style = str(ln_style).lower()
        self.use_moe = bool(use_moe)

        token_mixing_type = str(token_mixing_type).lower()
        if token_mixing_type in ("param_free", "parameter_free", "paper"):
            self.token_mixer = ParameterFreeTokenMixer(num_tokens=num_tokens, d_model=d_model,
                                                       num_heads=num_heads, dropout=token_dp,
                                                       name="token_mixer")
        else:
            self.token_mixer = MultiHeadTokenMixer(num_tokens=num_tokens, d_model=d_model,
                                                   num_heads=num_heads, dropout=token_dp,
                                                   name="token_mixer")

        if self.use_moe:
            self.per_token_ffn = PerTokenSparseMoE(num_tokens=num_tokens, d_model=d_model,
                                                   mult=ffn_mult, num_experts=moe_experts,
                                                   dropout=ffn_dp, l1_coef=moe_l1_coef,
                                                   use_dtsi=moe_use_dtsi, name="per_token_moe")
        else:
            self.per_token_ffn = PerTokenFFN(num_tokens=num_tokens, d_model=d_model,
                                             mult=ffn_mult, dropout=ffn_dp, name="per_token_ffn")
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        moe_loss = tf.constant(0.0)
        if self.ln_style == "post":
            y = self.token_mixer(x, training=training)
            x = self.ln1(x + y)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(x, training=training)
            else:
                z = self.per_token_ffn(x, training=training)
            out = self.ln2(x + z)
        else:
            y = self.ln1(x)
            y = self.token_mixer(y, training=training)
            x = x + y
            z = self.ln2(x)
            if self.use_moe:
                z, moe_loss = self.per_token_ffn(z, training=training)
            else:
                z = self.per_token_ffn(z, training=training)
            out = x + z
        self.moe_loss = moe_loss
        return out


class RankMixerEncoder(tf.layers.Layer):
    """堆叠 RankMixerBlock。"""
    def __init__(self, num_layers, num_tokens, d_model, num_heads, ffn_mult,
                 token_dp=0.0, ffn_dp=0.0, token_mixing_type="learned", ln_style="pre",
                 use_moe=False, moe_experts=4, moe_l1_coef=0.0, moe_use_dtsi=True,
                 use_final_ln=True, name=None):
        super(RankMixerEncoder, self).__init__(name=name)
        self.use_final_ln = bool(use_final_ln)
        self.blocks = [
            RankMixerBlock(num_tokens=num_tokens, d_model=d_model, num_heads=num_heads,
                           ffn_mult=ffn_mult, token_dp=token_dp, ffn_dp=ffn_dp,
                           token_mixing_type=token_mixing_type, ln_style=ln_style,
                           use_moe=use_moe, moe_experts=moe_experts,
                           moe_l1_coef=moe_l1_coef, moe_use_dtsi=moe_use_dtsi,
                           name="block_%d" % i)
            for i in range(num_layers)
        ]
        self.final_ln = LayerNorm(name="encoder_ln")
        self.moe_loss = tf.constant(0.0)

    def call(self, x, training=False):
        out = x
        moe_losses = []
        for blk in self.blocks:
            out = blk(out, training=training)
            moe_losses.append(blk.moe_loss)
        self.moe_loss = tf.add_n(moe_losses) if moe_losses else tf.constant(0.0)
        return self.final_ln(out) if self.use_final_ln else out


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


def _parse_summary_modes(value, default="mean"):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip().lower() for v in value.split(",") if v.strip()]
    else:
        items = []
    return items or [default]


def _get_train_date_from_flags():
    try:
        flags = tf.app.flags.FLAGS
        time_str = getattr(flags, "end_time_str", None) or getattr(flags, "time_str", None)
        if time_str:
            return str(time_str)[:8]
    except Exception:
        return None
    return None


def _resolve_learning_rate(opt_cfg):
    lr = float(opt_cfg.get("learning_rate", 1e-3))
    schedule = opt_cfg.get("lr_schedule") or opt_cfg.get("lr_by_date")
    if not schedule:
        return lr
    cutoff = str(schedule.get("cutoff_date", "")).strip()
    if not cutoff:
        return lr
    before = float(schedule.get("before", lr))
    after = float(schedule.get("after", lr))
    train_date = _get_train_date_from_flags()
    if not train_date:
        return lr
    return before if train_date <= cutoff else after


def _compile_group_rules(group_rules):
    rules = group_rules or DEFAULT_SEMANTIC_GROUP_RULES
    compiled = []
    for rule in rules:
        name = _sanitize_group_name(rule.get("name", "group"))
        patterns = [p for p in rule.get("patterns", []) if p]
        if not patterns:
            continue
        compiled.append((name, [re.compile(p) for p in patterns]))
    return compiled


def _assign_semantic_groups(feature_names, group_rules):
    compiled = _compile_group_rules(group_rules)
    grouped = OrderedDict((name, []) for name, _ in compiled)
    other = []
    for idx, feat in enumerate(feature_names):
        matched = False
        for name, patterns in compiled:
            for pat in patterns:
                if pat.search(feat):
                    grouped[name].append(idx)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            other.append(idx)
    ordered_indices = []
    for name in grouped:
        ordered_indices.extend(grouped[name])
    if other:
        ordered_indices.extend(other)
    return ordered_indices


def _maybe_project_tokens(tokens, in_dim, out_dim, name):
    if in_dim == out_dim:
        return tokens
    return tf.compat.v1.layers.dense(tokens, units=out_dim, activation=None, name=name)


def _semantic_tokenize(embeddings, feature_names, embedding_dim, target_tokens, token_pool, proj_dim,
                       group_rules=None, name="semantic_token_proj"):
    if embeddings is None or not feature_names:
        return None, 0, 0
    feature_count = len(feature_names)
    target_tokens = int(target_tokens) if int(target_tokens) > 0 else feature_count
    token_pool = str(token_pool).lower()

    ordered_indices = _assign_semantic_groups(feature_names, group_rules)
    if ordered_indices != list(range(feature_count)):
        gather_idx = tf.constant(ordered_indices, dtype=tf.int32)
        embeddings = tf.gather(embeddings, gather_idx, axis=1)

    token_size = int(math.ceil(float(feature_count) / float(target_tokens)))
    pad_needed = target_tokens * token_size - feature_count
    if pad_needed > 0:
        pad_tensor = tf.zeros([tf.shape(embeddings)[0], pad_needed, embedding_dim])
        embeddings = tf.concat([embeddings, pad_tensor], axis=1)

    tokens = tf.reshape(embeddings, [tf.shape(embeddings)[0], target_tokens, token_size, embedding_dim])
    if token_pool == "mean":
        tokens = tf.reduce_mean(tokens, axis=2)
        tokens_dim = int(embedding_dim)
    elif token_pool == "sum":
        tokens = tf.reduce_sum(tokens, axis=2)
        tokens_dim = int(embedding_dim)
    elif token_pool == "max":
        tokens = tf.reduce_max(tokens, axis=2)
        tokens_dim = int(embedding_dim)
    elif token_pool in ("concat", "concat_proj"):
        flat = tf.reshape(tokens, [tf.shape(tokens)[0], target_tokens, token_size * embedding_dim])
        tokens = tf.compat.v1.layers.dense(flat, units=proj_dim, activation=None, name=name)
        tokens_dim = int(proj_dim)
    else:
        raise ValueError("Unsupported token_pool: %s" % token_pool)
    return tokens, target_tokens, tokens_dim


def _prepare_seq_tokens(features, embeddings_table, policy, seq_cfg, pool_modes,
                        restrict, update_ops, return_names=False):
    if not seq_cfg:
        return (None, 0, []) if return_names else (None, 0)
    if "seq_features" not in features:
        logger.warning("features 中缺少 seq_features，RankMixer 仅使用非序列特征。")
        return (None, 0, []) if return_names else (None, 0)
    seq_features_flat = _dense_if_sparse(features["seq_features"], default_value="0")
    start = 0
    seq_tokens = []
    seq_names = []
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
            if return_names:
                seq_names.append("seq::%s::%s" % (seq_col, pool_mode))
    if not seq_tokens:
        return (None, 0, []) if return_names else (None, 0)
    seq_stack = tf.stack(seq_tokens, axis=1)
    return (seq_stack, len(seq_tokens), seq_names) if return_names else (seq_stack, len(seq_tokens))


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
    num_heads = int(rank_cfg.get("num_heads", 8))
    ffn_mult = int(rank_cfg.get("ffn_mult", 4))
    token_dp = float(rank_cfg.get("token_mixing_dropout", 0.0))
    ffn_dp = float(rank_cfg.get("ffn_dropout", 0.0))
    use_other = bool(rank_cfg.get("use_other_features", True))
    seq_pool_modes = _parse_pool_modes(rank_cfg.get("seq_pool", ["mean"]))
    dense_group_size = int(rank_cfg.get("dense_token_group_size", 0))
    dense_group_pool = str(rank_cfg.get("dense_token_pool", "mean")).lower()
    embedding_size = int(rank_cfg.get("embedding_size", 9))
    add_cls_token = bool(rank_cfg.get("add_cls_token", True))
    use_input_ln = bool(rank_cfg.get("input_ln", True))
    input_dropout = float(rank_cfg.get("input_dropout", 0.0))
    head_dropout = float(rank_cfg.get("head_dropout", 0.0))

    token_mixing_type = str(rank_cfg.get("token_mixing_type", "learned")).lower()
    ln_style = str(rank_cfg.get("ln_style", "pre")).lower()
    use_final_ln = bool(rank_cfg.get("use_final_ln", True))

    tokenization = str(rank_cfg.get("tokenization", "semantic")).lower()
    semantic_target_tokens = int(rank_cfg.get("semantic_target_tokens", 16))
    semantic_token_pool = str(rank_cfg.get("semantic_token_pool", "concat_proj")).lower()
    semantic_proj_dim = int(rank_cfg.get("semantic_proj_dim", d_model))
    semantic_group_rules = rank_cfg.get("semantic_group_rules")
    include_seq_in_tokenization = bool(rank_cfg.get("include_seq_in_tokenization", True))

    summary_modes = _parse_summary_modes(rank_cfg.get("summary_pooling", ["mean"]))
    summary_exclude_cls = bool(rank_cfg.get("summary_exclude_cls", True))

    use_moe = bool(rank_cfg.get("use_moe", False))
    moe_experts = int(rank_cfg.get("moe_num_experts", 4))
    moe_l1_coef = float(rank_cfg.get("moe_l1_coef", 0.0))
    moe_use_dtsi = bool(rank_cfg.get("moe_use_dtsi", True))

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

    token_chunks = []
    token_count = 0
    dense_token_count = 0
    seq_token_count = 0
    tokens_dim = embedding_size

    dense_embeddings = None
    dense_feature_names = []
    if use_other:
        other_emb, up_t, rs_t = _get_dense_emb_from_features(features, embeddings_table, policy)
        update_ops.append(up_t)
        if restrict:
            update_ops.append(rs_t)
        fea_size = len(select_feature)
        dense_embeddings = tf.reshape(other_emb, [-1, fea_size, embedding_size])
        dense_embeddings.set_shape([None, fea_size, embedding_size])
        dense_feature_names = list(select_feature)

    seq_embeddings = None
    seq_names = []
    seq_tokens, seq_token_count, seq_names = _prepare_seq_tokens(features, embeddings_table, policy, seq_cfg,
                                                                  seq_pool_modes, restrict, update_ops,
                                                                  return_names=True)
    if seq_tokens is not None:
        seq_embeddings = seq_tokens
        seq_embeddings.set_shape([None, seq_token_count, embedding_size])

    if tokenization == "semantic":
        feature_embeddings = []
        feature_names = []
        if dense_embeddings is not None:
            feature_embeddings.append(dense_embeddings)
            feature_names.extend(dense_feature_names)
        if include_seq_in_tokenization and seq_embeddings is not None:
            feature_embeddings.append(seq_embeddings)
            feature_names.extend(seq_names)
            seq_token_count = 0
        if not feature_embeddings:
            raise ValueError("RankMixer 无可用 token，请确认 use_other_features 或 seq_features 配置。")
        all_embeddings = feature_embeddings[0] if len(feature_embeddings) == 1 else tf.concat(feature_embeddings, axis=1)
        tokens, token_count, tokens_dim = _semantic_tokenize(
            all_embeddings, feature_names, embedding_size,
            semantic_target_tokens, semantic_token_pool, semantic_proj_dim,
            group_rules=semantic_group_rules, name="semantic_token_proj"
        )
        if tokens is None:
            raise ValueError("RankMixer 无可用 token，请确认 use_other_features 或 seq_features 配置。")
        token_chunks.append(tokens)
        dense_token_count = token_count
        if (not include_seq_in_tokenization) and (seq_embeddings is not None):
            seq_proj = _maybe_project_tokens(seq_embeddings, embedding_size, tokens_dim, name="seq_token_proj")
            token_chunks.append(seq_proj)
            token_count += seq_token_count
    else:
        if dense_embeddings is not None:
            dense_tokens = dense_embeddings
            fea_token_count = len(select_feature)
            if dense_group_size and dense_group_size > 1:
                group_count = (fea_token_count + dense_group_size - 1) // dense_group_size
                pad = group_count * dense_group_size - fea_token_count
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
        if seq_embeddings is not None:
            token_chunks.append(seq_embeddings)
            token_count += seq_token_count

    if not token_chunks:
        raise ValueError("RankMixer 无可用 token，请确认 use_other_features 或 seq_features 配置。")

    tokens = tf.concat(token_chunks, axis=1)
    tokens.set_shape([None, token_count, tokens_dim])
    if tokens_dim != d_model:
        tokens = tf.compat.v1.layers.dense(tokens, units=d_model, activation=None, name="token_projection")
    tokens.set_shape([None, token_count, d_model])

    slice_map = {}
    cursor = 0
    if add_cls_token:
        cls_embed = tf.compat.v1.get_variable(
            "rankmixer_cls_token", shape=[1, 1, d_model],
            initializer=tf.random_normal_initializer(stddev=0.02))
        cls_token = tf.tile(cls_embed, [tf.shape(tokens)[0], 1, 1])
        tokens = tf.concat([cls_token, tokens], axis=1)
        token_count += 1
        slice_map["cls"] = (0, 1)
        cursor = 1
    if dense_token_count > 0:
        slice_map["dense"] = (cursor, dense_token_count)
        cursor += dense_token_count
    if seq_token_count > 0:
        slice_map["seq"] = (cursor, seq_token_count)
        cursor += seq_token_count
    tokens.set_shape([None, token_count, d_model])

    if use_input_ln:
        input_ln = LayerNorm(name="input_ln")
        tokens = input_ln(tokens)
    if input_dropout and is_training:
        tokens = tf.nn.dropout(tokens, keep_prob=1.0 - input_dropout)

    if token_mixing_type in ("param_free", "parameter_free", "paper"):
        if num_heads != token_count:
            logger.warning("param_free token mixing 强制 num_heads=token_count (got %d, set %d)",
                           num_heads, token_count)
        num_heads = token_count

    encoder = RankMixerEncoder(num_layers=num_layers, num_tokens=token_count, d_model=d_model,
                               num_heads=num_heads, ffn_mult=ffn_mult,
                               token_dp=token_dp, ffn_dp=ffn_dp,
                               token_mixing_type=token_mixing_type, ln_style=ln_style,
                               use_moe=use_moe, moe_experts=moe_experts,
                               moe_l1_coef=moe_l1_coef, moe_use_dtsi=moe_use_dtsi,
                               use_final_ln=use_final_ln, name="rankmixer_encoder")
    encoded = encoder(tokens, training=is_training)
    encoded.set_shape([None, token_count, d_model])

    summaries = []
    if "cls" in summary_modes and slice_map.get("cls"):
        summaries.append(encoded[:, 0, :])
    pooled_source = encoded
    if summary_exclude_cls and slice_map.get("cls") and token_count > 1:
        pooled_source = encoded[:, 1:, :]
    if "mean" in summary_modes:
        summaries.append(tf.reduce_mean(pooled_source, axis=1))
    if "max" in summary_modes:
        summaries.append(tf.reduce_max(pooled_source, axis=1))
    if "dense" in summary_modes and slice_map.get("dense"):
        beg, length = slice_map["dense"]
        dense_encoded = encoded[:, beg:beg + length, :]
        summaries.append(tf.reduce_mean(dense_encoded, axis=1))
    if "seq" in summary_modes and slice_map.get("seq"):
        beg, length = slice_map["seq"]
        seq_encoded = encoded[:, beg:beg + length, :]
        summaries.append(tf.reduce_mean(seq_encoded, axis=1))
    if not summaries:
        summaries.append(tf.reduce_mean(pooled_source, axis=1))
    head_input = summaries[0] if len(summaries) == 1 else tf.concat(summaries, axis=1)
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
    total_loss = ctr_loss + ctcvr_loss
    moe_l1_loss = encoder.moe_loss if use_moe else tf.constant(0.0)
    if use_moe:
        total_loss = total_loss + moe_l1_loss

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
    })
    if use_moe:
        loggings["moe_l1_loss"] = moe_l1_loss
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
        export_outputs = {"serving_default": tf.compat.v1.estimator.export.PredictOutput(model.outputs)}
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
    learning_rate = _resolve_learning_rate(opt_cfg)
    beta1 = float(opt_cfg.get("beta1", 0.9))
    beta2 = float(opt_cfg.get("beta2", 0.999))
    epsilon = float(opt_cfg.get("epsilon", 1e-8))
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(opt)
    dense_op = opt.minimize(model.losses, global_step=global_step)
    train_op = tf.group(dense_op, *groups)

    loggings["learning_rate"] = tf.constant(learning_rate, dtype=tf.float32)
    log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                      loss=model.losses, train_op=train_op, training_hooks=[log_hook])
