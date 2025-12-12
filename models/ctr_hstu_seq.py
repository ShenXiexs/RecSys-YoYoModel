# models/ctr_hstu_seq.py
# TF v1 风格的 HSTU：改为“多头点积注意力 + U 门控（pointwise vs aggregated）”
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ========= 实用小工具 =========
def _dense_if_sparse(x, default_value=""):
    return tf.sparse.to_dense(x, default_value=default_value) if isinstance(x, tf.SparseTensor) else x

def _pad_trunc_to_length(tokens_dense, L):
    T = tf.shape(tokens_dense)[1]
    tokens_cut = tokens_dense[:, :tf.minimum(T, L)]
    pad_len = tf.maximum(0, L - tf.shape(tokens_cut)[1])
    tokens_fix = tf.pad(tokens_cut, paddings=[[0, 0], [0, pad_len]])
    tokens_fix.set_shape([None, L])
    return tokens_fix

def gelu(x):
    return 0.5 * x * (1.0 + tf.tanh(tf.sqrt(2.0 / 3.141592653589793) * (x + 0.044715 * tf.pow(x, 3))))

class LayerNorm(tf.layers.Layer):
    def __init__(self, epsilon=1e-5, name=None):
        super(LayerNorm, self).__init__(name=name); self.epsilon = epsilon
    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.gamma = self.add_weight("gamma", shape=[dim], initializer=tf.ones_initializer())
        self.beta  = self.add_weight("beta",  shape=[dim], initializer=tf.zeros_initializer())
        super(LayerNorm, self).build(input_shape)
    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[-1], keepdims=True)
        return self.gamma * (x - mean) / tf.sqrt(var + self.epsilon) + self.beta

def causal_mask(batch_size, seq_len, dtype=tf.float32):
    m = tf.linalg.band_part(tf.ones([seq_len, seq_len], dtype=dtype), -1, 0)  # [L,L]
    return tf.tile(tf.reshape(m, [1, seq_len, seq_len]), [batch_size, 1, 1])

def pad_mask_from_tokens(tokens, pad_token="", dtype=tf.float32):
    """
    非 PAD 为 1，PAD 为 0；兼容 "" 和 "0"
    """
    pad_main = tf.constant(pad_token, dtype=tokens.dtype)
    pad_zero = tf.constant("0", dtype=tokens.dtype)
    is_pad = tf.logical_or(tf.equal(tokens, pad_main), tf.equal(tokens, pad_zero))
    zeros = tf.zeros_like(tokens, dtype=dtype)
    ones  = tf.ones_like(tokens, dtype=dtype)
    return tf.where(is_pad, zeros, ones)


# def sinusoidal_position_embedding(seq_len, dim, dtype=tf.float32):
#     position = tf.cast(tf.range(seq_len), dtype=dtype)  # [L]
#     i = tf.cast(tf.range(dim), dtype=dtype)             # [D]
#     angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.maximum(1.0, tf.cast(dim, dtype)))
#     angles = tf.expand_dims(position, 1) * tf.expand_dims(angle_rates, 0)  # [L, D]
#     sin = tf.sin(angles[:, 0::2]);  cos = tf.cos(angles[:, 1::2])
#     pe = tf.reshape(tf.stack([sin, cos], axis=-1), [seq_len, -1])
#     return tf.slice(pe, [0, 0], [seq_len, dim])

def rotary_pos_emb(seq_len, dim, dtype=tf.float32):
    """
    生成 RoPE 用的 cos/sin：
      cos, sin 形状都是 [1, L, dim/2]
    要求 dim 为偶数（单头维度 d_k 一般是偶数）
    """
    assert dim % 2 == 0, "RoPE 要求每头维度 dim 为偶数"

    dim = int(dim)
    pos = tf.cast(tf.range(seq_len), dtype=dtype)  # [L]
    inv_freq = 1.0 / tf.pow(
        10000.0,
        tf.cast(tf.range(0, dim, 2), dtype) / tf.cast(dim, dtype)
    )  # [dim/2]

    # freqs: [L, dim/2]
    freqs = tf.einsum('i,j->ij', pos, inv_freq)
    cos = tf.expand_dims(tf.cos(freqs), axis=0)  # [1, L, dim/2]
    sin = tf.expand_dims(tf.sin(freqs), axis=0)  # [1, L, dim/2]
    return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """
    x:   [B*h, L, dim]
    cos: [1,    L, dim/2]
    sin: [1,    L, dim/2]
    """
    dim = tf.shape(x)[-1]
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)  # [B*h,L,dim/2] + [B*h,L,dim/2]

    cos = tf.cast(cos, x1.dtype)  # 自动 broadcast 到 [B*h,L,dim/2]
    sin = tf.cast(sin, x1.dtype)

    # RoPE 旋转
    x_rotated = tf.concat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)  # [B*h,L,dim]
    return x_rotated

# ========= 多头注意力 + U 门控 =========
class MultiHeadPointwiseAggregatedAttention(tf.layers.Layer):
    """
    多头 scaled dot-product attention（h 个头，单头维 d_k），再用 U 门控融合“逐位置的 V_self”和“聚合的 context”：
        out = g * V_self + (1 - g) * context
    - g 的形态由 gate_dim 决定：
        'scalar': 每个位置标量门控，形状 [B,L,1]
        'vector': 每个位置逐维门控，形状 [B,L,h*d_k]
    说明：
    - 计算遵循 Transformer 规范：d_k = d_model // num_heads，scale = 1/sqrt(d_k)。:contentReference[oaicite:1]{index=1}
    """
    def __init__(self, d_model, num_heads=4, dropout=0.0, gate_dim="scalar", name=None):
        super(MultiHeadPointwiseAggregatedAttention, self).__init__(name=name)
        assert gate_dim in ("scalar", "vector")
        self.d_model = d_model
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.gate_dim = gate_dim

    def build(self, input_shape):
        D = int(input_shape[-1])
        if D % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads: got d_model=%d, num_heads=%d" % (D, self.num_heads))
        self.d_k = D // self.num_heads
        hD = self.num_heads * self.d_k
        init = tf.variance_scaling_initializer(scale=2.0)

        # 线性映射到 h*d_k 后再 reshape 成多头
        self.Wq = self.add_weight("Wq", [D, hD], initializer=init)
        self.Wk = self.add_weight("Wk", [D, hD], initializer=init)
        self.Wv = self.add_weight("Wv", [D, hD], initializer=init)
        # U 门控：按 scalar 或 vector 生成
        if self.gate_dim == "scalar":
            self.Wu = self.add_weight("Wu", [D, 1], initializer=init)       # [D,1]
        else:
            self.Wu = self.add_weight("Wu", [D, hD], initializer=init)      # [D,h*d_k]
        self.Wo = self.add_weight("Wo", [hD, D], initializer=init)          # 合并各头后线性回投
        super(MultiHeadPointwiseAggregatedAttention, self).build(input_shape)

    def call(self, x, attn_mask=None, key_pad_mask=None, training=False):
        # x: [B,L,D]
        B = tf.shape(x)[0]; L = tf.shape(x)[1]; D = self.d_model; h = self.num_heads; d_k = self.d_k
        x2 = tf.reshape(x, [-1, D])  # [B*L, D]

        # 1) 线性映射并 reshape 成多头
        q = tf.matmul(x2, self.Wq); k = tf.matmul(x2, self.Wk); v = tf.matmul(x2, self.Wv)  # [B*L, h*d_k]
        q = tf.reshape(q, [B, L, h, d_k]); k = tf.reshape(k, [B, L, h, d_k]); v = tf.reshape(v, [B, L, h, d_k])
        # 方便批量矩阵乘：合并 batch 与 head
        qh = tf.reshape(tf.transpose(q, [0, 2, 1, 3]), [B * h, L, d_k])  # [B*h, L, d_k]
        kh = tf.reshape(tf.transpose(k, [0, 2, 1, 3]), [B * h, L, d_k])  # [B*h, L, d_k]
        vh = tf.reshape(tf.transpose(v, [0, 2, 1, 3]), [B * h, L, d_k])  # [B*h, L, d_k]

        # === 新增：RoPE 位置编码，对每个 head 的 Q/K 做旋转 ===
        cos, sin = rotary_pos_emb(L, d_k, dtype=qh.dtype)  # [1, L, d_k/2]
        qh = apply_rotary_pos_emb(qh, cos, sin)            # [B*h, L, d_k]
        kh = apply_rotary_pos_emb(kh, cos, sin)            # [B*h, L, d_k]

        # 2) Scaled dot-product attention（按头）: logits=[B*h, L, L]
        scale = tf.rsqrt(tf.cast(d_k, tf.float32))  # 1/sqrt(d_k) 以稳定梯度与 softmax 区分度 :contentReference[oaicite:2]{index=2}
        logits = tf.matmul(qh, kh, transpose_b=True) * scale  # [B*h, L, L]

        # 3) 掩码（加性遮蔽）：causal 和 padding
        if attn_mask is not None:  # attn_mask: [B,L,L] -> [B,1,L,L] -> [B*h,L,L]
            am = tf.expand_dims(attn_mask, axis=1)  # [B,1,L,L]
            am = tf.reshape(tf.tile(am, [1, h, 1, 1]), [B * h, L, L])
            logits = logits + (am - 1.0) * 1e9
        if key_pad_mask is not None:  # key_pad_mask: [B,L] -> [B,1,1,L] -> [B*h,1,L]
            km = tf.expand_dims(tf.expand_dims(key_pad_mask, 1), 1)  # [B,1,1,L]
            km = tf.reshape(tf.tile(km, [1, h, L, 1]), [B * h, L, L])  # broadcast 到 [B*h,L,L]
            logits = logits + (km - 1.0) * 1e9

        weight = tf.nn.softmax(logits, axis=-1)      # [B*h, L, L]
        if self.dropout and training:
            weight = tf.nn.dropout(weight, keep_prob=1.0 - self.dropout)

        context_h = tf.matmul(weight, vh)            # [B*h, L, d_k]
        # 还原为 [B,L,h,d_k] -> 拼接到 [B,L,h*d_k]
        context = tf.transpose(tf.reshape(context_h, [B, h, L, d_k]), [0, 2, 1, 3])  # [B,L,h,d_k]
        context_concat = tf.reshape(context, [B, L, h * d_k])                        # [B,L,h*d_k]

        # pointwise 的 V_self（逐位置值，未聚合）：直接用 v 还原并 concat
        v_self = tf.transpose(tf.reshape(vh, [B, h, L, d_k]), [0, 2, 1, 3])          # [B,L,h,d_k]
        v_self_concat = tf.reshape(v_self, [B, L, h * d_k])                           # [B,L,h*d_k]

        # 4) U 门控：g∈[0,1]，融合 V_self 与 context
        g_pre = tf.matmul(x2, self.Wu)                                                # [B*L, 1] 或 [B*L, h*d_k]
        g = tf.reshape(g_pre, [B, L, -1])                                            # [B,L,1] 或 [B,L,h*d_k]
        g = tf.nn.sigmoid(g)
        if self.gate_dim == "scalar":
            g = tf.tile(g, multiples=[1, 1, h * d_k])                                # [B,L,h*d_k]
        fused = g * v_self_concat + (1.0 - g) * context_concat                       # [B,L,h*d_k]

        # 5) 线性回投到 d_model
        out = tf.reshape(tf.matmul(tf.reshape(fused, [-1, h * d_k]), self.Wo), [B, L, D])
        if self.dropout and training:
            out = tf.nn.dropout(out, keep_prob=1.0 - self.dropout)
        return out

class PositionwiseFFN(tf.layers.Layer):
    def __init__(self, d_model, d_ff, dropout=0.0, name=None):
        super(PositionwiseFFN, self).__init__(name=name)
        self.d_model, self.d_ff, self.dropout = d_model, d_ff, dropout
    def build(self, input_shape):
        D = int(input_shape[-1]); init = tf.variance_scaling_initializer(scale=2.0)
        self.W1 = self.add_weight("W1", [D, self.d_ff], initializer=init)
        self.b1 = self.add_weight("b1", [self.d_ff], initializer=tf.zeros_initializer())
        self.W2 = self.add_weight("W2", [self.d_ff, D], initializer=init)
        self.b2 = self.add_weight("b2", [D], initializer=tf.zeros_initializer())
        super(PositionwiseFFN, self).build(input_shape)
    def call(self, x, training=False):
        B, L, D = tf.shape(x)[0], tf.shape(x)[1], int(x.get_shape()[-1])
        h = tf.matmul(tf.reshape(x, [-1, D]), self.W1) + self.b1
        h = gelu(h)
        if self.dropout and training: h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        h = tf.matmul(h, self.W2) + self.b2
        h = tf.reshape(h, [B, L, D])
        if self.dropout and training: h = tf.nn.dropout(h, keep_prob=1.0 - self.dropout)
        return h

class HSTUBlock(tf.layers.Layer):
    def __init__(self, d_model, d_ff, num_heads=4, attn_dropout=0.0, ffn_dropout=0.0, name=None):
        super(HSTUBlock, self).__init__(name=name)
        self.ln1 = LayerNorm(name="ln1")
        self.attn = MultiHeadPointwiseAggregatedAttention(d_model=d_model, num_heads=num_heads,
                                                          dropout=attn_dropout, gate_dim="scalar", name="mhpaa")
        self.ln2 = LayerNorm(name="ln2")
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff, dropout=ffn_dropout, name="ffn")
    def call(self, x, attn_mask=None, key_pad_mask=None, training=False):
        y = self.ln1(x)
        y = self.attn(y, attn_mask=attn_mask, key_pad_mask=key_pad_mask, training=training)
        x = x + y
        z = self.ln2(x)
        z = self.ffn(z, training=training)
        return x + z

class HSTUEncoder(tf.layers.Layer):
    """堆叠 HSTUBlock；支持：RoPE、causal mask、padding mask、多头注意力"""
    def __init__(self, num_layers, d_model, d_ff, num_heads=4, attn_dropout=0.0, ffn_dropout=0.0, name=None):
        super(HSTUEncoder, self).__init__(name=name)
        self.blocks = [HSTUBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                 attn_dropout=attn_dropout, ffn_dropout=ffn_dropout, name=f"block_{i}")
                       for i in range(num_layers)]
        self.final_ln = LayerNorm(name="final_ln")
    def call(self, x, key_pad_mask, use_causal=True, training=False, pos_emb=None):
        B = tf.shape(x)[0]; L = tf.shape(x)[1]
        if pos_emb is not None: x = x + tf.reshape(pos_emb, [1, L, -1])
        c_mask = causal_mask(B, L) if use_causal else tf.ones([B, L, L], dtype=tf.float32)
        out = x
        for blk in self.blocks:
            out = blk(out, attn_mask=c_mask, key_pad_mask=key_pad_mask, training=training)
        return self.final_ln(out)  # [B,L,D]

def build_hstu_sequences(prepared, embeddings_table, *,
                         seq_cfg, d_model, d_ff, num_layers, num_heads,
                         attn_dropout, ffn_dropout, use_causal, pool_mode,
                         training):
    """
    输入:
      prepared[seq_col]        : [B,L]  的 token（字符串，PAD已转为""）
      prepared[f"{seq_col}__emb"]: [B,L,D] 的 embedding
    返回:
      seq_repr_dict: {seq_col: [B,D]}  每条序列一个向量
      seq_hidden_all: {seq_col: [B,L,D]} 可选的逐位隐藏态（若后续要可视化/对齐）
    """
    seq_repr_dict, seq_hidden_all = {}, {}
    for seq_col, L in seq_cfg.items():
        if f"{seq_col}__emb" not in prepared or seq_col not in prepared:
            continue
        seq_emb = prepared[f"{seq_col}__emb"]        # [B,L,D]
        tokens  = prepared[seq_col]                  # [B,L]
        key_pad = pad_mask_from_tokens(tokens)       # [B,L] {0,1}
        # HSTU 编码
        encoder = HSTUEncoder(num_layers=num_layers, d_model=d_model,
                              d_ff=d_ff, num_heads=num_heads,
                              attn_dropout=attn_dropout, ffn_dropout=ffn_dropout,
                              name=f"{seq_col}_hstu")
        hidden = encoder(seq_emb,
                         key_pad_mask=key_pad,
                         use_causal=use_causal,
                         training=training,
                         pos_emb=None)               # [B,L,D]
        if pool_mode == "target":
            seq_repr = hidden[:, -1, :]             # [B,D]
        else:
            denom = tf.reduce_sum(key_pad, axis=1, keepdims=True) + 1e-12
            seq_repr = tf.reduce_sum(hidden * tf.expand_dims(key_pad, -1), axis=1) / denom
        seq_repr_dict[seq_col]  = seq_repr
        seq_hidden_all[seq_col] = hidden
    return seq_repr_dict, seq_hidden_all

