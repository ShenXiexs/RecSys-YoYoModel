# models/hstu_main.py
# Responsibilities: embedding lookup (incl. TFRA dynamic tables), call HSTU model, compute loss/metrics, return EstimatorSpec
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
from collections import OrderedDict
# Import the actual functions to use
from models.ctr_hstu_seq import build_hstu_sequences, LayerNorm
# Also import seq_features_config to build seq_length here
from common.utils import select_feature, slots_dict, train_config as TrainConfig, seq_features_config
from common.metrics import evaluate  # Reuse existing evaluation utilities
logger = tf.compat.v1.logging


# ===== Compatibility: if TrainConfig lacks seq_length, build it from seq_features_config =====
if not hasattr(TrainConfig, "seq_length"):
    # Build an ordered dict: {sequence_name -> length}, matching seq_features_config order
    TrainConfig.seq_length = OrderedDict(
        (cfg["name"], cfg["length"])
        for cfg in seq_features_config
        if cfg.get("is_download", 1) == 1
    )


# ========= Helpers: handle SparseTensor, pad/truncate to fixed length =========
def _dense_if_sparse(x, default_value=""):
    """If x is a SparseTensor, convert to dense; otherwise return as-is."""
    return tf.sparse.to_dense(x, default_value=default_value) if isinstance(x, tf.SparseTensor) else x

def _pad_trunc_to_length(tokens_dense, L):
    """
    Convert variable-length 2D string tensor [B, T] to fixed length [B, L]:
      - Truncate to L if longer
      - Pad with "" on the right to L
    """
    T = tf.shape(tokens_dense)[1]
    tokens_cut = tokens_dense[:, :tf.minimum(T, L)]
    pad_len = tf.maximum(0, L - tf.shape(tokens_cut)[1])
    tokens_fix = tf.pad(tokens_cut, paddings=[[0, 0], [0, pad_len]])
    tokens_fix.set_shape([None, L])  # Helps static shape inference
    return tokens_fix

# ========= Dynamic embedding: sequence lookup =========
def _get_seq_embedding(tokens_2d, embeddings_table, policy, name="seq_lookup"):
    """
    tokens_2d: [B, L] (tf.string) -> dynamic embedding lookup -> [B, L, D]
    """
    B = tf.shape(tokens_2d)[0]
    L = tf.shape(tokens_2d)[1]
    flat = tf.reshape(tokens_2d, [-1])                                 # [B*L]
    uniq, idx = tf.unique(flat)
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])  # [U]
    # Use the provided global policy
    update_tstp_op = policy.apply_update(ids)
    restrict_op     = policy.apply_restriction(int(1e8))
    emb_u, trainable_wrapper = de.embedding_lookup(embeddings_table, ids, return_trainable=True, name=name)
    gathered = tf.gather(emb_u, idx)                                    # [B*L, D]
    seq_emb = tf.reshape(gathered, [B, L, embeddings_table.dim])        # [B,L,D]
    return seq_emb, update_tstp_op, restrict_op

# ========= Dynamic embedding: other discrete features (features) =========
def _get_dense_emb_from_features(features, embeddings_table, policy):
    """
    Convert features['features'] (selected slots) to dynamic embeddings and concatenate into [B, F*D].
    """
    x = features["features"]
    x = _dense_if_sparse(x, default_value="")  # Handle SparseTensor
    B = tf.shape(x)[0]
    fea_size = len(select_feature)

    flat = tf.reshape(x, [-1])                       # [B*F]
    uniq, idx = tf.unique(flat)
    ids = tf.strings.to_hash_bucket_strong(uniq, 2 ** 63 - 1, [1, 2])

    # Use the provided global policy
    update_tstp_op = policy.apply_update(ids)
    restrict_op     = policy.apply_restriction(int(1e8))

    emb_u, trainable_wrapper = de.embedding_lookup(
        embeddings_table, ids, return_trainable=True, name="features_lookup")
    gathered = tf.gather(emb_u, idx)                 # [B*F, D]
    dense_emb = tf.reshape(gathered, [B, fea_size * embeddings_table.dim])
    return dense_emb, update_tstp_op, restrict_op

# ========= Estimator entry =========
def compute_ctr_bce_loss(logits, labels, eps=1e-7):
    """
    Standard binary BCE loss:
      logits: [B, 1], raw (unscaled) model logits
      labels: [B, 1] or [B], 0/1 labels
    Returns:
      loss:   scalar loss
      prob:   [B, 1], click/convert probability
    """
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 1])          # [B,1]
    logits = tf.cast(logits, tf.float32)         # [B,1]

    # Sigmoid cross-entropy
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)  # [B,1]
    loss = tf.reduce_mean(ce)

    # Probabilities used for evaluation and export
    prob = tf.nn.sigmoid(logits)                 # [B,1]
    return loss, prob


# ========= Estimator entry =========
def model_fn(features, labels, mode, params):
    """
    Two tasks only:
      1) Prepare HSTU inputs via the new interface (features + seq_features);
      2) Use HSTU for CTR main task + CTCVR head (for logging alignment), keep log/output field names unchanged.
    """
    # ====== Read config ======
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    hcfg = params.get("hstu", {}) if params is not None else {}
    d_model    = int(hcfg.get("d_model", 64))
    num_layers = int(hcfg.get("num_layers", 2))
    d_ff       = int(hcfg.get("d_ff", 4 * d_model))
    num_heads  = int(hcfg.get("num_heads", 4))
    attn_dp    = float(hcfg.get("attn_dropout", 0.0))
    ffn_dp     = float(hcfg.get("ffn_dropout", 0.0))
    use_causal = bool(hcfg.get("causal", True))
    pool_mode  = str(hcfg.get("pool_mode", "target"))
    use_other  = bool(hcfg.get("use_other_features", True))
    ps_num     = int(params.get("ps_num", 0)) if params else 0
    restrict   = bool(params.get("restrict", False)) if params else False

    # ====== Device & dynamic tables ======
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
        name="embeddings", dim=d_model, devices=devices_info,
        trainable=is_training, initializer=initializer)

    # ====== Key change: create global policy ======
    policy = tfra.dynamic_embedding.TimestampRestrictPolicy(embeddings_table)
    update_tstp_op = policy.apply_update(tf.constant([0], dtype=tf.int64))  # dummy init
    restrict_op    = policy.apply_restriction(int(1e8))
    groups = [update_tstp_op] + ([restrict_op] if restrict else [])

    # ====== 1) Other discrete feature embeddings (same as original logic) ======
    update_ops = []
    if use_other:
        other_emb, up_t, rs_t = _get_dense_emb_from_features(features, embeddings_table, policy)
        update_ops += [up_t, rs_t]
    else:
        other_emb = None

    # ====== 2) Slice sequences from flat seq_features -> dynamic lookup ======
    seq_cfg = TrainConfig.seq_length
    prepared = dict(features)

    if "seq_features" not in features:
        raise ValueError("Missing 'seq_features' in features; please verify the dataset outputs the new interface.")
    seq_features_flat = _dense_if_sparse(features["seq_features"], default_value="0")  # [B, sum(L)]
    B = tf.shape(seq_features_flat)[0]

    start = 0
    for seq_col, L in seq_cfg.items():
        tokens_slice = seq_features_flat[:, start:start+L]  # [B,L]
        start += L
        # Replace "0" with ""; note all shapes are [B,L]
        mask = tf.equal(tokens_slice, tf.constant("0", dtype=tf.string))  # [B,L]
        empty_tokens = tf.fill(tf.shape(tokens_slice), "")               # [B,L]
        tokens_slice = tf.where(mask, empty_tokens, tokens_slice)         # [B,L]
        tokens = _pad_trunc_to_length(tokens_slice, L)      # [B,L]
        # Use the global policy
        seq_emb, up_s, rs_s = _get_seq_embedding(tokens, embeddings_table, policy, name=f"{seq_col}_lookup")
        update_ops.append(up_s)
        if restrict:
            update_ops.append(rs_s)
        prepared[seq_col] = tokens
        prepared[f"{seq_col}__emb"] = seq_emb

    # ====== 3) HSTU: get representations for each sequence ======
    seq_repr_dict, _ = build_hstu_sequences(
        prepared, embeddings_table,
        seq_cfg=seq_cfg, d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads,
        attn_dropout=attn_dp, ffn_dropout=ffn_dp, use_causal=use_causal, pool_mode=pool_mode,
        training=is_training
    )

    # ====== 4) Head: concat dense features + sequence reps -> LN -> two heads (CTR/CTCVR) ======
    chunks = []
    if use_other and other_emb is not None:
        chunks.append(other_emb)                          # [B, F*D]
    for col in seq_cfg.keys():
        if col in seq_repr_dict:
            chunks.append(seq_repr_dict[col])             # [B, D]
    if not chunks:
        raise ValueError("No available features: check TrainConfig.seq_length or use_other_features.")
    h = tf.concat(chunks, axis=1)

    with tf.compat.v1.variable_scope("head_ln"):
        ln = LayerNorm(name="ln")
        h = ln(h)

    # CTR head (BCE)
    ctr_logit = tf.compat.v1.layers.dense(
        h, units=1, activation=None, name="ctr_logit"
    )  # [B,1]

    # CTCVR head (BCE)
    ctcvr_logit = tf.compat.v1.layers.dense(
        h, units=1, activation=None, name="ctcvr_logit"
    )  # [B,1]

    # ====== 5) Fetch labels (compatible with legacy ESMM fields) ======
    def _pick_label(candidates):
        x = None
        for k in candidates:
            if k in features:
                x = features[k]
                break
        if x is None and labels is not None:
            if isinstance(labels, dict):
                for k in candidates:
                    if k in labels:
                        x = labels[k]
                        break
            else:
                x = labels
        return x

    # CTR
    ctr_label_raw = _pick_label(["click_label", "ctr_label", "is_click"])
    if ctr_label_raw is None:
        ctr_label_raw = tf.zeros((tf.shape(ctr_logit)[0],), tf.float32)
    ctr_label_raw = tf.reshape(tf.cast(ctr_label_raw, tf.float32), [-1])   # [B]
    ctr_label = tf.reshape(ctr_label_raw, [-1, 1])                          # [B,1]

    # CTCVR
    ctcvr_label_raw = _pick_label(["ctcvr_label", "is_conversion"])
    has_ctcvr = ctcvr_label_raw is not None
    if ctcvr_label_raw is None:
        ctcvr_label_raw = tf.zeros((tf.shape(ctcvr_logit)[0],), tf.float32)
    ctcvr_label = tf.reshape(tf.cast(ctcvr_label_raw, tf.float32), [-1, 1])  # [B,1]

    # ====== 6) Loss (total = CTR + lambda*CTCVR; if no ctcvr label, ctcvr_loss=0) ======
    lam = float(params.get("ctcvr_loss_weight", 1.0)) if params else 1.0

    # CTR BCE
    ctr_loss, ctr_prob = compute_ctr_bce_loss(ctr_logit, ctr_label)  # [B,1]

    # CTCVR BCE (if no label, output prob only, no loss)
    if has_ctcvr:
        ctcvr_loss, ctcvr_prob = compute_ctr_bce_loss(ctcvr_logit, ctcvr_label)
    else:
        zero_label = tf.zeros_like(ctcvr_label)
        _, ctcvr_prob = compute_ctr_bce_loss(ctcvr_logit, zero_label)
        ctcvr_loss = tf.constant(0.0, dtype=tf.float32)

    total_loss = ctr_loss + lam * ctcvr_loss


    # ====== 7) Metrics (field names aligned with original ESMM) ======
    eval_metric_ops = OrderedDict()
    evaluate(tf.reshape(ctr_label_raw, [-1, 1]), ctr_prob, "task1_ctr",   eval_metric_ops)
    evaluate(ctcvr_label,                         ctcvr_prob, "task1_ctcvr", eval_metric_ops)

    # ====== 8) Log fields fully aligned ======
    global_step = tf.compat.v1.train.get_or_create_global_step()
    loggings = OrderedDict({
        "step": global_step,
        "emb_size": embeddings_table.size(),
        "losses": total_loss,        # Same name as original ESMM
        "ctr_losses": ctr_loss,      # Same name as original ESMM
        "ctcvr_losses": ctcvr_loss,  # Same name as original ESMM
    })
    for k, (val, up_op) in eval_metric_ops.items():
        loggings[k] = val
        groups.append(up_op)
    # Chain embedding timestamp/restriction updates with metrics updates
    groups.extend(update_ops)

    # ====== 9) predictions & export (key alignment) ======
    batch_size = tf.shape(ctr_prob)[0]
    predictions = {
        "requestid": features.get("requestid", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "combination_un_id": features.get("combination_un_id", tf.as_string(tf.zeros((batch_size,), tf.int16))),
        "out": tf.concat([tf.reshape(ctr_label_raw, [-1, 1]), ctr_prob], axis=1)
    }

    # Provide a model-style container to keep legacy exports aligned
    class _ModelView(object):
        pass
    model = _ModelView()
    model.losses        = total_loss
    model.ctr_losses    = ctr_loss
    model.ctcvr_losses  = ctcvr_loss
    model.ctr_labels    = [tf.reshape(ctr_label_raw, [-1, 1])]
    model.ctr_probs     = [ctr_prob]
    model.ctcvr_labels  = [ctcvr_label]
    model.ctcvr_probs   = [ctcvr_prob]
    model.predictions   = predictions
    model.outputs       = {  # Fallback compatibility: out/ctr_output/ctcvr_output
        "out": tf.concat([tf.reshape(ctr_label_raw, [-1, 1]), ctr_prob], axis=1),
        "ctr_output": ctr_prob,
        "ctcvr_output": ctcvr_prob
    }

    # ====== 10) Keep the three modes consistent with trainer ======
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {"serving_default": tf.compat.v1.estimator.export.PredictOutput(model.outputs)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                          loss=model.losses, eval_metric_ops=eval_metric_ops)

    # TRAIN
    trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
    l2_reg = float(params.get("l2_reg", 1e-6)) if params else 1e-6
    if l2_reg > 0:
        l2_loss = l2_reg * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                                     for v in trainable_variables if 'bias' not in v.name])
        model.losses = model.losses + l2_loss
        loggings["l2_loss"] = l2_loss

    opt = tf.compat.v1.train.AdamOptimizer()
    opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(opt)
    dense_op = opt.minimize(model.losses, global_step=global_step)
    train_op = tf.group(dense_op, *groups)

    log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=100)
    return tf.estimator.EstimatorSpec(mode=mode, predictions=model.predictions,
                                      loss=model.losses, train_op=train_op, training_hooks=[log_hook])
