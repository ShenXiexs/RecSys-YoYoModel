# @File : train_config.py
import copy
import os
import sys
import re

# Keep the path hierarchy consistent with the original project (config/<model_version>/...).
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _extract_train_date(argv):
    for flag in ("--end_time_str", "--time_str"):
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return str(argv[i + 1])[:8]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1][:8]
    return None


def _resolve_learning_rate(default_lr=2e-4, cutoff_mmdd="1102", before_lr=3e-4):
    train_date = _extract_train_date(sys.argv)
    if not train_date or len(train_date) < 8:
        return default_lr
    cutoff_date = train_date[:4] + cutoff_mmdd
    return before_lr if train_date <= cutoff_date else default_lr



# === Semantic groups built from TOP5_V2 features ===
_SEMANTIC_GROUP_RULES = [
    ("seq_awake", [r"^user_awake_"]),
    ("seq_first_awke", [r"^user_first_awke_"]),
    ("seq_imp_app", [r"^user_imp_app_"]),
    ("seq_clk_app", [r"^user_clk_app_"]),
    ("seq_imp_launch", [r"^user_imp_launch_"]),
    ("seq_clk_launch", [r"^user_clk_launch_"]),

    ("doc_key_one", [r"^doc__key_one__"]),
    ("doc_key_two", [r"^doc__key_two__"]),
    ("doc_key_three", [r"^doc__key_three__"]),
    ("doc_key_four", [r"^doc__key_four__"]),
    ("doc_key_five", [r"^doc__key_five__"]),
    ("doc_key_six", [r"^doc__key_six__"]),
    ("doc_key_seven", [r"^doc__key_seven__"]),

    ("user_cnt", [r"^user__imp_cnt_", r"^user__clk_cnt_", r"^user__clk_div_imp_cnt_"]),
    ("user_kv", [r"^user__kv_"]),

    ("match_flags", [r"^is_match_"]),

    ("ad_ids", [
        r"^ad_idea_id$", r"^ad_unit_id$", r"^ad_plan_id$", r"^combination_un_id$",
        r"^template_id$", r"^template_type$", r"^promotion_type$", r"^target_type$", r"^is_new_item$",
    ]),
    ("creative_text", [r"^title$", r"^content$", r"^keywords$"]),
    ("app_task", [
        r"^package_name$", r"^app_first_type$", r"^app_second_type$", r"^log10_app_size$",
        r"^product_name$", r"^first_industry_id$", r"^industry$", r"^rta_type$", r"^rta_product_code$",
        r"^crowd_type$", r"^white_crowd_code$", r"^package_channel_type$", r"^dispatch_center_id$",
    ]),
    ("context_src", [r"^app_pkg_src$", r"^app_src_first_type$", r"^app_src_second_type$"]),
    ("context_traffic", [r"^adslot_id$", r"^channel_id$", r"^ssp_adslot_id$", r"^model_type$"]),
    ("context_device", [r"^device_", r"^network$", r"^ip_region$", r"^ip_city$"]),
    ("context_time", [r"^day_h$"]),
]


def _load_select_features(path):
    features = []
    if not path:
        return features
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            features.append(line)
    return features


def _build_semantic_groups(select_feat_path, seq_features_config, rules=_SEMANTIC_GROUP_RULES):
    dense_features = _load_select_features(select_feat_path)
    seq_features = [cfg.get("name") for cfg in (seq_features_config or []) if cfg.get("name")]
    feature_names = dense_features + seq_features

    used = set()
    groups = []
    for name, patterns in rules:
        regexes = [re.compile(p) for p in patterns]
        matched = [
            f for f in feature_names
            if f not in used and any(r.search(f) for r in regexes)
        ]
        if matched:
            groups.append({"name": name, "features": matched})
            used.update(matched)

    leftovers = [f for f in feature_names if f not in used]
    if leftovers:
        if groups:
            groups[-1]["features"].extend(leftovers)
        else:
            groups.append({"name": "other", "features": leftovers})
    return groups


class TrainConfig(object):
    """
    Key changes:
    1) Switch model entry to RankMixer (GPU preferred).
    2) Explicitly set seq_length (fixed length) for RankMixer.
    3) Add rankmixer hyperparameters in train_params and align d_model with dynamic Embedding dims.
    """

    # ======================= Basic Info =======================
    model_version = "RankMixer_Shen0202"                       # Required: version name (affects config/output paths)
    model_modul   = "models.rankmixer_shen0202.model_fn"       # RankMixer_Shen0202 Estimator entry
    dataset_modul = "dataset.dataset_seq.input_fn"             # Reuse existing TF data pipeline

    ### GPU training parameter configuration
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "0"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth
    keep_date_ranges = [("1101", "1110")]
    
    # ======================= Training Params (passed to model_fn) =======================
    train_params = {
        "use_seq_features": "v1",  # Default does not use seq features. choice:[v1,v2]; v1: pad+truncate; v2: fixed length.
        # Optimizer config (for RankMixer backbone)
        "optimize_config": {
            "learning_rate": _resolve_learning_rate(),
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "warmup_steps": 4000,
            "decay_type": "none",
            "decay_steps": 0,
            "min_learning_rate": 0.0,
            "grad_clip_norm": 5.0,
        },
        # RankMixer_Shen0118 hyperparameters (paper 100M configuration)
        "rankmixer": {
            # ===== Model scale =====
            "d_model": 368,
            "num_layers": 8,
            "num_tokens": 23,
            "num_heads": 23,  # Paper requires H = T
            "ffn_mult": 8,

            # ===== Tokenization =====
            "tokenization_strategy": "semantic",
            "tokenization_version": "v3",
            "semantic_target_tokens": 23,
            "semantic_groups": None,  # auto-filled from TOP5_V2 features
            "token_projection": "linear",
            "include_seq_in_tokenization": True,

            # ===== Token Mixing (paper strict) =====
            "token_mixing_type": "paper_strict",
            "token_mixing_dropout": 0.0,

            # ===== Per-token FFN =====
            "per_token_ffn": True,
            "ffn_activation": "gelu",
            "ffn_dropout": 0.0,

            # ===== LN style =====
            "ln_style": "pre",
            "input_ln": False,
            "use_final_ln": True,

            # ===== Pooling / Head =====
            "output_pooling": "mean",
            "add_cls_token": False,
            "head_dropout": 0.0,

            # ===== MoE (disabled for 100M) =====
            "use_moe": False,
            "moe_num_experts": 8,
            "moe_sparsity_ratio": 0.125,
            "moe_routing_type": "relu_dtsi",
            "moe_l1_lambda": 0.01,
            "moe_use_dtsi": True,

            # ===== Other base settings =====
            "use_other_features": True,
            "seq_pool": "mean",
            "embedding_size": 9,
            "input_dropout": 0.0,

            # ===== Loss =====
            "use_ctcvr_loss": True,
            "use_conditional_cvr": False,
        },

        # Resource and dynamic table strategy (aligned with baseline)
        "ps_num": 1,                  # Place TFRA dynamic tables on GPU:0 to avoid CPU fallback
        "restrict": True,             # Control dynamic table size to keep memory manageable
        "l2_reg": 1e-6,
    }

    # ======================= Data Schema =======================
    data_schema = ["user_id", "requestid", "combination_un_id", "is_click", "is_conversion", "features", "app_pkg_src", "app_pkg", "app_first_type", "seq_features" ] # required
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}                                                        # required
    # seq_feature config
    seq_features_config = [
        {"name": "user_awake_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User awake package sequence over the past 90 days"},
        {"name": "user_awake_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User awake package first-level app category sequence over the past 90 days"},
        {"name": "user_awake_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 10 awake events sequence over the past 30 days"},
        {"name": "user_awake_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent awake packages sequences over the past 30 days"},
        {"name": "user_awake_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness awake packages sequences over the past 30 days"},
        {"name": "user_first_awke_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User first-awake package sequence over the past 90 days"},
        {"name": "user_first_awke_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User first-awake package first-level app category sequence over the past 90 days"},
        {"name": "user_first_awke_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 10 first-awake events sequence over the past 30 days"},
        {"name": "user_first_awke_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent first-awake packages sequences over the past 30 days"},
        {"name": "user_first_awke_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness first-awake packages sequences over the past 30 days"},
        {"name": "user_imp_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User impression package sequence over the past 90 days"},
        {"name": "user_imp_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User impression package first-level app category sequence over the past 90 days"},
        {"name": "user_imp_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 10 impression packages sequence over the past 30 days"},
        {"name": "user_imp_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression packages sequences over the past 30 days"},
        {"name": "user_imp_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression packages sequences over the past 30 days"},
        {"name": "user_clk_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User click package sequence over the past 90 days"},
        {"name": "user_clk_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User click package first-level app category sequence over the past 90 days"},
        {"name": "user_clk_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 10 click packages sequence over the past 30 days"},
        {"name": "user_clk_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click packages sequences over the past 30 days"},
        {"name": "user_clk_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click packages sequences over the past 30 days"},
        {"name": "user_imp_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User most recent 5 impression launches sequence"},
        {"name": "user_imp_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression launches sequences over the past 30 days"},
        {"name": "user_imp_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression launches sequences over the past 30 days"},
        {"name": "user_clk_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User most recent 5 click launches sequence"},
        {"name": "user_clk_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click launches sequences over the past 30 days"},
        {"name": "user_clk_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click launches sequences over the past 30 days"},
        {"name": "user_awake_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 5 awake events sequence over the past 7 days"},
        {"name": "user_awake_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent awake events sequences over the past 15 days"},
        {"name": "user_awake_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent awake events sequences over the past 7 days"},
        {"name": "user_awake_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 most frequent awake events sequences over the past 1 day"},
        {"name": "user_awake_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent awake package first-level app categories sequences over the past 15 days"},
        {"name": "user_awake_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent awake package first-level app categories sequences over the past 7 days"},
        {"name": "user_awake_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 most frequent awake package first-level app categories sequences over the past 1 day"},
        {"name": "user_awake_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness awake events sequences over the past 15 days"},
        {"name": "user_awake_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness awake events sequences over the past 7 days"},
        {"name": "user_awake_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness awake events sequences over the past 1 day"},
        {"name": "user_awake_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness awake package first-level app categories sequences over the past 15 days"},
        {"name": "user_awake_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness awake package first-level app categories sequences over the past 7 days"},
        {"name": "user_awake_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness awake package first-level app categories sequences over the past 1 day"},
        {"name": "user_first_awke_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 5 first-awake events sequence over the past 7 days"},
        {"name": "user_first_awke_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent first-awake events sequences over the past 15 days"},
        {"name": "user_first_awke_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent first-awake events sequences over the past 7 days"},
        {"name": "user_first_awke_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 most frequent first-awake events sequences over the past 1 day"},
        {"name": "user_first_awke_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 most frequent first-awake package first-level app categories sequences over the past 15 days"},
        {"name": "user_first_awke_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 most frequent first-awake package first-level app categories sequences over the past 7 days"},
        {"name": "user_first_awke_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 most frequent first-awake package first-level app categories sequences over the past 1 day"},
        {"name": "user_first_awke_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness first-awake events sequences over the past 15 days"},
        {"name": "user_first_awke_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness first-awake events sequences over the past 7 days"},
        {"name": "user_first_awke_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness first-awake events sequences over the past 1 day"},
        {"name": "user_first_awke_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 highest-stickiness first-awake package first-level app categories sequences over the past 15 days"},
        {"name": "user_first_awke_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 highest-stickiness first-awake package first-level app categories sequences over the past 7 days"},
        {"name": "user_first_awke_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 5 highest-stickiness first-awake package first-level app categories sequences over the past 1 day"},
        {"name": "user_imp_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 5 impression packages sequence over the past 7 days"},
        {"name": "user_imp_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression packages sequences over the past 15 days"},
        {"name": "user_imp_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression packages sequences over the past 7 days"},
        {"name": "user_imp_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 most frequent impression packages sequences over the past 1 day"},
        {"name": "user_imp_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression package first-level app categories sequences over the past 15 days"},
        {"name": "user_imp_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression package first-level app categories sequences over the past 7 days"},
        {"name": "user_imp_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 most frequent impression package first-level app categories sequences over the past 1 day"},
        {"name": "user_imp_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression packages sequences over the past 15 days"},
        {"name": "user_imp_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression packages sequences over the past 7 days"},
        {"name": "user_imp_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness impression packages sequences over the past 1 day"},
        {"name": "user_imp_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 highest-stickiness impression package first-level app categories sequences over the past 15 days"},
        {"name": "user_imp_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression package first-level app categories sequences over the past 7 days"},
        {"name": "user_imp_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness impression package first-level app categories sequences over the past 1 day"},
        {"name": "user_clk_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User most recent 5 click packages sequence over the past 7 days"},
        {"name": "user_clk_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click packages sequences over the past 15 days"},
        {"name": "user_clk_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click packages sequences over the past 7 days"},
        {"name": "user_clk_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 most frequent click packages sequences over the past 1 day"},
        {"name": "user_clk_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click package first-level app categories sequences over the past 15 days"},
        {"name": "user_clk_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click package first-level app categories sequences over the past 7 days"},
        {"name": "user_clk_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 most frequent click package first-level app categories sequences over the past 1 day"},
        {"name": "user_clk_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click packages sequences over the past 15 days"},
        {"name": "user_clk_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click packages sequences over the past 7 days"},
        {"name": "user_clk_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness click packages sequences over the past 1 day"},
        {"name": "user_clk_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "User top 10 highest-stickiness click package first-level app categories sequences over the past 15 days"},
        {"name": "user_clk_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click package first-level app categories sequences over the past 7 days"},
        {"name": "user_clk_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness click package first-level app categories sequences over the past 1 day"},
        {"name": "user_imp_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 20 most frequent impression launches sequences over the past 15 days"},
        {"name": "user_imp_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 most frequent impression launches sequences over the past 7 days"},
        {"name": "user_imp_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 5 most frequent impression launches sequences over the past 1 day"},
        {"name": "user_imp_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 20 highest-stickiness impression launches sequences over the past 15 days"},
        {"name": "user_imp_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness impression launches sequences over the past 7 days"},
        {"name": "user_imp_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness impression launches sequences over the past 1 day"},
        {"name": "user_clk_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 20 most frequent click launches sequences over the past 15 days"},
        {"name": "user_clk_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 most frequent click launches sequences over the past 7 days"},
        {"name": "user_clk_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 5 most frequent click launches sequences over the past 1 day"},
        {"name": "user_clk_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 20 highest-stickiness click launches sequences over the past 15 days"},
        {"name": "user_clk_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 10 highest-stickiness click launches sequences over the past 7 days"},
        {"name": "user_clk_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "User top 5 highest-stickiness click launches sequences over the past 1 day"},
    ]

    # ======================= Label Mapping / Prediction Output Columns =======================
    # Align with model prediction output; add keys to build JSON for inference result storage
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                    + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]  # required

    # ======================= Parsing/Compression Config =======================
    field_sep = "\003"  # Field separator
    features_sep = "\002"  # Features separator
    compression_type = "GZIP"  # Data compression format

    # ======================= Offline Feature/Bucket Config =======================
    # Define bucketed table with feature selection
    binning_table_name = "tmp_ad_rank_cvr_activation_sample_data_v2"
    partitions = "ds_date='{day}',durations='1',model_type='TO5'"
    downodps_datas = ['20250901']

    # ======================= Local/OSS Paths =======================
    schema_path = f"{dirname}/config/{model_version}/schema.conf"  # Required
    slot_path = f"{dirname}/config/{model_version}/slot.conf"  # Required
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"  # Required
    fg_path = f"{dirname}/config/{model_version}/fg.json"  # Required
    feature_config_path = f"{dirname}/config/{model_version}/feature_config.json"
    body_json_name = f"{dirname}/config/{model_version}/body.json"

    # ======================= Estimator Run Config (aligned with baseline) =======================
    es_run_config = {
        "keep_checkpoint_max": 1,
        "save_checkpoints_steps": 100000,
        "log_step_count_steps": 5000,
        "save_summary_steps": 10000
    }

    # ======================= Dataset input_fn Config =======================
    data_nm = "TO5"
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        },
        "train_batch_size": 1024,
        "train_epoch": 1,
        "batch_size": 1024
    }

    # ======================= Writeback/Export/Metrics =======================
    # Result table for inference writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "rankmixer_shen0202_model_test"  # Model export OSS path
    oss_offline_root_path = "deep_model/offline"  # Offline feature OSS path; check feature push before pushing model
    # Model training metrics table (yoyo_model only)
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr, etc.
    eval_type = "ctcvr"
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "rankmixer_shen0202"


TrainConfig.train_params["rankmixer"]["semantic_groups"] = _build_semantic_groups(
    TrainConfig.sel_feat_path, TrainConfig.seq_features_config
)


class TrainConfig_1B(TrainConfig):
    model_version = "RankMixer_Shen0202"
    train_params = copy.deepcopy(TrainConfig.train_params)
    train_params["rankmixer"].update({
        "d_model": 368,
        "num_tokens": 23,
        "num_heads": 23,
        "use_moe": True,
        "moe_num_experts": 8,
        "moe_sparsity_ratio": 0.125,
        "moe_routing_type": "relu_dtsi",
        "moe_l1_lambda": 0.01,
    })
