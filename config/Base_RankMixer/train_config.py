# @File : train_config.py
import os
import sys

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


def _resolve_learning_rate(default_lr=2e-4, cutoff_mmdd="1101", before_lr=3e-4):
    train_date = _extract_train_date(sys.argv)
    if not train_date or len(train_date) < 8:
        return default_lr
    cutoff_date = train_date[:4] + cutoff_mmdd
    return before_lr if train_date <= cutoff_date else default_lr


class TrainConfig:
    """
    Key changes:
    1) Switch model entry to Base_RankMixer (GPU preferred).
    2) Explicitly set seq_length (fixed length) for RankMixer.
    3) Add RankMixer hyperparameters in train_params and align d_model with dynamic Embedding dims.
    """

    # ======================= Basic Info =======================
    model_version = "Base_RankMixer"                           # Required: version name (affects config/output paths)
    model_modul   = "models.rankmixer_main.model_fn"           # RankMixer Estimator entry
    dataset_modul = "dataset.dataset_seq.input_fn"             # Reuse existing TF data pipeline

    ### GPU training parameter configuration
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "0"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth

    # ======================= Training Params (passed to model_fn) =======================
    train_params = {
        # Optimizer config (for RankMixer backbone)
        "optimize_config": {
            # RankMixer prefers small lr + warmup (more stable training)
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
        # ★ RankMixer hyperparameters (must match dynamic Embedding dim)
        "rankmixer": {
            # ===== Paper-aligned defaults =====
            # tokenization: concat e_input, then split into fixed T tokens and project (paper 3.2)
            # token_mixer_type: parameter-free Split+Shuffle+Merge (paper 3.3.1)
            # pooling: mean pooling (paper 3.1)
            "tokenization": "paper",
            "num_tokens": 64,          # Suggest a factor of d_model (requires d_model % num_tokens == 0)
            "token_mixer_type": "paper",
            "pooling": "mean",
            "add_cls_token": False,

            "d_model": 128,
            "num_layers": 8,
            # In paper token mixing, num_heads is set to num_tokens; this only applies to learned mixer
            "num_heads": 8,
            "ffn_mult": 8,             # k in the paper; slightly larger by default to increase capacity
            "token_mixing_dropout": 0.1,
            "ffn_dropout": 0.1,
            "input_dropout": 0.0,
            "head_dropout": 0.0,

            "use_other_features": True,
            "seq_pool": "mean",
            "embedding_size": 9,

            # Group pooling for legacy tokenization (unused in paper mode)
            "dense_token_group_size": 0,
            "dense_token_pool": "mean",

            # CVR auxiliary loss under ESMM decomposition (only on clicked samples), can improve cvr_auc/cvr_pcoc
            "ctr_loss_weight": 1.0,
            "ctcvr_loss_weight": 1.0,
            "cvr_loss_weight": 0.2,

            "use_mmoe": True,
            "mmoe_config": {
                "num_domains": 2,
                "num_experts": 4,
                "expert_units": [128, 64, 128],
                "tower_units": [256, 128],
            },
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
    # seq_feature configuration
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
    # Align with prediction outputs in the model, add keys to form JSON for inference result storage
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                    + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]  # required

    # ======================= Parsing/Compression Config =======================
    field_sep = "\003"  # Field separator
    features_sep = "\002"  # Features separator
    compression_type = "GZIP"  # Data compression format

    # ======================= Offline Features/Bucketing Config =======================
    # Define bucketing and feature selection table
    binning_table_name = "tmp_ad_rank_cvr_activation_sample_data_v2"
    partitions = "ds_date='{day}',durations='1',model_type='TO5'"
    downodps_datas = ['20250901']

    # ======================= Local/OSS Config Paths =======================
    schema_path = f"{dirname}/config/{model_version}/schema.conf"  # Required file
    slot_path = f"{dirname}/config/{model_version}/slot.conf"  # Required file
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"  # Required file
    fg_path = f"{dirname}/config/{model_version}/fg.json"  # Required file
    feature_config_path = f"{dirname}/config/{model_version}/feature_config.json"
    body_json_name = f"{dirname}/config/{model_version}/body.json"

    # ======================= Estimator Runtime Config (aligned with baseline) =======================
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
        "train_batch_size": 2048,
        "train_epoch": 1,
        "batch_size": 2048
    }

    # ======================= Write-back/Export/Metrics =======================
    # Result table for inference data writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "rankmixer_model_test"  # Define OSS path for model export
    oss_offline_root_path = "deep_model/offline"  # OSS path for offline features; used to check online push before export
    # Table for training metrics, yoyo_model only
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr..
    eval_type = "ctcvr"
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "rankmixer"
