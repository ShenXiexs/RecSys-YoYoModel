import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    """
    Key changes:
    1) Switch model entry to RankMixer (GPU preferred)
    2) Explicitly set seq_length (fixed length) for RankMixer
    3) Add rankmixer hyperparameters in train_params and align d_model with dynamic Embedding dims
    """

    # Basic info
    model_version = "RankMixer_Refined"                        # Version name
    model_modul   = "models.rankmixer_main_refined.model_fn"   # RankMixer Estimator entry
    dataset_modul = "dataset.dataset_seq.input_fn"             # Reuse existing TF data pipeline
    keep_date_ranges = [("1101", "1110")]                      # Keep data for date ranges (YYYYMMDD or MMDD)

    ### GPU training parameter configuration
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "1"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth

    # Training params (passed to model_fn)
    train_params = {
        # Optimizer config (for RankMixer backbone)
        "optimize_config": {
            "learning_rate": 0.001,
            "lr_schedule": {
                "cutoff_date": "20251101",
                "before": 0.001,
                "after": 0.0001
            },
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        # ★ RankMixer hyperparameters (must match dynamic Embedding dim)
        "rankmixer": {
            "d_model": 128,
            "num_layers": 2,
            "num_heads": 16,
            "ffn_mult": 2,
            "token_mixing_dropout": 0.0,
            "ffn_dropout": 0.0,
            "use_other_features": True,
            "seq_pool": "mean",
            "embedding_size": 9,
            "dense_token_group_size": 0,
            "dense_token_pool": "mean",
            "token_mixing_type": "param_free",
            "ln_style": "post",
            "use_final_ln": False,
            "tokenization": "semantic",
            "semantic_target_tokens": 16,
            "semantic_token_pool": "concat_proj",
            "semantic_proj_dim": 128,
            "include_seq_in_tokenization": True,
            "add_cls_token": False,
            "input_ln": False,
            "summary_pooling": "mean",
            "summary_exclude_cls": True,
            "use_moe": False,
            "moe_num_experts": 4,
            "moe_l1_coef": 1e-4,
            "moe_use_dtsi": True,
            "use_mmoe": True,
            "mmoe_config": {
                "num_domains": 2,
                "num_experts": 4,
                "expert_units": [128, 64, 128],
                "tower_units": [256, 128],
            }
        },

        # Resource and dynamic table strategy (aligned with baseline)
        "ps_num": 1,                  # Place TFRA dynamic tables on GPU:0 to avoid CPU fallback
        "restrict": True,             # Control dynamic table size to keep memory manageable
        "l2_reg": 1e-6,
    }

    # Data Schema
    data_schema = ["user_id", "requestid", "combination_un_id", "is_click", "is_conversion", "features", "app_pkg_src", "app_pkg", "app_first_type", "seq_features" ] # required
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}                                                        # required
    # seq_feature configuration
    seq_features_config = [
        # invisible for satey reasons
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

    # Dataset input_fn Config
    data_nm = "TO5"
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        },
        "train_batch_size": 512,
        "train_epoch": 1,
        "batch_size": 512
    }

    # Write-back/Export/Metrics
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

