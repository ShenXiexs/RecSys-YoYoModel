# @File : train_config.py
import os

# Keep the path hierarchy consistent with the original project (config/<model_version>/...).
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    """
    Key changes:
    1) Use HSTU: model_modul -> models.hstu_main.model_fn
    2) Explicitly set seq_length (fixed length) for HSTU
    3) Add "hstu" hyperparameters in train_params and align d_model with dynamic Embedding dims
    """

    # ======================= Basic Info =======================
    model_version = "Base_hstu"                                # Required: version name (affects config/output paths)
    model_modul   = "models.hstu_main.model_fn"               # Switched to HSTU Estimator entry
    dataset_modul = "dataset.dataset_seq.input_fn"                # Reuse existing TF data pipeline

    ### GPU training parameter configuration
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "1"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth

    # ======================= Training Params (passed to model_fn) =======================
    train_params = {
            # Optimizer config (used by hstu_loss.model_fn)
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        # ★ HSTU hyperparameters (must match dynamic Embedding dim)
        "hstu": {
            "d_model": 12,            # Embedding/model hidden dim; must match TFRA dynamic table dim
            "num_layers": 4,          # HSTU layers
            "num_heads": 3,           # HSTU heads
            "d_ff": 64,              # FFN hidden dim
            "attn_dropout": 0.1,      # Attention dropout
            "ffn_dropout": 0.1,       # FFN dropout
            "causal": True,           # Autoregressive (causal mask)
            "pool_mode": "target",    # "target" = last step; or "mean" = mask average
            "use_other_features": True  # Whether to concat non-sequence feature embeddings
        },

        # Resource and dynamic table strategy (aligned with baseline)
        "ps_num": 0,                  # Parameter server count (0 if no multi-host/multi-GPU)
        "restrict": False,            # Enable TFRA restrict policy
        "l2_reg": 1e-6,               # L2 regularization (tunable)
    }

    # ======================= Data Schema =======================
    data_schema = ["user_id", "requestid", "combination_un_id", "is_click", "is_conversion", "features", "app_pkg_src", "app_pkg", "app_first_type", "seq_features" ] # required
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}                                                        # required
    # seq_feature configuration
    seq_features_config = [
        {"name": "user_awake_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天唤醒包序列"},
        {"name": "user_awake_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天唤醒包一级分类序列"},
        {"name": "user_awake_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日最近10次唤醒序列"},
        {"name": "user_awake_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日唤醒包频次最多10个序列"},
        {"name": "user_awake_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个唤醒包序列"},
        {"name": "user_first_awke_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天首唤包序列"},
        {"name": "user_first_awke_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天首唤包一级分类序列"},
        {"name": "user_first_awke_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日最近10次首唤序列"},
        {"name": "user_first_awke_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日首唤包频次最多10个序列"},
        {"name": "user_first_awke_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个首唤包序列"},
        {"name": "user_imp_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天曝光包序列"},
        {"name": "user_imp_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天曝光包应用一级分类序列"},
        {"name": "user_imp_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30最近10次曝光包序列"},
        {"name": "user_imp_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日曝光包频次最多10个序列"},
        {"name": "user_imp_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个曝光包序列"},
        {"name": "user_clk_app_90d_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户90天点击包序列"},
        {"name": "user_clk_app_90d_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户90天点击包应用一级分类序列"},
        {"name": "user_clk_app_30d_recent_10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30最近10次点击包序列"},
        {"name": "user_clk_app_30d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日点击包频次最多10个序列"},
        {"name": "user_clk_app_30d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个点击包序列"},
        {"name": "user_imp_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户最近5次曝光启动序列"},
        {"name": "user_imp_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日曝光启动频次最多10个序列"},
        {"name": "user_imp_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个曝光启动序列"},
        {"name": "user_clk_launch_recent_5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户最近5次点击启动序列"},
        {"name": "user_clk_launch_30d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日点击启动频次最多10个序列"},
        {"name": "user_clk_launch_30d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户30日粘性最高10个点击启动序列"},
        {"name": "user_awake_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次唤醒序列"},
        {"name": "user_awake_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日唤醒频次最多10个序列"},
        {"name": "user_awake_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日唤醒频次最多10个序列"},
        {"name": "user_awake_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日唤醒频次最多5个序列"},
        {"name": "user_awake_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日唤醒包应用一级分类频次最多10个序列"},
        {"name": "user_awake_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日唤醒包应用一级分类频次最多10个序列"},
        {"name": "user_awake_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日唤醒包应用一级分类频次最多5个序列"},
        {"name": "user_awake_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个唤醒序列"},
        {"name": "user_awake_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个唤醒序列"},
        {"name": "user_awake_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个唤醒序列"},
        {"name": "user_awake_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个唤醒包应用一级分类序列"},
        {"name": "user_awake_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个唤醒包应用一级分类序列"},
        {"name": "user_awake_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个唤醒包应用一级分类序列"},
        {"name": "user_first_awke_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次首唤序列"},
        {"name": "user_first_awke_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日首唤频次最多10个序列"},
        {"name": "user_first_awke_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日首唤频次最多10个序列"},
        {"name": "user_first_awke_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日首唤频次最多5个序列"},
        {"name": "user_first_awke_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日首唤包应用一级分类频次最多10个序列"},
        {"name": "user_first_awke_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户7日首唤包应用一级分类频次最多10个序列"},
        {"name": "user_first_awke_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日首唤包应用一级分类频次最多5个序列"},
        {"name": "user_first_awke_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个首唤序列"},
        {"name": "user_first_awke_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个首唤序列"},
        {"name": "user_first_awke_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个首唤序列"},
        {"name": "user_first_awke_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个首唤包应用一级分类序列"},
        {"name": "user_first_awke_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户7日粘性最高10个首唤包应用一级分类序列"},
        {"name": "user_first_awke_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户1日粘性最高5个首唤包应用一级分类序列"},
        {"name": "user_imp_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次曝光包序列"},
        {"name": "user_imp_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日曝光包频次最多10个序列"},
        {"name": "user_imp_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日曝光包频次最多10个序列"},
        {"name": "user_imp_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日曝光包频次最多5个序列"},
        {"name": "user_imp_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日曝光包应用一级分类频次最多10个序列"},
        {"name": "user_imp_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日曝光包应用一级分类频次最多10个序列"},
        {"name": "user_imp_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日曝光包应用一级分类频次最多5个序列"},
        {"name": "user_imp_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个曝光包序列"},
        {"name": "user_imp_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光包序列"},
        {"name": "user_imp_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光包序列"},
        {"name": "user_imp_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个曝光包应用一级分类序列"},
        {"name": "user_imp_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光包应用一级分类序列"},
        {"name": "user_imp_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光包应用一级分类序列"},
        {"name": "user_clk_app_7d_recent_5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日最近5次点击包序列"},
        {"name": "user_clk_app_15d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日点击包频次最多10个序列"},
        {"name": "user_clk_app_7d_freq_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日点击包频次最多10个序列"},
        {"name": "user_clk_app_1d_freq_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日点击包频次最多5个序列"},
        {"name": "user_clk_app_15d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户15日点击包应用一级分类频次最多10个序列"},
        {"name": "user_clk_app_7d_freq_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日点击包应用一级分类频次最多10个序列"},
        {"name": "user_clk_app_1d_freq_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日点击包应用一级分类频次最多5个序列"},
        {"name": "user_clk_app_15d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户15日粘性最高10个点击包序列"},
        {"name": "user_clk_app_7d_sticky_top10_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击包序列"},
        {"name": "user_clk_app_1d_sticky_top5_seq", "is_download": 1, "index": "26", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击包序列"},
        {"name": "user_clk_app_15d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20,
         "pad": "0", "description": "用户15日粘性最高10个点击包应用一级分类序列"},
        {"name": "user_clk_app_7d_sticky_top10_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击包应用一级分类序列"},
        {"name": "user_clk_app_1d_sticky_top5_app_type_seq", "is_download": 1, "index": "27", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击包应用一级分类序列"},
        {"name": "user_imp_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日曝光启动频次最多20个序列"},
        {"name": "user_imp_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日曝光启动频次最多10个序列"},
        {"name": "user_imp_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日曝光启动频次最多5个序列"},
        {"name": "user_imp_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日粘性最高20个曝光启动序列"},
        {"name": "user_imp_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个曝光启动序列"},
        {"name": "user_imp_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个曝光启动序列"},
        {"name": "user_clk_launch_15d_freq_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日点击启动频次最多20个序列"},
        {"name": "user_clk_launch_7d_freq_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日点击启动频次最多10个序列"},
        {"name": "user_clk_launch_1d_freq_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日点击启动频次最多5个序列"},
        {"name": "user_clk_launch_15d_sticky_top20_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户15日粘性最高20个点击启动序列"},
        {"name": "user_clk_launch_7d_sticky_top10_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户7日粘性最高10个点击启动序列"},
        {"name": "user_clk_launch_1d_sticky_top5_seq", "is_download": 1, "index": "3", "length": 20, "pad": "0",
         "description": "用户1日粘性最高5个点击启动序列"},
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
    binning_table_name = "tmp_ad_rank_cvr_activation_sample_data_v2"                    # required
    partitions = "ds_date='{day}',durations='1',model_type='TO5'"                    # required
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
    data_nm = "TO5"                   # required
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

    # ======================= Write-back/Export/Metrics =======================
    # Result table for inference data writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "hstu_model_test"  # Define OSS path for model export                   # required
    oss_offline_root_path = "deep_model/offline"  # OSS path for offline features; used to check online push before export
    # Table for training metrics, yoyo_model only
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr..
    eval_type = "ctcvr"                   # required
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "hstu"       # required


