# @File : train_config.py
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    ### GPU training parameter configuration
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "1"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth
    ###
    model_version = "TO5_v2"                    # required
    model_modul = "models.cvr_model_fn_combo.model_fn"           # required
    dataset_modul = "dataset.dataset_seq.input_fn"                    # required
    train_params = {
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
    }  # Model training params, accessible via params in model_fn
    ### downodps
    data_schema = ["user_id", "requestid", "combination_un_id", "is_click", "is_conversion", "features", "app_pkg_src", "app_pkg", "app_first_type", "seq_features" ] # required
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}                                                        # required
    # seq_feature configuration
    seq_features_config = [
        # invisible for satey reasons
    ]
    # Align with prediction outputs in the model, add keys to form JSON for inference result storage
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                    + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]  # required
    field_sep = "\003"  # Field separator
    features_sep = "\002"  # Features separator
    compression_type = "GZIP"  # Data compression format
    # Define bucketing and feature selection table
    binning_table_name = "tmp_ad_rank_cvr_activation_sample_data_v2"                    # required
    partitions = "ds_date='{day}',durations='1',model_type='TO5'"                    # required
    downodps_datas = ['20250901']
    ### config path
    schema_path = f"{dirname}/config/{model_version}/schema.conf"  # Required file
    slot_path = f"{dirname}/config/{model_version}/slot.conf"  # Required file
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"  # Required file
    fg_path = f"{dirname}/config/{model_version}/fg.json"  # Required file
    feature_config_path = f"{dirname}/config/{model_version}/feature_config.json"
    body_json_name = f"{dirname}/config/{model_version}/body.json"
    ### es config
    es_run_config = {
        "keep_checkpoint_max": 1,
        "save_checkpoints_steps": 100000,
        "log_step_count_steps": 5000,
        "save_summary_steps": 10000
    }
    ### dataset input_fn config
    data_nm = "TO5"                   # required
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
        "batch_size": 5120
    }
    ###
    # Result table for inference data writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "TO5_model_test"  # Define OSS path for model export                   # required
    oss_offline_root_path = "deep_model/offline"  # OSS path for offline features; used to check online push before export
    # Table for training metrics, yoyo_model only
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr..
    eval_type = "ctcvr"                   # required
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "TO5"       # required
