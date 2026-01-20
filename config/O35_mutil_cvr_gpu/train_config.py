# @File : train_config.py
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    # mkdir -p logs/O35_mutil_cvr_gpu
    # nohup bash test.sh O35_mutil_cvr_gpu 20250902 > logs/O35_mutil_cvr_gpu/nohup_20251022183000.log 2>&1 & tail -f logs/O35_mutil_cvr_gpu/nohup_20251022183000.log
    ###
    device = "GPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "1"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth
    ###
    model_version = "O35_mutil_cvr_gpu"                    # required
    model_modul = "models.mutil_cvr_model_fn.model_fn"           # required
    dataset_modul = "dataset.dataset.input_fn"                    # required
    train_params = {
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        "use_senet": True,
        "mlp_config": {
            "hidden_units": [512, 256, 128],
            "hidden_activations": "relu",
            "output_dim": 1,
            "output_activation": "sigmoid",
            "dropout_rates": 0,
            "batch_norm": False,
            "bn_only_once": False,
            "kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_uniform",
            "use_bias": True
        },
        "mmoe_config": {
            "num_domains": 4,
            "num_experts": 8,
            "exprt_units": [128, 64, 128],
            "hidden_units": [128, 64, 32],
            "hidden_activations": "relu",
            "dropout_rates": 0,
            "batch_norm": False
        }
    }  # Model training params, accessible via params in model_fn
    ### downodps
    data_schema = ['user_id', 'requestid', 'combination_un_id',
                   'is_click',
                   'is_awake',
                   'is_first_appearance',
                   'is_valid_shouden',
                   'is_claim_red_envelopes',
                   'is_valid_linghongbao',
                   'is_page_visit',
                   'is_valid_yemianfangwen',
                   'features']  # required
    label_schema = {
        "is_click": "click_label",
        "is_awake": "awake_label",
        'is_valid_shouden': 'sd_weight',
        'is_first_appearance': "sd_label",
        'is_claim_red_envelopes': "lhb_label",
        'is_valid_linghongbao': 'lhb_weight',
        'is_page_visit': "ymfw_label",
        'is_valid_yemianfangwen': 'ymfw_weight'
    }  # required
    # Align with prediction outputs in the model, add keys to form JSON for inference result storage
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                      + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]    # required
    field_sep = "\t"  # Field separator
    features_sep = "\002"  # Features separator
    compression_type = "GZIP"  # Data compression format
    # Define bucketing and feature selection table
    binning_table_name = "ad_rank_multi_cvr_sample_data"                    # required
    partitions = "ds_date='{day}',durations='1',model_type='O35'"                    # required
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
    data_nm = "O35_mutil_cvr"                   # required
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        },
        "train_batch_size": 10240,
        "train_epoch": 1,
        "batch_size": 1024
    }
    ###
    # Result table for inference data writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "O35_model"  # Define OSS path for model export                   # required
    oss_offline_root_path = "deep_model/offline"  # OSS path for offline features; used to check online push before export
    # Table for training metrics, yoyo_model only
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr..
    eval_type = "cvr"                   # required
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "O35"       # required

