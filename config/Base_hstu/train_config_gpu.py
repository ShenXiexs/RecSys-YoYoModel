# @File : train_config.py
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    # nohup bash test.sh Base_sbm_gpu 20251014 > logs/Base_sbm_gpu/nohup_20251017100000.log 2>&1 &
    # tail -f logs/Base_sbm_gpu/nohup_20251017100000.log
    ### GPU training parameter configuration
    device = "CPU"  # Device to use: cpu, gpu, or multi_gpu
    gpu_list = "0"  # Comma-separated list of GPU IDs for multi-GPU mode
    gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    gpu_memory_growth = True  # Allow GPU memory growth
    ###
    model_version = "Base_sbm_gpu"                    # required
    model_modul = "models.ctr_dnn_seq.model_fn"           # required
    dataset_modul = "dataset.dataset.input_fn"                    # required
    train_params = {
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        "dnn_config": {
            "hidden_units": [512, 256, 128, 64],
            "hidden_activations": "relu",
            "output_dim": 1,
            "output_activation": "sigmoid",
            "dropout_rates": 0.0,
            "batch_norm": False,
            "bn_only_once": False,  # Set True for inference speed up
            "use_bias": False
        }
    }  # Model training params, accessible via params in model_fn
    ### downodps
    data_schema = [
        "user_id"
        ,"requestid"
        , "combination_un_id"
        , "dpa_commodity_id"
        , "title_sem_id"
        , "image_sem_id"
        , "is_click"
        , "is_self_pay"
        , "features"
        , "user_seq_launch_15d"
        , "user_seq_commodity_id_clk_30d"
        , "user_seq_commodity_id_pay_30d"
        , "user_seq_shop_id_clk_30d"
        , "user_seq_second_category_clk_30d"
        , "user_seq_second_category_awke_30d"
        , "user_seq_second_category_pay_30d"
        , "user_seq_title_expo_15d"
        , "user_seq_title_clk_30d"
        , "user_seq_image_expo_15d"
        , "user_seq_image_clk_30d"]  # required
    seq_length = {
        #   "user_seq_launch_15d": 50
        # , "user_seq_commodity_id_clk_30d": 50
        # , "user_seq_commodity_id_pay_30d": 50
        # , "user_seq_shop_id_clk_30d": 50
        # , "user_seq_second_category_clk_30d": 50
        # , "user_seq_second_category_awke_30d": 50
        # , "user_seq_second_category_pay_30d": 50
        # , "user_seq_title_expo_15d": 50
        # , "user_seq_title_clk_30d": 50
        # , "user_seq_image_expo_15d": 50
        # , "user_seq_image_clk_30d": 50
    }
    label_schema = {"is_click": "click_label"}                                      # required
    # Align with prediction outputs in the model, add keys to form JSON for inference result storage
    predict_columns = ["is_click", "click_label"]    # required
    features_sep = "\002"  # Features separator
    compression_type = "GZIP"  # Data compression format
    # Define bucketing and feature selection table
    binning_table_name = "tmp_da_dsp_dpa_algo_skuid_feature_rank_encoder_binning_di"                    # required
    partitions = "idx_date='{day}'"                    # required
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
    data_nm = "tmp_da_dsp_dpa_algo_skuid_feature_rank_encoder_binning_di"                   # required
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        },
        "train_batch_size": 4096,
        "train_epoch": 1,
        "batch_size": 1024
    }
    ###
    # Result table for inference data writes
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "Base_sbm_model"  # Define OSS path for model export                   # required
    oss_offline_root_path = "deep_model/offline"  # OSS path for offline features; used to check online push before export
    # Table for training metrics, yoyo_model only
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # Current model type: ctr, cvr, ctcvr..
    eval_type = "cvr"                   # required
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = ""       # required
