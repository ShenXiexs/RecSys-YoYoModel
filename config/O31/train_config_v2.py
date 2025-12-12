# -*- coding: utf-8 -*-
# @Time : 2025/8/25 18:20
# @Author : huangmian
# @File : train_config.py
import os
import pandas as pd

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    # mkdir -p logs/O31_v2
    # touch logs/O31_v2/nohup_20250801093000.log
    # nohup bash run.sh O31_v2 20250801 > logs/O31_v2/nohup_20250801093000.log 2>&1 & tail -f logs/O31_v2/nohup_20250801093000.log
    model_version = "O31_v2"
    model_modul = "models.mutil_task_model_fn.model_fn"
    dataset_modul = "dataset.dataset.input_fn"
    train_params = {
        "num_domains": 3,
        "num_experts": 3,
        "exprt_units": [128, 64, 128],
        "hidden_units": [128, 64, 32],
        "hidden_activations": "relu",
        "dropout_rates": 0,
        "batch_norm": False
    }
    ### downodps
    data_schema = ['user_id', 'requestid', 'combination_un_id', 'ad_unit_id', 'features', 'is_awaken', 'is_middle', 'is_order']
    label_schema = {"is_awaken": "awaken_label",
                    "is_middle": "middle_label",
                    "is_order": "order_label"}
    predict_columns = list(label_schema.keys()) + list(label_schema.values())
    features_sep = "\002"
    compression_type = "GZIP"
    # 定义分桶且特征选择表
    binning_table_name = "ocpx_mutil_label_cvr_order_sample_train_data"
    downodps_datas = pd.date_range("20250901", "20250901").map(lambda x: x.strftime("%Y%m%d")).tolist()
    partitions = "idx_date='{day}',feature_version='O31',durations='1'"
    ### config path
    schema_path = f"{dirname}/config/{model_version}/schema.conf"
    slot_path = f"{dirname}/config/{model_version}/slot.conf"
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"
    fg_path = f"{dirname}/config/{model_version}/fg.json"
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
    data_nm = "O31"
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
        "batch_size": 1024
    }
    ###
    # infer数据写入的结果表
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "O31_model"  # 定义模型导出OSS路径
    oss_offline_root_path = "deep_model/offline"  # 离线特征推送OSS路径，判断特征是否推线上，再推模型到OSS供线上推理使用
    # 模型训练指标写入表, yoyo_model独有
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # 当前模型类型，ctr,cvr,ctcvr..
    eval_type = "cvr"
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "O29"

