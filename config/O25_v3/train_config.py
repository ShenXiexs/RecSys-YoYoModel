# -*- coding: utf-8 -*-
# @Time : 2025/8/25 18:20
# @Author : huangmian
# @File : train_config.py
import os
import pandas as pd

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    model_version = "O25_v3"
    model_modul = "models.cvr_model_fn.model_fn"
    dataset_modul = "dataset.dataset_cvr.input_fn"
    train_params = {}
    ### downodps
    data_schema = ['user_id', 'requestid', 'combination_un_id', 'adslot_id_type', 'is_click', 'is_conversion', 'features']
    label_schema = {"is_click": "ctr_label",
                    "is_conversion": "ctcvr_label"}
    features_sep = "\002"
    compression_type = "GZIP"
    # 定义分桶且特征选择表
    binning_table_name = "da_algorithm_search_ad_feature_rank_sample_ctr_cvr_encoded_binning_di"
    downodps_datas = pd.date_range("20250805", "20500101").map(lambda x: x.strftime("%Y%m%d")).tolist()
    partitions = "idx_date='{day}',feature_version='O25_v3',durations='1'"
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
    data_nm = "O25_v3"
    inp_fn_config = {
        "train_spec": {
            "max_steps": None
        },
        "eval_spec": {
            "start_delay_secs": 1e20,
            "steps": None
        }
    }
    ###
    # infer数据写入的结果表
    infer_table_name = 'adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict'
    ### upload
    oss_bucket_name = "adx-oss"
    upload_oss_path = "O25_model_v3"  #定义模型导出OSS路径
    oss_offline_root_path = "deep_model/offline"  # 离线特征推送OSS路径，判断特征是否推线上，再推模型到OSS供线上推理使用
    # 模型训练指标写入表,针对ocpc模型
    metric_table = 'adx_dmp.ads_yoyo_self_built_algorithm_model_eval_metric_table_dm'
