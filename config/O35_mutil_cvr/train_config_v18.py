# @File : train_config.py
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    # mkdir -p logs/O35_mutil_cvr_v18
    # touch logs/O35_mutil_cvr_v18/nohup_20251107170000.log
    # nohup bash test.sh O35_mutil_cvr_v18 20251001 > logs/O35_mutil_cvr_v18/nohup_20251107170000.log 2>&1 & tail -f logs/O35_mutil_cvr_v18/nohup_20251107170000.log
    # bash export_after.sh O35_mutil_cvr_v18 20251029
    model_version = "O35_mutil_cvr_v18"                    #必填
    model_modul = "models.mutil_cvr_model_fn_v3_2.model_fn"           #必填
    dataset_modul = "dataset.dataset.input_fn"                    #必填
    train_params = {
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
        "use_senet": False,
        "mlp_config": {
            "hidden_units": [256, 128, 64, 32],
            "hidden_activations": "relu",
            "output_dim": 1,
            "output_activation": None,
            "dropout_rates": 0.2,
            "batch_norm": True,
            "bn_only_once": False,
            "kernel_initializer": "glorot_uniform",
            "output_kernel_initializer": "glorot_uniform",
            "bias_initializer": "glorot_uniform",
            "use_bias": True
        },
        "awake_fusion_type": "mlp",
        "awake_fusion_hidden_units": [32],
        "multi_task_fusion_type": "mlp",
        "multi_task_fusion_hidden_units": [32],
        "multi_task_config": {
            "type": "mmoe",
            "num_domains": 3,
            "num_experts": 6,
            "exprt_units": [128, 64, 128],
            "hidden_units": [128, 64, 32],
            "hidden_activations": "relu",
            "dropout_rates": 0.2,
            "batch_norm": False
        },
        "tower_config": {
            "hidden_units": [128, 64, 32],
            "hidden_activations": "relu",
            "dropout_rates": 0,
            "batch_norm": False
        },
        "adslot_add_weight_config": {
            "type": "emb",
            "data_path": f"{dirname}/config/{model_version}/adslot_id_count.csv",
            "key": "adslot_id",
            "values": ["sd_cnt", "lhb_cnt", "ymfw_cnt"],
            "sep":"\t",
            "default_value": 0
        },     #加权到adslot_id的emb表征上/加权到loss上
    }  # 模型训练参数，可在model_fn函数中通过params获取到
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
                   'features']  # 必填
    label_schema = {
        "is_click": "ctr_label",
        "is_awake": "awake_label",
        'is_valid_shouden': 'sd_weight',
        'is_first_appearance': "sd_label",
        'is_claim_red_envelopes': "lhb_label",
        'is_valid_linghongbao': 'lhb_weight',
        'is_page_visit': "ymfw_label",
        'is_valid_yemianfangwen': 'ymfw_weight'
    }  # 必填
    # 跟模型中prediction中out的输出对其，给模型输出补key，组成json格式，用于推理模型的效果存储
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                      + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")] \
                      + ['sd_weight', 'lhb_weight', 'ymfw_weight'] #必填
    field_sep = "\003"  # 字段的分隔符
    features_sep = "\002"  # features特征的分隔符
    compression_type = "GZIP"  # 数据压缩格式
    # 定义分桶且特征选择表
    binning_table_name = "ad_rank_multi_cvr_sample_data"                    #必填
    partitions = "ds_date='{day}',durations='1',model_type='O35'"                    #必填
    downodps_datas = ['20250901']
    ### config path
    schema_path = f"{dirname}/config/{model_version}/schema.conf"  # 该文件必须要
    slot_path = f"{dirname}/config/{model_version}/slot.conf"  # 该文件必须要
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"  # 该文件必须要
    fg_path = f"{dirname}/config/{model_version}/fg.json"  # 该文件必须要
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
    data_nm = "O35_mutil_cvr"                   #必填
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
    upload_oss_path = "O35_model"  #定义模型导出OSS路径                   #必填
    oss_offline_root_path = "deep_model/offline"  # 离线特征推送OSS路径，判断特征是否推线上，再推模型到OSS供线上推理使用
    # 模型训练指标写入表, yoyo_model独有
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # 当前模型类型，ctr,cvr,ctcvr..
    eval_type = "cvr"                   #必填
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "O35"       #必填
