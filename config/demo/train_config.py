# @File : train_config.py
import os

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainConfig:
    ### GPU训练参数配置
    # device = "CPU"  # Device to use: cpu, gpu, or multi_gpu
    # gpu_list = "0"  # Comma-separated list of GPU IDs for multi-GPU mode
    # gpu_memory_limit = 0  # GPU memory limit in MB (0 for no limit)
    # gpu_memory_growth = True  # Allow GPU memory growth
    ###
    model_version = ""                    #必填
    model_modul = "models.mutil_task_model_fn.model_fn"           #必填
    dataset_modul = "dataset.dataset.input_fn"                    #必填
    train_params = {
        "optimize_config": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8
        },
    }  # 模型训练参数，可在model_fn函数中通过params获取到
    ### downodps
    data_schema = ['user_id', 'requestid', 'combination_un_id', 'ad_unit_id', 'features', 'is_click'] #必填
    label_schema = {"is_click": "click_label"}                                                        #必填
    # 跟模型中prediction中out的输出对其，给模型输出补key，组成json格式，用于推理模型的效果存储
    predict_columns = [k for k,v in label_schema.items() if v.endswith("_label")] \
                    + [v.replace("_label", "_pred") for k,v in label_schema.items() if v.endswith("_label")]  # 必填
    field_sep = "\003"  # 字段的分隔符
    features_sep = "\002"  # features特征的分隔符
    compression_type = "GZIP"  # 数据压缩格式
    # 定义分桶且特征选择表
    binning_table_name = "ocpx_mutil_label_cvr_order_sample_train_data"                    #必填
    partitions = "idx_date='{day}',feature_version='',durations='1'"                    #必填
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
    data_nm = ""                   #必填
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
    upload_oss_path = "_model"  #定义模型导出OSS路径                   #必填
    oss_offline_root_path = "deep_model/offline"  # 离线特征推送OSS路径，判断特征是否推线上，再推模型到OSS供线上推理使用
    # 模型训练指标写入表, yoyo_model独有
    metric_table = 'adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm'
    # 当前模型类型，ctr,cvr,ctcvr..
    eval_type = "cvr"                   #必填
    # deep_model/offline/{}/20250924/_FEATURE_SUCCESS is exists
    oss_offline_model_ver = "O29"       #必填
