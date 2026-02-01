# @File : train_config.py
import copy
import os
import re
import sys

from config.TO5_v2.train_config import TrainConfig as _TO5TrainConfig

dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _extract_train_date(argv):
    for flag in ("--end_time_str", "--time_str"):
        for i, arg in enumerate(argv):
            if arg == flag and i + 1 < len(argv):
                return str(argv[i + 1])[:8]
            if arg.startswith(flag + "="):
                return arg.split("=", 1)[1][:8]
    return None


def _resolve_learning_rate(default_lr=2e-4, cutoff_mmdd="1102", before_lr=3e-4):
    train_date = _extract_train_date(sys.argv)
    if not train_date or len(train_date) < 8:
        return default_lr
    cutoff_date = train_date[:4] + cutoff_mmdd
    return before_lr if train_date <= cutoff_date else default_lr


def _load_feature_list(path):
    features = []
    if not os.path.exists(path):
        return features
    with open(path) as f:
        for line in f:
            name = line.strip("\n").strip()
            if not name or name.startswith("#") or name.startswith("label"):
                continue
            features.append(name)
    return features


def _extract_seq_from_schema(schema_path):
    schema_features = _load_feature_list(schema_path)
    return [name for name in schema_features if name.endswith("_seq")]


def _build_seq_config(seq_names, base_seq_map):
    seq_config = []
    missing = []
    for name in seq_names:
        cfg = base_seq_map.get(name)
        if cfg is None:
            missing.append(name)
            continue
        seq_config.append(copy.deepcopy(cfg))
    if missing:
        raise ValueError(
            "Schema contains seq features missing from TO5_v2 seq config: %s" % ", ".join(missing[:20])
        )
    return seq_config


def _parse_pool_modes(value, default="mean"):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [v.strip().lower() for v in value.split(",") if v.strip()]
    else:
        items = []
    return items or [default]


def _build_seq_token_names(seq_names, pool_modes):
    tokens = []
    for name in seq_names:
        for mode in pool_modes:
            tokens.append("seq::%s::%s" % (name, mode))
    return tokens


def _looks_like_regex(pattern):
    if pattern.startswith("re:"):
        return True
    for token in ("^", "$", ".*", "[", "]", "(", ")", "|", "?"):
        if token in pattern:
            return True
    return False


def _resolve_patterns(patterns, available_names):
    matched = set()
    for raw in patterns:
        if raw in available_names:
            matched.add(raw)
            continue
        pattern = raw[3:] if raw.startswith("re:") else raw
        if _looks_like_regex(raw):
            regex = re.compile(pattern)
            for name in available_names:
                if regex.search(name):
                    matched.add(name)
    return matched


def _validate_semantic_groups(groups, dense_names, seq_token_names, require_dense=True, require_seq=True):
    available = set(dense_names) | set(seq_token_names)
    matched = set()
    for group in groups:
        patterns = list(group.get("features") or [])
        matched |= _resolve_patterns(patterns, available)

    dense_set = set(dense_names)
    seq_set = set(seq_token_names)
    dense_missing = sorted(list(dense_set - matched))
    seq_missing = sorted(list(seq_set - matched))

    dense_ratio = 0.0 if not dense_set else (len(dense_set) - len(dense_missing)) / float(len(dense_set))
    seq_ratio = 0.0 if not seq_set else (len(seq_set) - len(seq_missing)) / float(len(seq_set))

    if require_dense and dense_missing:
        raise ValueError(
            "Semantic groups do not cover all dense features. Missing count=%d sample=%s"
            % (len(dense_missing), ", ".join(dense_missing[:20]))
        )
    if require_seq and seq_missing:
        raise ValueError(
            "Semantic groups do not cover all seq tokens. Missing count=%d sample=%s"
            % (len(seq_missing), ", ".join(seq_missing[:20]))
        )

    return {
        "dense_total": len(dense_set),
        "dense_missing": len(dense_missing),
        "dense_coverage_ratio": dense_ratio,
        "seq_total": len(seq_set),
        "seq_missing": len(seq_missing),
        "seq_coverage_ratio": seq_ratio,
        "dense_missing_sample": dense_missing[:20],
        "seq_missing_sample": seq_missing[:20],
    }


class TrainConfig(_TO5TrainConfig):
    """
    Shen0128: schema-driven seq features + strict seq coverage.

    Design guarantees:
    1) seq features must come from config/TO5_v2/schema.conf (90 *_seq features).
    2) seq must be covered by semantic_groups, otherwise raise at import time.
    """

    # ===== Basic model entry =====
    model_version = "RankMixer_Shen0128"
    model_modul = "models.rankmixer_shen0128.model_fn"
    dataset_modul = "dataset.dataset_seq.input_fn"

    # ===== Device =====
    device = "GPU"
    gpu_list = "0"
    gpu_memory_limit = 0
    gpu_memory_growth = True

    # ===== Paths (must be overridden when model_version changes) =====
    schema_path = f"{dirname}/config/{model_version}/schema.conf"
    slot_path = f"{dirname}/config/{model_version}/slot.conf"
    sel_feat_path = f"{dirname}/config/{model_version}/select_feature.conf"
    boundaries_map_path = f"{dirname}/config/{model_version}/boundaries_map.json"
    fg_path = f"{dirname}/config/{model_version}/fg.json"
    feature_config_path = f"{dirname}/config/{model_version}/feature_config.json"
    body_json_name = f"{dirname}/config/{model_version}/body.json"
    # When select_feature.conf is used, downodps already filters features. Use contiguous indices for dataset gather.
    select_index_mode = "contiguous"

    # ===== Schema-driven seq features =====
    _base_seq_map = {cfg["name"]: cfg for cfg in _TO5TrainConfig.seq_features_config}
    _schema_seq_names = _extract_seq_from_schema(schema_path)
    if len(_schema_seq_names) != 90:
        raise ValueError(
            "Expected 90 seq features from schema.conf, got %d" % len(_schema_seq_names)
        )
    seq_features_config = _build_seq_config(_schema_seq_names, _base_seq_map)

    # ===== Dense feature names used by the model =====
    _dense_feature_names = _load_feature_list(sel_feat_path) or _load_feature_list(slot_path)

    # ===== Train params passed into model_fn =====
    train_params = {
        "optimize_config": {
            "optimizer": "sgd",
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
        "rankmixer": {
            # ===== Model scale =====
            "d_model": 368,
            "num_layers": 8,
            "num_tokens": 23,
            "num_heads": 23,
            "ffn_mult": 8,

            # ===== Tokenization =====
            "tokenization_strategy": "semantic",
            "tokenization_version": "v3",
            "semantic_target_tokens": 23,
            "semantic_groups": [
                {"name": "customer_adplan_attr", "features": [
                    "combination_un_id",
                    "ad_idea_id",
                    "ad_unit_id",
                    "ad_plan_id",
                    "promotion_type",
                    "target_type",
                    "model_type",
                ]},
                {"name": "customer_ad_attr", "features": [
                    "template_id",
                    "template_type",
                    "title",
                    "content",
                ]},
                {"name": "customer_task_attr", "features": [
                    "package_name",
                    "app_first_type",
                    "app_second_type",
                    "log10_app_size",
                    "product_name",
                    "first_industry_id",
                    "industry",
                    "rta_type",
                    "crowd_type",
                    "rta_product_code",
                    "white_crowd_code",
                    "dispatch_center_id",
                    "package_channel_type",
                ]},
                {"name": "context_traffic_attr", "features": [
                    "adslot_id",
                    "ssp_adslot_id",
                    "channel_id",
                    "app_pkg_src",
                    "keywords",
                ]},
                {"name": "context_device_attr", "features": [
                    "network",
                    "ip_region",
                    "ip_city",
                    "device_brand",
                    "device_os",
                    "device_model",
                    "device_screen_height",
                    "device_screen_width",
                    "device_carrier",
                    "device_os_version",
                ]},
                {"name": "context_time_attr", "features": [
                    "day_h",
                ]},
                {"name": "context_src_attr", "features": [
                    "app_src_first_type",
                    "app_src_second_type",
                ]},
                {"name": "match_flags", "features": [
                    "is_new_item",
                    "is_match_awake",
                    "is_match_first_awke",
                    "is_match_app_type_awake",
                    "is_match_app_type_first_awke",
                    "is_match_clk_app",
                    "is_match_clk_launch",
                ]},
                {"name": "user_stat_global", "features": [
                    "re:^user__imp_cnt_",
                    "re:^user__clk_cnt_",
                ]},
                {"name": "user_stat_day", "features": [
                    "re:^user__kv_day_h_",
                ]},
                {"name": "user_stat_ad_template", "features": [
                    "re:^user__kv_(ad_idea_id|ad_plan_id|adslot_id|template_id|template_type)_",
                ]},
                {"name": "user_stat_app_industry", "features": [
                    "re:^user__kv_(app_first_type|package_name|product_name|first_industry_id|industry_id)_",
                ]},
                {"name": "stat_doc_one", "features": [
                    "re:^doc__key_one__",
                ]},
                {"name": "stat_doc_two", "features": [
                    "re:^doc__key_two__",
                ]},
                {"name": "stat_doc_three", "features": [
                    "re:^doc__key_three__",
                ]},
                {"name": "stat_doc_four", "features": [
                    "re:^doc__key_four__",
                ]},
                {"name": "stat_doc_five", "features": [
                    "re:^doc__key_five__",
                ]},
                {"name": "stat_doc_six", "features": [
                    "re:^doc__key_six__",
                ]},
                {"name": "stat_doc_seven", "features": [
                    "re:^doc__key_seven__",
                ]},
                {"name": "seq_awake", "features": [
                    "re:^seq::user_awake_",
                    "re:^seq::user_first_awke_",
                ]},
                {"name": "seq_app_imp", "features": [
                    "re:^seq::user_imp_app_",
                ]},
                {"name": "seq_app_clk", "features": [
                    "re:^seq::user_clk_app_",
                ]},
                {"name": "seq_launch", "features": [
                    "re:^seq::user_(imp|clk)_launch_",
                ]},
            ],
            "token_projection": "linear",
            "include_seq_in_tokenization": True,

            # ===== Strict coverage requirements =====
            "require_seq_coverage": True,
            "require_dense_coverage": True,
            "require_seq_in_tokenization": True,
            "coverage_log_limit": 20,

            # ===== Token Mixing =====
            "token_mixing_type": "paper_strict",
            "token_mixing_dropout": 0.0,

            # ===== Per-token FFN =====
            "per_token_ffn": True,
            "ffn_activation": "gelu",
            "ffn_dropout": 0.0,

            # ===== LayerNorm =====
            "ln_style": "pre",
            "input_ln": False,
            "use_final_ln": True,

            # ===== Pooling / Head =====
            "output_pooling": "mean",
            "add_cls_token": False,
            "head_dropout": 0.0,

            # ===== MoE =====
            "use_moe": False,
            "moe_num_experts": 8,
            "moe_sparsity_ratio": 0.125,
            "moe_routing_type": "relu_dtsi",
            "moe_l1_lambda": 0.01,
            "moe_use_dtsi": True,

            # ===== Other base config =====
            "use_other_features": True,
            "seq_pool": "mean",
            "embedding_size": 9,
            "input_dropout": 0.0,

            # ===== Loss =====
            "use_ctcvr_loss": True,
            "use_conditional_cvr": False,
        },
        "ps_num": 1,
        "restrict": True,
        "l2_reg": 1e-6,
    }

    # ===== Static coverage validation at import time =====
    _seq_pool_modes = _parse_pool_modes(train_params["rankmixer"].get("seq_pool", "mean"))
    _seq_token_names = _build_seq_token_names(_schema_seq_names, _seq_pool_modes)
    semantic_coverage_info = _validate_semantic_groups(
        train_params["rankmixer"]["semantic_groups"],
        _dense_feature_names,
        _seq_token_names,
        require_dense=train_params["rankmixer"].get("require_dense_coverage", True),
        require_seq=train_params["rankmixer"].get("require_seq_coverage", True),
    )

    # ===== Data schema =====
    data_schema = list(_TO5TrainConfig.data_schema)
    label_schema = dict(_TO5TrainConfig.label_schema)

    # Keep predict_columns aligned with label_schema
    predict_columns = [k for k, v in label_schema.items() if v.endswith("_label")] + [
        v.replace("_label", "_pred") for _, v in label_schema.items() if v.endswith("_label")
    ]

    # ===== IO / ODPS / Estimator config =====
    field_sep = getattr(_TO5TrainConfig, "field_sep", "\003")
    features_sep = getattr(_TO5TrainConfig, "features_sep", "\002")
    compression_type = getattr(_TO5TrainConfig, "compression_type", "GZIP")

    binning_table_name = _TO5TrainConfig.binning_table_name
    partitions = _TO5TrainConfig.partitions
    downodps_datas = list(getattr(_TO5TrainConfig, "downodps_datas", []))

    es_run_config = copy.deepcopy(getattr(_TO5TrainConfig, "es_run_config", {}))

    data_nm = getattr(_TO5TrainConfig, "data_nm", "TO5")
    inp_fn_config = copy.deepcopy(getattr(_TO5TrainConfig, "inp_fn_config", {}))

    # ===== Export / metrics paths =====
    infer_table_name = getattr(
        _TO5TrainConfig, "infer_table_name", "adx_dmp.ads_algorithm_yoyo_model_offline_shallow_predict"
    )
    oss_bucket_name = getattr(_TO5TrainConfig, "oss_bucket_name", "adx-oss")
    upload_oss_path = "rankmixer_shen0128_model_test"
    oss_offline_root_path = getattr(_TO5TrainConfig, "oss_offline_root_path", "deep_model/offline")
    metric_table = getattr(_TO5TrainConfig, "metric_table", "adx_dmp.ads_algorithm_yoyo_model_eval_metric_table_dm")
    eval_type = getattr(_TO5TrainConfig, "eval_type", "ctcvr")
    oss_offline_model_ver = "rankmixer_shen0128"


class TrainConfig_1B(TrainConfig):
    model_version = "RankMixer_Shen0128"
    train_params = copy.deepcopy(TrainConfig.train_params)
    train_params["rankmixer"].update({
        "use_moe": True,
        "moe_num_experts": 8,
        "moe_sparsity_ratio": 0.125,
        "moe_routing_type": "relu_dtsi",
        "moe_l1_lambda": 0.01,
    })
