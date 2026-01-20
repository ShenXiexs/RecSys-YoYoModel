# -*- coding: utf-8 -*-
# MiniOneRec config. Extend config + models without changing main/common.
# Bring SFT/RL training flow from the MiniOneRec paper into the existing CLI.
import os
from config.TO5_v2.train_config import TrainConfig as TO5TrainConfig


dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainConfig:
    """
    MiniOneRec core flow: treat SID as tokens and use a lightweight LLM for next-token generation.
      1. Encode items into semantic tokens (this project already provides previously trained semid).
      2. SFT: treat historical interaction sequence as a token sequence for next-token prediction.
      3. RL (GRPO): fine-tune generation policy with ranking feedback.

    The config below writes sft/rl/eval script paths and hyperparameters from MiniOneRec-main into
    train_params['minionerec'], and points the model entry to models.minionerec_main.model_fn,
    which triggers PyTorch training via tf.py_function.
    """

    model_version = "MiniOneRec"
    model_modul = "models.minionerec_main.model_fn"
    dataset_modul = "models.minionerec_main.input_fn"  # Placeholder input_fn is defined in the model file
    device = "GPU"
    gpu_list = "0"  # MiniOneRec training uses GPU-0
    gpu_memory_limit = getattr(TO5TrainConfig, "gpu_memory_limit", 0)
    gpu_memory_growth = getattr(TO5TrainConfig, "gpu_memory_growth", True)
    data_nm = getattr(TO5TrainConfig, "data_nm", "TO5")

    _amazon_root = os.path.join(dirname, "MiniOneRec-main", "data", "Amazon")
    _default_category = "Industrial_and_Scientific"

    train_params = {
        # tf.estimator requires optimize_config; reuse TO5_v2 settings
        "optimize_config": getattr(
            TO5TrainConfig, "train_params", {}
        ).get("optimize_config", {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }),
        "minionerec": {
            "train_stage": "sft",   # Optional: sft / rl
            "eval_stage": "eval",
            "predict_stage": "eval",
            # SFT config follows paper appendix Table 7
            "sft": {
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "train_file": os.path.join(
                    _amazon_root, "train", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "eval_file": os.path.join(
                    _amazon_root, "valid", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "output_dir": os.path.join(dirname, "MiniOneRec-main", "output", "sft"),
                "batch_size": 2048,          # 48G L20 can handle a larger global batch
                "micro_batch_size": 32,      # Single GPU micro-batch 32, accumulate 64 steps to reach batch_size
                "num_epochs": 2,
                "learning_rate": 2e-4,
                "cutoff_len": 512,
                "seed": 42,
                "category": _default_category,
                "sid_index_path": os.path.join(
                    _amazon_root, "index", f"{_default_category}.index.json"
                ),
                "item_meta_path": os.path.join(
                    _amazon_root, "index", f"{_default_category}.item.json"
                ),
                "freeze_LLM": False,
                "wandb_project": "",
                "wandb_run_name": "",
            },
            # GRPO stage: reward combines Top-K correctness + ranking constraints (paper)
            "rl": {
                "model_path": os.path.join(dirname, "MiniOneRec-main", "output", "sft"),
                "train_file": os.path.join(
                    _amazon_root, "train", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "eval_file": os.path.join(
                    _amazon_root, "valid", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "info_file": os.path.join(
                    _amazon_root, "info", f"{_default_category}_5_2016-10-2018-11.txt"
                ),
                "output_dir": os.path.join(dirname, "MiniOneRec-main", "output", "rl"),
                "train_batch_size": 256,
                "eval_batch_size": 256,
                "gradient_accumulation_steps": 4,
                "temperature": 1.0,
                "eval_step": 0.05,
                "num_generations": 32,
                "num_train_epochs": 2,
                "learning_rate": 5e-6,
                "beta": 5e-4,
                "beam_search": True,
                "test_during_training": False,
                "reward_type": "ranking",
                "category": _default_category,
                "sid_index_path": os.path.join(
                    _amazon_root, "index", f"{_default_category}.index.json"
                ),
                "item_meta_path": os.path.join(
                    _amazon_root, "index", f"{_default_category}.item.json"
                ),
            },
            # Eval: use constrained decoding beam search to compute HR@K/NDCG@K
            "eval": {
                "base_model": os.path.join(dirname, "MiniOneRec-main", "output", "sft"),
                "info_file": os.path.join(
                    _amazon_root, "info", f"{_default_category}_5_2016-10-2018-11.txt"
                ),
                "test_data_path": os.path.join(
                    _amazon_root, "test", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "category": _default_category,
                "batch_size": 4,
                "K": 10,
                "num_beams": 10,
                "max_new_tokens": 64,
            },
        },
    }

    # Keep common/utils and other components compatible
    data_schema = getattr(TO5TrainConfig, "data_schema", [])
    label_schema = getattr(TO5TrainConfig, "label_schema", {})
    seq_features_config = getattr(TO5TrainConfig, "seq_features_config", [])
    compression_type = getattr(TO5TrainConfig, "compression_type", "GZIP")
    features_sep = getattr(TO5TrainConfig, "features_sep", "\002")
    field_sep = getattr(TO5TrainConfig, "field_sep", "\t")
    schema_path = os.path.join(dirname, "config", "MiniOneRec", "schema.conf")
    slot_path = os.path.join(dirname, "config", "MiniOneRec", "slot.conf")
    sel_feat_path = os.path.join(dirname, "config", "MiniOneRec", "select_feature.conf")
    boundaries_map_path = os.path.join(dirname, "config", "MiniOneRec", "boundaries_map.json")
    fg_path = os.path.join(dirname, "config", "MiniOneRec", "fg.json")

    es_run_config = getattr(TO5TrainConfig, "es_run_config", {"keep_checkpoint_max": 1})
    inp_fn_config = getattr(TO5TrainConfig, "inp_fn_config", {
        "train_spec": {"max_steps": None},
        "eval_spec": {"steps": None},
        "train_batch_size": 1024,
        "batch_size": 1024,
        "train_epoch": 1,
    })
