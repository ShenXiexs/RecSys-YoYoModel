# -*- coding: utf-8 -*-
# MiniOneRec config. 通过 config + models 的扩展，在不改动 main/common 的情况下
# 将 MiniOneRec 论文中的 SFT/RL 训练流程纳入现有 CLI。
import os
from config.TO5_v2.train_config import TrainConfig as TO5TrainConfig


dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainConfig:
    """
    MiniOneRec 的核心流程是「SID 作为 token，使用轻量 LLM 做下一 token 生成」。
      1. 先把物品编码成语义 token（本工程数据已提供我之前训练的 semid）。
      2. SFT：把历史交互序列视作 token 序列，做 next-token prediction。
      3. RL (GRPO)：用排名反馈微调生成策略。

    下面的配置直接把 MiniOneRec-main 中的 sft/rl/eval 脚本路径、超参写入
    train_params['minionerec']，模型入口指向 models.minionerec_main.model_fn，
    由该 model_fn 通过 tf.py_function 触发 PyTorch 训练。
    """

    model_version = "MiniOneRec"
    model_modul = "models.minionerec_main.model_fn"
    dataset_modul = "models.minionerec_main.input_fn"  # 占位 input_fn 定义在 model 文件中
    device = "GPU"
    gpu_list = "0"  # MiniOneRec 训练固定使用 GPU-0
    gpu_memory_limit = getattr(TO5TrainConfig, "gpu_memory_limit", 0)
    gpu_memory_growth = getattr(TO5TrainConfig, "gpu_memory_growth", True)
    data_nm = getattr(TO5TrainConfig, "data_nm", "TO5")

    _amazon_root = os.path.join(dirname, "MiniOneRec-main", "data", "Amazon")
    _default_category = "Industrial_and_Scientific"

    train_params = {
        # tf.estimator 需要 optimize_config，直接复用 TO5_v2 的设置
        "optimize_config": getattr(
            TO5TrainConfig, "train_params", {}
        ).get("optimize_config", {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }),
        "minionerec": {
            "train_stage": "sft",   # 可选：sft / rl
            "eval_stage": "eval",
            "predict_stage": "eval",
            # 参照论文附录 Table 7 的 SFT 配置
            "sft": {
                "base_model": "Qwen/Qwen2-0.5B-Instruct",
                "train_file": os.path.join(
                    _amazon_root, "train", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "eval_file": os.path.join(
                    _amazon_root, "valid", f"{_default_category}_5_2016-10-2018-11.csv"
                ),
                "output_dir": os.path.join(dirname, "MiniOneRec-main", "output", "sft"),
                "batch_size": 2048,          # 48G L20 足以容纳更大的全局 batch
                "micro_batch_size": 32,      # 单卡 micro batch 32，梯度累积 64 次得到 batch_size
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
            # GRPO 阶段：论文中的 reward 结合 Top-K correctness + 排名约束
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
            # 评估：使用  constrained decoding 的 beam search 计算 HR@K/NDCG@K
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

    # 使 common/utils 等组件保持兼容
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
