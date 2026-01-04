# -*- coding: utf-8 -*-
# MiniOneRec 内置实现（不依赖 MiniOneRec-main 目录）。在 Estimator 的 train/eval 流中，
# 用 tf.py_function 触发一个简化版的 SFT/GRPO 训练，模型主体使用 HuggingFace Transformers。
import os
import sys
import ast
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE_DONE = {"train": False, "eval": False, "predict": False}


def _resolve_path(path):
    if not isinstance(path, str):
        return path
    path = path.format(project_root=PROJECT_ROOT)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return path


def _prepare_kwargs(stage_cfg):
    cfg = {}
    for key, value in (stage_cfg or {}).items():
        cfg[key] = _resolve_path(value)
    return cfg


class MiniOneRecRLDataset(Dataset):
    """简化版 RL 数据集：读取 CSV，使用 history_item_sid -> item_sid 作为训练对。"""
    def __init__(self, csv_path, tokenizer, max_len=512, sample=-1, seed=42):
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_path)
        if sample and sample > 0:
            df = df.sample(sample, random_state=seed)
        self.data = df
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def _parse_history(self, raw):
        # history_item_sid 列预期形如 "['sid1','sid2',...]" 或列表字符串
        try:
            lst = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if isinstance(lst, list):
                return lst
        except Exception:
            pass
        return []

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        history = self._parse_history(row.get("history_item_sid", ""))
        target = str(row.get("item_sid", ""))
        prompt = f"User history: {' '.join(history)}. Recommend next item SID:"
        target_text = target + self.tokenizer.eos_token

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        input_ids = prompt_ids + target_ids
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
        # mask prompt 部分，只有 target 参与 loss
        labels = [-100] * len(prompt_ids) + target_ids
        labels = labels[-len(input_ids):]
        attn_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate_fn(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn_mask, labels = [], [], []
    for sample in batch:
        L = len(sample["input_ids"])
        pad_len = max_len - L
        input_ids.append(
            torch.cat([sample["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attn_mask.append(
            torch.cat([sample["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        # label 用 -100 作为 pad，避免参与 loss
        labels.append(
            torch.cat([sample["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn_mask),
        "labels": torch.stack(labels),
    }


def _train_minionerec(stage_cfg, stage_name):
    """简化版训练：SFT & RL 都用监督交叉熵，RL 仅增加随机采样增强。"""
    cfg = _prepare_kwargs(stage_cfg)
    base_model = cfg.get("base_model") or cfg.get("model_path")
    train_file = cfg.get("train_file")
    eval_file = cfg.get("eval_file", train_file)
    output_dir = cfg.get("output_dir", os.path.join(PROJECT_ROOT, "MiniOneRec_output"))
    batch_size = int(cfg.get("train_batch_size", cfg.get("batch_size", 64)))
    eval_batch_size = int(cfg.get("eval_batch_size", cfg.get("batch_size", 64)))
    epochs = int(cfg.get("num_train_epochs", cfg.get("num_epochs", 1)))
    lr = float(cfg.get("learning_rate", 5e-5))
    max_len = int(cfg.get("cutoff_len", 512))
    seed = int(cfg.get("seed", 42))

    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model.to(device)

    train_ds = MiniOneRecRLDataset(train_file, tokenizer, max_len=max_len, sample=-1, seed=seed)
    eval_ds = MiniOneRecRLDataset(eval_file, tokenizer, max_len=max_len, sample= min(len(train_ds), 1024), seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id)
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=eval_batch_size, shuffle=False,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id)
    )

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 简化版 eval
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_losses.append(outputs.loss.item())
        avg_eval_loss = sum(eval_losses) / max(1, len(eval_losses))
        model.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def _py_launcher(stage_key, params):
    stage_cfg = params.get("minionerec", {})
    if not stage_cfg:
        raise ValueError("train_params['minionerec'] is missing in TrainConfig.")

    gpu_ids = params.get("gpu_ids") or ["0"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    if stage_key == "train":
        selected_stage = stage_cfg.get("train_stage", "sft")
    elif stage_key == "eval":
        selected_stage = stage_cfg.get("eval_stage", "eval")
    else:
        selected_stage = stage_cfg.get("predict_stage", stage_cfg.get("eval_stage", "eval"))

    payload = stage_cfg.get(selected_stage, {})

    def _runner():
        if _STAGE_DONE[stage_key]:
            return np.int64(0)
        # 简化：SFT/RL 都调用同一训练流程（RL 可进一步增强，但这里保持最小可跑）
        _train_minionerec(payload, selected_stage)
        _STAGE_DONE[stage_key] = True
        return np.int64(0)

    return tf.py_function(_runner, [], Tout=tf.int64)


def _dummy_predictions(features):
    if features:
        any_tensor = next(iter(features.values()))
        batch = tf.shape(any_tensor)[0]
    else:
        batch = 1
    return {
        "minionerec_dummy": tf.zeros([batch, 1], dtype=tf.float32)
    }


def input_fn(filenames,
             model_dir,
             task_number=1,
             task_idx=0,
             shuffle=True,
             epochs=1,
             batch_size=1):
    """提供给 Estimator 的占位 input_fn，避免加载大规模 TF 数据。"""
    dummy = tf.zeros([batch_size, 1], dtype=tf.float32)
    return {"dummy_feature": dummy}


def model_fn(features, labels, mode, params):
    """Estimator-compatible wrapper around内置的 MiniOneRec 训练（简化版）。"""
    loss = tf.constant(0.0, dtype=tf.float32)
    predictions = _dummy_predictions(features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        trigger = _py_launcher("train", params)
        with tf.control_dependencies([trigger]):
            loss_train = tf.identity(loss)
            train_op = tf.no_op()
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_train,
            train_op=train_op
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        trigger = _py_launcher("eval", params)
        with tf.control_dependencies([trigger]):
            loss_eval = tf.identity(loss)
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss_eval,
            eval_metric_ops={}
        )

    trigger = _py_launcher("predict", params)
    with tf.control_dependencies([trigger]):
        preds = {k: tf.identity(v) for k, v in predictions.items()}
    export_outputs = {
        "serving_default": tf.estimator.export.PredictOutput(preds)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=preds,
        export_outputs=export_outputs
    )
