# YoYo Recommendation Model

YoYo Recommendation Model is a TensorFlow-based ranking and conversion prediction stack that powers multi-task ad recommendation research. The repo contains production-style pipelines for downloading features from MaxCompute/ODPS, training several neural architectures (e.g., TIGER, RankMixer, DeepFM variants, sequence-aware HSTU models), evaluating performance, exporting SavedModels, and syncing artifacts to OSS.

The code base is organized so a single `MODEL_TASK` (for example `O35_mutil_cvr_v10`) encapsulates everything it needs: dataset schema, feature configuration, model hyper-parameters, and cron-style shell wrappers. The same runner can execute training, evaluation, inference, and exporting on GPU or CPU with identical business logic.

## Repository Layout
- `bin/`: automation scripts used in production (train/eval/export/infer, TF-Serving smoke tests, ODPS downloads).
- `common/`: shared utilities for data ingestion, feature parsing, TF dataset helpers, ODPS/OSS bridges, metric logging, etc.
- `config/`: per-model configuration packages containing `train_config.py`, HTTP body templates, cron samples, and DataWorks scripts.
- `dataset/`: dataset readers and preprocessing utilities.
- `layers/`, `models/`: model architecture definitions (MMoE, RankMixer blocks, TIGER encoder, HSTU sequence modules, etc.).
- `scripts/` and `test/`: demo runners plus TF-Serving regression tests.

## Key Capabilities
- **Multi-task modeling**: CTR, CVR, CTCVR, and auxiliary heads running on a unified parameter server via Estimator.
- **Feature rich input pipeline**: supports dense/sparse features, scripted slot selection, and sequential histories with custom padding logic.
- **Cluster-friendly execution**: `TF_CONFIG` driven distribution, mirrored strategy on GPU, and job orchestration through `bin/run.sh`.
- **Artifact management**: hooks that write predictions/metrics back to ODPS tables and keep checkpoints/exported models in OSS.
- **Serving preview**: ready-to-use TF-Serving scripts (`test/run_tfserving_test_http.sh`) and sample request bodies for sanity checks.

## Getting Started
1. **Environment**
   - Python 3.8+ with TensorFlow 2.x, `tensorflow-recommenders-addons`, `pandas`, `redis`, `oss2`, `pyodps`, etc.
   - (Optional) GPU runtime with CUDA/cuDNN if you want to enable multi-GPU training.
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt  # update the file to match your environment
   ```
3. **Select a task**
   - Copy one of the `config/<task>/train_config*.py` templates and adjust schema paths, feature settings, optimization params, and ODPS/OSS destinations.
   - Set the task name before running scripts: `export MODEL_TASK=O35_mutil_cvr_v10` (for example).
4. **Download data + train**
   ```bash
   cd bin
   bash run.sh ${MODEL_TASK} 20250101 20250102  # <start_date> <end_date> [batch_days]
   ```
   `run.sh` orchestrates ODPS downloads, trains via `main.py`, optionally evaluates, exports, and triggers inference jobs.
5. **Evaluate/Export/Infer separately**
   - `bash eval.sh ${MODEL_TASK} 20250102`
   - `bash export.sh ${MODEL_TASK} 20250102`
   - `bash infer.sh ${MODEL_TASK} 20250102`
6. **Serving smoke test**
   - Point `MODEL_ROOT` and `MODEL_TASK`, then run `test/run_tfserving_test_http.sh` to start a local TF-Serving container and issue sample HTTP requests.

## Secrets & Environment Variables
All credentials and server addresses have been placeholdered so nothing sensitive ships with the repo. Populate them via environment variables before executing any job.

| Purpose | Environment variable | Description |
| --- | --- | --- |
| OSS uploads | `UPLOAD_OSS_ACCESS_KEY_ID`, `UPLOAD_OSS_ACCESS_KEY_SECRET`, `UPLOAD_OSS_ENDPOINT`, `UPLOAD_OSS_BUCKET` | Credentials and endpoint used by `common/upload.py` when syncing SavedModels. |
| ODPS connectivity | `ODPS_ACCESS_KEY_ID`, `ODPS_ACCESS_KEY_SECRET`, `ODPS_ENDPOINT`, `ODPS_TUNNEL_ENDPOINT`, `ODPS_DEFAULT_PROJECT` | Required for any ODPS read/write (data download, metrics upload, SQL execution). |
| Generic OSS access | `OSS_TEST_ACCESS_KEY_ID`, `OSS_TEST_ACCESS_KEY_SECRET`, `OSS_TEST_BUCKET`, `OSS_TEST_ENDPOINT`, `OSS_TEST_INTERNAL_ENDPOINT`, `OSS_SOURCE_BUCKET` | Used in helper scripts (`common/aliyun.py`, DataWorks checkers) and for feature sync jobs. |
| Redis instances | `YOYO_REDIS_HOST`, `YOYO_REDIS_PORT`, `YOYO_REDIS_USERNAME`, `YOYO_REDIS_PASSWORD`, `YOYO_REDIS_DB` | Primary Redis instance used for online feature pushes. |
| Feature Redis | `YOYO_FEATURE_REDIS_HOST`, `YOYO_FEATURE_REDIS_PORT`, `YOYO_FEATURE_REDIS_USERNAME`, `YOYO_FEATURE_REDIS_PASSWORD`, `YOYO_FEATURE_REDIS_DB` | Secondary Redis instance that caches feature bins. |

Keep these variables in a secure vault or CI secret store and never commit actual values. The defaults inside the repo are non-functional placeholders like `YOUR_OSS_ACCESS_KEY_ID` so running code without overrides will raise authentication errors—this is intentional to prevent accidental leakage.

## Typical Workflow
1. **Configure schema & slots**: Update the CSV/JSON assets referenced in `train_config.py` (schema definitions, slot mappings, feature groups, sequence configs).
2. **Schedule jobs**: Use the sample crontabs under each `config/<task>/crontab` as a starting point for production scheduling.
3. **Monitor metrics**: `common/save_eval_metric.py` can push evaluation results to ODPS tables, which you can visualize downstream.
4. **Manage artifacts**: After export, models land in `${MODEL_ROOT}/${MODEL_TASK}/export_dir`. `common/upload.py` then uploads them to the OSS path configured in the train config.
5. **Serve online**: Deploy the exported SavedModel via TF-Serving (Docker example under `test`). Use the provided HTTP bodies under `config/*/body.json` or adapt them to your business requests.

## Data & Privacy Notes
- Internal server URLs, Redis hosts, ODPS endpoints, and tokens have been replaced with placeholders. Double-check `common/connect_config.py`, `common/aliyun.py`, and the DataWorks scripts in `config/*/run_datawork_*.py` to confirm your own secrets are injected via env vars before deployment.
- The `bin/logs` directory contains historical log snapshots. Remove or git-ignore it if you do not want to publish operational traces.
- Make sure any additional artifacts (datasets, dumps, screenshots) undergo the same sanitization pass before sharing this project.

## Troubleshooting
- **Dataset decode errors**: `bin/run.sh` automatically retries downloads and clears corrupted inputs via `common/clear_history_data.py` when it detects decoding issues.
- **Distributed runs**: Ensure `TF_CONFIG` is exported correctly; `tfconfig.py` generates a single-worker setup by default but can be extended to multi-worker if required.
- **OSS/ODPS auth failures**: Verify the environment variables listed above and confirm the network path (internal vs external endpoint) matches your runtime.

Feel free to adapt the configs, add new architectures under `models/`, or wrap different feature stores while keeping this scaffold as the backbone of your UCSD MSDS submission.
