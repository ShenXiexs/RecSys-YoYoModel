#!/usr/bin/env python3
# Minimal example: call RankMixer via the Python API (library usage).
#
# Usage:
#   python rankmixer/example_api.py \
#     --task RankMixer_Shen0202 \
#     --time_str 20260201 \
#     --overrides_path /path/to/overrides.json \
#     --groups_path /path/to/semantic_groups.json

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rankmixer import train


def build_parser():
    parser = argparse.ArgumentParser(description="RankMixer Python API example")
    parser.add_argument("--task", required=True, help="Task name, e.g. RankMixer_Shen0202")
    parser.add_argument("--time_str", required=True, help="YYYYMMDD or YYYYMMDDHHMM")
    parser.add_argument("--end_time_str", help="Optional end time")
    parser.add_argument("--overrides_path", help="Overrides JSON/YAML file path")
    parser.add_argument("--groups_path", help="Semantic groups JSON/YAML file path")
    return parser


def main():
    args = build_parser().parse_args()
    train(
        task=args.task,
        time_str=args.time_str,
        end_time_str=args.end_time_str,
        overrides_path=args.overrides_path,
        groups_path=args.groups_path,
    )


if __name__ == "__main__":
    main()
