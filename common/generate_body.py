# coding=utf-8
import os
import multiprocessing
import logging
import json
import argparse
from odps import ODPS

from common.connect_config import *
try:
    from common.utils import schema,train_config as TrainConfig, seq_features_index, seq_idxs, seq_feat_names
except:
    from utils import schema, train_config as TrainConfig

multiprocessing.log_to_stderr()
logger = multiprocessing.get_logger()
logger.setLevel(logging.INFO)

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_odps():
    """ 链接odps """
    odps_obj = ODPS(your_accesskey_id,
                    your_accesskey_secret,
                    your_default_project,
                    endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj


def gen_body_json(table_name,day,feature_version=None,durations='1'):
    partitions = TrainConfig.partitions.format(day=day)
    print(f"generate_body: partitions:{partitions}")
    #################################################################################
    table = get_odps().get_table(table_name)
    sql = f"""
        select features{",seq_features" if seq_features_index else ""}
        from {table_name}
        where {partitions.replace(',', ' and ')}
        DISTRIBUTE BY RAND()
        SORT BY RAND()
        limit 5
    """
    print(sql)
    body_json_str = {"inputs": {}}

    if table.exist_partition(partitions):
        datas = get_odps().execute_sql(sql).open_reader()
        for row in datas:
            features = row["features"]
            lst = features.strip("\n").split(TrainConfig.features_sep)
            for fea_name, feat_val in zip(schema, lst):
                if fea_name not in body_json_str["inputs"]:
                    body_json_str["inputs"][fea_name] = []
                body_json_str["inputs"][fea_name].append(str(feat_val))
            if seq_features_index:
                seq_features = row["seq_features"]
                seq_lst = seq_features.strip("\n").split(TrainConfig.features_sep)
                for i, (start, end, index) in enumerate(seq_idxs):
                    fea_name = seq_feat_names[i]
                    if fea_name not in body_json_str["inputs"]:
                        body_json_str["inputs"][fea_name] = []
                    body_json_str["inputs"][fea_name].append(TrainConfig.features_sep.join(seq_lst[start:end]))
    print(f"body_json_str={body_json_str}")
    with open(TrainConfig.body_json_name, "w", encoding='utf-8') as wf:
        json.dump(body_json_str, wf, indent=4, ensure_ascii=False)

def main():
    if os.path.exists(TrainConfig.body_json_name):
        with open(TrainConfig.body_json_name, encoding='utf-8') as f:
            if len(f.read().split()) > 1000:
                return
    gen_body_json(TrainConfig.binning_table_name, args.day)

def get_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--day", default="", type=str, help="")
    args, unknown = parser.parse_known_args()
    print((args, unknown))
    return args

if __name__ == '__main__':
    args=get_args()
    main()

