# coding=utf-8
import os
import multiprocessing
import logging
import json
import argparse
from odps import ODPS

from common.connect_config import *

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
    partitions = glob_partitions.format(day=day)
    print(f"generate_body: partitions:{partitions}")
    #################################################################################
    table = get_odps().get_table(table_name)
    sql = f"""
        select features
        from {table_name}
        where {partitions.replace(',', ' and ')}
        AND requestid in (
            "3bb86e226d1e456ea4d754b1ed4da27f1758948614<>hn.apprec.r1", 
            "eee4356375f24efcab6b7dfbe58376361758949340<>hn.apprec.r1"
            )
        AND combination_un_id = "613552"
    """
    print(sql)
    body_json_str = {"inputs": {}}

    if table.exist_partition(partitions):
        datas = get_odps().execute_sql(sql).open_reader()
        for row in datas:
            features = row["features"]
            lst = features.strip("\n").split("\002")
            for fea_name, feat_val in zip(schema, lst):
                if fea_name not in body_json_str["inputs"]:
                    body_json_str["inputs"][fea_name] = []
                body_json_str["inputs"][fea_name].append(str(feat_val))

    print(body_json_str)
    with open(body_json_name, "w", encoding='utf-8') as wf:
        json.dump(body_json_str, wf, indent=4, ensure_ascii=False)

def main():
    # if os.path.exists(TrainConfig.body_json_name):
    #     with open(TrainConfig.body_json_name, encoding='utf-8') as f:
    #         if len(f.read().split()) > 1000:
    #             return
    gen_body_json(binning_table_name, args.day)

def get_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--day", default="", type=str, help="")
    args, unknown = parser.parse_known_args()
    print((args, unknown))
    return args

if __name__ == '__main__':
    args=get_args()
    schema_path = "/opt/huangmian/yoyo_model/config/O31/schema.conf"
    binning_table_name = "ocpx_mutil_label_cvr_order_sample_train_data"
    glob_partitions = "idx_date='{day}',feature_version='O31',durations='1'"
    body_json_name = "/opt/huangmian/yoyo_model/test/test_body.json"
    with open(schema_path) as f:
        schema = [l.strip("\n") for l in f if not (l.startswith("#") or l.startswith("label"))]
    main()

