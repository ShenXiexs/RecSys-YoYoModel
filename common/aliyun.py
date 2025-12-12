import os
import oss2
from odps import ODPS
from common.connect_config import *


def get_odps(your_default_project='adx_dmp'):
    """ 链接odps """
    odps_obj = ODPS(your_accesskey_id, your_accesskey_secret, your_default_project, endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj


def get_oss(bucket_name='adx-oss',if_inner_net=False):
    """ 链接oss """
    access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', 'ALIYUN_ACCESS_KEY_ID_PLACEHOLDER')
    access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', 'ALIYUN_ACCESS_KEY_SECRET_PLACEHOLDER')
    bucket_name = os.getenv('OSS_TEST_BUCKET', bucket_name)
    endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing.aliyuncs.com')  # 外网
    if if_inner_net:
        endpoint = os.getenv('OSS_TEST_ENDPOINT', 'http://oss-cn-beijing-internal.aliyuncs.com')  # 内网
    bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
    return bucket


def execute_sql(sql,project_name='adx_dmp',hints=None):
    print('===> sql:',sql)
    return get_odps(project_name).execute_sql(f'{sql}',hints=hints).open_reader()
