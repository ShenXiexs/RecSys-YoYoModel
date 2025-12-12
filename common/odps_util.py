from odps import ODPS
from pandas import DataFrame
from common.connect_config import *


def get_odps():
    """ 链接odps """
    odps_obj = ODPS(your_accesskey_id, your_accesskey_secret, your_default_project, endpoint=your_end_point,
                    tunnel_endpoint=tunnel_endpoint)
    return odps_obj


def write_df2odps(datas:DataFrame,table_name,partitions):

    odps = get_odps()
    table = odps.get_table(table_name)
    print(f'write_df2odps:{table_name},partitions:{partitions}')
    if not table.exist_partition(partitions):
        table.create_partition(partitions)
    with table.open_writer(partitions,reopen=False) as writer:
        writer.write(datas.values.tolist())
