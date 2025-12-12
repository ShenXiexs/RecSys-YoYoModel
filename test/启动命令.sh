#sh downodps_test.sh O25_v3

cd /opt/huangmian/aimodel/dnn_model/yoyo_model/bin && nohup bash run.sh O25_v3 >> logs/nohup_O25_v3.log 2>&1 &
cd /opt/huangmian/aimodel/dnn_model/yoyo_model/bin && nohup bash run.sh TC3 20250815 >> logs/nohup_TC3.log 2>&1 &
nohup bash test.sh TC3 20250817 >> logs/TC3/nohup_TC3.log 2>&1 &
nohup bash run.sh TC3 20250904 >> logs/TC3/nohup_TC3.log 2>&1 &

nohup bash /opt/huangmian/aimodel/dnn_model/yoyo_model/bin/test.sh O25_v3 20250827 > /opt/huangmian/aimodel/dnn_model/yoyo_model/bin/logs/nohup_O25_v3.log 2>&1 & tail -f /opt/huangmian/aimodel/dnn_model/yoyo_model/bin/logs/nohup_O25_v3.log

# PPU  TC1_ctr
cd /mnt/huangmian/yoyo_model/bin
nohup bash test.sh TC1_ctr 20250807 > logs/TC1_ctr/nohup_202509021107.log 2>&1 & tail -f logs/TC1_ctr/nohup_202509021107.log

cd /mnt/huangmian/yoyo_model/bin
nohup bash test.sh TC1_ctr2 20250823 > logs/TC1_ctr2/nohup_20250903103400.log 2>&1 & tail -f logs/TC1_ctr2/nohup_20250903103400.log

# O31
cd /opt/huangmian/aimodel/dnn_model/yoyo_model/bin

nohup bash test.sh O31 20250801 > logs/O31/nohup_20250924110000.log 2>&1 & tail -f logs/O31/nohup_20250924110000.log
# O31_v2
nohup bash test.sh O31_v2 20250801 > logs/O31_v2/nohup_20250924110000.log 2>&1 & tail -f logs/O31_v2/nohup_20250924110000.log
# O31_v3
nohup bash test.sh O31_v3 20250801 > logs/O31_v3/nohup_20250924110000.log 2>&1 & tail -f logs/O31_v3/nohup_20250924110000.log
nohup bash run.sh O31_v3 20250922 > logs/O31_v3/nohup_20250925160000.log 2>&1 & tail -f logs/O31_v3/nohup_20250925160000.log
# O31_v3_2
nohup bash test.sh O31_v3_2 20250801 > logs/O31_v3_2/nohup_2025098160000.log 2>&1 & tail -f logs/O31_v3_2/nohup_2025098160000.log
# O31_v4
mkdir -p logs/O31_v4
nohup bash test.sh O31_v4 20250801 > logs/O31_v4/nohup_20250924110000.log 2>&1 & tail -f logs/O31_v4/nohup_20250924110000.log
# O31_v4_2
nohup bash test.sh O31_v4_2 20250801 > logs/O31_v4_2/nohup_20250928170000.log 2>&1 & tail -f logs/O31_v4_2/nohup_20250928170000.log
# O31_v5
nohup bash test.sh O31_v5 20250801 > logs/O31_v5/nohup_20250929120000.log 2>&1 & tail -f logs/O31_v5/nohup_20250929120000.log
# O31_v6
mkdir -p logs/O31_v6
nohup bash test.sh O31_v6 20250801 > logs/O31_v6/nohup_20250929120000.log 2>&1 & tail -f logs/O31_v6/nohup_20250929120000.log
nohup bash test.sh O31_v6 20250928 > logs/O31_v6/nohup_20250930120000.log 2>&1 & tail -f logs/O31_v6/nohup_20250930120000.log
# O31_v7
mkdir -p logs/O31_v7
nohup bash test.sh O31_v7 20251010 > logs/O31_v7/nohup_20251011170000.log 2>&1 & tail -f logs/O31_v7/nohup_20251011170000.log

