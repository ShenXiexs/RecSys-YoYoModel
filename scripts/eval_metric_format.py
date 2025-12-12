# -*- coding: utf-8 -*-
# @Time : 2025/11/6 17:33
# @Author : huangmian
# @File : eval_metric_format.py
import pandas as pd
string=""""""


def main():
    res = []
    for line in string.split("\n"):
        if not line:
            continue
        metrics = {}
        for l in line.split(","):
            l = l.strip().split("=")
            if l[0].strip() in ['task', 'time'] or 'auc' in l[0].strip() or 'pcoc' in l[0].strip():
                metrics[l[0].strip()] = [l[1].strip() if l[0].strip() in ('task', 'time') else float(l[1].strip())]
        res.append(pd.DataFrame(metrics))
        # print(res[-1])
    res = pd.concat(res)
    # print(res)
    res.to_excel(f"metrics_{metrics['task'][0]}.xlsx")

if __name__ == '__main__':
    main()
