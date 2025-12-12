#!/usr/bin/env python
# coding=utf-8
import sys, json

workers = ["192.168.50.251:%d" % (5330 + i) for i in range(2)] + \
    ["192.168.126.173:%d" % (5330 + i) for i in range(1)]# + \
    # ["192.168.126.172:%d" % (5330 + i) for i in range(1)]

tf_config = {
    'cluster': {
        'ps': ['192.168.50.236:55555'],
        'chief': ['192.168.50.236:5550'],
        'worker': workers,
    },
    "task": {
        'type': sys.argv[1],
        'index': int(sys.argv[2])
    }
}
tf_config = {}
print(json.dumps(tf_config))
