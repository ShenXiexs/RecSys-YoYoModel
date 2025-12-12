import os
import json
model_version = "O31"

dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(dirname)
fg = json.load(open("config/{}/fg.json".format(model_version)))
fg_features = [feat['feature_name'] for feat in fg['features']]
slot = open("config/{}/test_slot".format(model_version)).read().split()

if len(fg_features) != len(slot):
    print(f"please check fg['features'] and slot.conf, length not equal")
    exit(-1)

for i,(f1,f2) in enumerate(zip(fg_features, slot)):
    if f1!=f2:
        print(f"index={i}, fg_features={f1}, slot={f2}, not equal, please check")
