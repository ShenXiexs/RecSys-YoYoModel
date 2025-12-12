import requests
import json
from common.utils import train_config

with open(train_config.fg_path, "r") as json_f:
    schema_fea2idx_dict = {v['feature_name']: ("f" + str(idx + 1)) for idx, v in
                           enumerate(json.load(json_f)['features'])}

data={}
with open(train_config.body_json_name) as f:
    bodys = json.load(f)["inputs"]
    for name, feat in bodys.items():
        if name == "label":
            continue
        name = schema_fea2idx_dict[name]
        data[name] = [str(v) for v in feat] 
    
body={"inputs": data}
print(body)
# Convert data to JSON format
json_data = json.dumps(body)

# Define the URL
url = "http://localhost:8501/v1/models/model:predict"

# Send POST request
response = requests.post(url, data=json_data)

# Print the response
print(json.loads(response.text))

