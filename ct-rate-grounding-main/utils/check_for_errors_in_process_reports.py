import json
import os
import pandas as pd

# CHANGE THIS DEPENDING ON WHICH ROUND / SPLIT YOU ARE PROCESSING
round_num = 'round2'
split = 'val' # 'train' or 'val'
#########################################################

root = f'./outputs/{round_num}/{split}/stats'
files = os.listdir(root)
files_with_errors = []

for file in files:
    with open(os.path.join(root, file), 'r') as f:
        data = json.load(f)
        if isinstance(data['translation'], str) or isinstance(data['extraction'], str):
            print(f'ERROR WITH {file}')
            print(f'Translation: {data["translation"]}')
            print(f'Extraction: {data["extraction"]}')
            print()
            files_with_errors.append(file)
        # else:
            # print(f'NO ERROR WITH {file} | Translation: {data["translation"]} | Extraction: {data["extraction"]}')
print(f'Files with errors: {files_with_errors}')