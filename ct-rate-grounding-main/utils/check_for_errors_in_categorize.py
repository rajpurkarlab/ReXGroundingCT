import os
import json

# CHANGE THIS DEPENDING ON WHICH ROUND / SPLIT YOU ARE PROCESSING
round_num = 'round2'
split = 'val' # 'train' or 'val'
#########################################################

results_root = f'./outputs/{round_num}/{split}/categorized_findings/stats'
results = [x for x in os.listdir(results_root) if x.endswith('.json')]
for result in results:
    with open(os.path.join(results_root, result), 'r') as file:
        data = json.load(file)
    if not isinstance(data['categorization'], dict) and data['categorization'] != 'No Positive Findings':
    # if not isinstance(data['categorization'], dict):
        print(f'result: {result}')
        print(data['categorization'])
        print()
