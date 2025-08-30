import os
import json
import pandas as pd
import re

# CHANGE THIS DEPENDING ON WHICH ROUND YOU ARE PROCESSING
round_num = 'round6'
scans_root_path = f'/home/mob999/rajpurkarlab/datasets/ReXGroundingCT/CT-RATE-Round6/round6_2'

#########################################################
findings_root_path = f'/home/mob999/ReportGrounding/data/ct_rate_categorization/{round_num}'

scans_to_annotate = pd.read_excel(f'/home/mob999/ReportGrounding/data/ct_rate_categorization/{round_num}/{round_num}_scans_to_annotate.xlsx')
json_save_path = f'/home/mob999/ReportGrounding/data/ct_rate_categorization/{round_num}/round6_2.json'

files = scans_to_annotate['Unnamed: 0'].tolist()
points = []
for i, file in enumerate(files):
    # Have to find the scan name for the file
    found = False
    for scan_file in os.listdir(scans_root_path):
        if file in scan_file:
            file = scan_file
            found = True
            break
    if not found:
        continue

    reconstruction_name = file.split('.')[0]
    study_name = '_'.join(reconstruction_name.split('_')[:3])
    
    split = 'train' if 'train' in file else 'val'
    positive_findings_path = os.path.join(findings_root_path, split, 'positive_finding_reports', f'{study_name}.json')
    categorization_path = os.path.join(findings_root_path, split, 'categorized_findings', 'categorization', f'{study_name}.json')
    
    with open(positive_findings_path, 'r') as f:
        positive_findings = json.load(f)
    with open(categorization_path, 'r') as f:
        categorization = json.load(f)
    
    metadata = {}
    for p_code, finding in positive_findings.items():
        # get categorization for this finding
        category = categorization[p_code]
        # if category is '8b' or '8d', replace with OMIT
        # if category == '8b' or category == '8d':
        #     finding = f'OMIT'
            # print(f'file: {file}, p_code: {p_code}, finding: {finding}')
        category = int(re.match(r'\d+', category).group())
        if category == 1 or category == 2:
            metadata[p_code] = finding
    
    if len(metadata.keys()) == 0:
        continue
    
    task_dict = {
        'name': file,
        "series": [
            {
                "items": file
            }
        ],
        'metaData': metadata
    }
    
    points.append(task_dict)

# Save points to a json file
with open(json_save_path, 'w') as f:
    json.dump(points, f, indent=4)