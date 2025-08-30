import os
import json

# CHANGE THIS DEPENDING ON WHICH ROUND YOU ARE PROCESSING
round_num = 'round3'

#########################################################

root = f'./outputs/{round_num}'

splits = ['train', 'val']
for split in splits:
    
    os.makedirs(os.path.join(root, split, 'pos_neg_labeled_reports'), exist_ok=True)
    os.makedirs(os.path.join(root, split, 'positive_finding_reports'), exist_ok=True)
    
    extracted_reports = sorted(
        [x for x in os.listdir(os.path.join(root, split, 'extracted_reports')) if x.endswith('.json')],
        key=lambda x: (x.split('_')[0], int(x.split('_')[1]))
    )

    for report in extracted_reports:
        print(report)
        with open(os.path.join(root, split, 'extracted_reports', report), 'r') as extracted_file:
            report_json = json.load(extracted_file)
            
            pos_idx = 1
            neg_idx = 1
            pos_neg_labeled_report_json = {}
            
            positive_finding_report = {}
            
            for k, v in report_json.items():
                if v[2] == 'Y':
                    modified_k = f'{k},P{pos_idx}'
                    positive_finding_report[f'P{pos_idx}'] = v[0]
                    pos_idx += 1
                elif v[2] == 'N':
                    modified_k = f'{k},N{neg_idx}'
                    neg_idx += 1
                else:
                    print(f"ERROR: Neither positive nor negative for {report}, sentence {k}")
                
                pos_neg_labeled_report_json[modified_k] = v
            
            # Save to new files
            with open(os.path.join(root, split, 'pos_neg_labeled_reports', report), 'w') as f:
                json.dump(pos_neg_labeled_report_json, f, indent=4)
            with open(os.path.join(root, split, 'positive_finding_reports', report), 'w') as f:
                json.dump(positive_finding_report, f, indent=4)
            

    


 