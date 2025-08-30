
import os
import json
import pandas as pd
import seaborn as sns

# CHANGE THIS DEPENDING ON WHICH ROUND YOU ARE PROCESSING
round_num = 'round6'
#########################################################

output_name = f'/home/mob999/ReportGrounding/data/ct_rate_categorization/{round_num}'
# dictionary to store the counts of each category 
category_counts = {
    '1': 0,
    '1a': 0, '1b': 0, '1c': 0, '1d': 0, '1e': 0, '1f': 0,
    '2': 0,
    '2a': 0, '2b': 0, '2c': 0, '2d': 0, '2e': 0, '2f': 0, '2g': 0, '2h': 0,
    '3': 0,
    '3a': 0, '3b': 0, '3c': 0, '3d': 0, '3e': 0, '3f': 0, '3g': 0, '3h': 0, '3i': 0, '3j': 0, '3k': 0,
    '4': 0,
    '4a': 0, '4b': 0, '4c': 0, '4d': 0,
    '5': 0,
    '5a': 0, '5b': 0, '5c': 0,
    '6': 0,
    '6a': 0, '6b': 0, '6c': 0, '6d': 0, '6e': 0,
    '7': 0,
    '7a': 0, '7b': 0, '7c': 0, '7d': 0, '7e': 0,
    '8': 0,
    '8a': 0, '8b': 0, '8c': 0, '8d': 0, '8e': 0, '8f': 0, '8g': 0,
    '9': 0,
    '9a': 0, '9b': 0, '9c': 0, '9d': 0, '9e': 0, '9f': 0,
    '10': 0,
    '10a': 0, '10b': 0, '10c': 0,
    '11': 0,
    '11a': 0, '11b': 0, '11c': 0,
    '12': 0
}

# splits = ['train', 'val']
splits = ['train']

samples = {
}

for split in splits:    
    categorized = f'{output_name}/{split}/categorized_findings/categorization'
    pos_findings = f'{output_name}/{split}/positive_finding_reports'
       
    for category_file in os.listdir(categorized):
        if category_file.endswith('.json'):
            
            file_name = category_file.split('.')[0]
            sample_counts = category_counts.copy()
            
            with open(os.path.join(categorized, category_file), 'r') as file:
                data = json.load(file)
            with open(os.path.join(pos_findings, category_file), 'r') as f:
                pos = json.load(f)
                
            for k, v in data.items():
                if v == '12':
                    # print(f'Parent Category: {v}')
                    sample_counts[v] += 1
                else:
                    sub_category_name = v
                    parent_category = str(v[:-1])
                    # print(f'Sub Category: {sub_category_name}, Parent Category: {parent_category}')
                    sample_counts[sub_category_name] += 1
                    sample_counts[parent_category] += 1
                
                
            samples[file_name] = sample_counts
            

counts_df = pd.DataFrame.from_dict(samples, orient='index')
counts_df = counts_df.sort_index(key=lambda x: x.map(lambda k: (k.split('_')[0], int(k.split('_')[1]), k.split('_')[2])))


counts_df.to_excel(f'{output_name}/{round_num}_category_counts.xlsx')

# Find parent_category counts
parent_categories = [str(x) for x in range(1, 13)]
parent_category_counts = {parent_category: counts_df[parent_category].sum() for parent_category in parent_categories}

# Save a histogram distribution of all the parent categories
barplot = sns.barplot(x=list(parent_category_counts.keys()), y=list(parent_category_counts.values()))
for index, value in enumerate(parent_category_counts.values()):
    barplot.text(index, value, str(value), ha='center', va='bottom')
barplot.set_title('Counts of Each Parent Category')
barplot.set_xlabel('Parent Categories')
barplot.set_ylabel('Counts')
figure = barplot.get_figure()
figure.savefig(f'{output_name}/{round_num}_total_parent_category_hist.png')

# Filter for rows that have (non-zero value in 1, 2, 3a, or 8) and rows that are 0 in (3b, 3c, 3d, 3e, 3f, 3g, 3h, 3i, 3j, 3k, 4, 5, 6, 7, 9, 10, 11, 12)
filtered_df = counts_df[((counts_df['1'] > 0) | (counts_df['2'] > 0) | (counts_df['8'] > 0) | counts_df['3a'] > 0)
                            & 
                        ((counts_df['3b'] == 0) & (counts_df['3c'] == 0) & (counts_df['3d'] == 0) & 
                         (counts_df['3e'] == 0) & (counts_df['3f'] == 0) & (counts_df['3g'] == 0) & 
                         (counts_df['3h'] == 0) & (counts_df['3i'] == 0) & (counts_df['3j'] == 0) & (counts_df['3k'] == 0) &
                         (counts_df['4'] == 0) & (counts_df['5'] == 0) & (counts_df['6'] == 0) &
                         (counts_df['7'] == 0) & (counts_df['9'] == 0) & (counts_df['10'] == 0) & (counts_df['11'] == 0) & (counts_df['12'] == 0))
                        ]

filtered_df.to_excel(f'{output_name}/{round_num}_scans_to_annotate.xlsx')

print(f'Filtered Scans to Annotate: {len(filtered_df)}')
print(f'Number of rows that have the index containing "train": {len(filtered_df[filtered_df.index.str.contains("train")])}')
print(f'Number of rows that have the index containing "valid": {len(filtered_df[filtered_df.index.str.contains("valid")])}')
