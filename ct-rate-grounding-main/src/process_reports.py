from openai import AzureOpenAI
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import time
import json
import os
from multiprocessing import Pool

nltk.download('punkt')

# Using valid_333_a_tokenized.txt as example
TRANSLATION_PROMPT = """
You are an expert radiologist with extensive experience in reading and writing radiology reports in American English.
**OBJECTIVE**:
1. Rewrite the provided 'Findings' and 'Impressions' sections of radiology reports to ensure they use the appropriate medical terminology, phrasing, and grammar that American radiologists would use.
2. Categorize each rewritten sentence in the 'Findings' section into one of 7 regions to structure the 'Findings' section of the report.

**RULES**:
1. Correct any unusual phrases, awkward grammar, or non-standard terminology. For example, English-speaking American radiologists would replace these phrases:
- Instead of "millimetric", say "subcentimeter" 
- Instead of "sequelae" or "pleuroparenchymal sequelae," say "scarring" or "fibrosis" 
- Instead of "esophagus calibration was normal," say "esophagus is normal in caliber"
- Instead of "no space-occupying lesion," say "no lesion"
- Instead of "within the cross-sectional area," "within the examined area," or "sections visualized," say "within the field of view"
2. Ensure each rewritten sentence accurately conveys the same medical information and details as the corresponding sentence in the original report. DO NOT LOSE OR CHANGE ANY NUMERICAL MEASUREMENT DETAILS OF ANY FINDINGS. YOU WILL BE PENALIZED IF YOU MISS OR CHANGE THE MEANING OF INFORMATION IN THE ORIGINAL REPORT.
3. Make sure each sentence in the original report is rewritten and categorized appropriately. YOU WILL BE PENALIZED IF YOU MISS ANY SENTENCES FROM OR ADD ANY NEW SENTENCES TO THE ORIGINAL REPORT.
4. Remove any irrelevant comments that should not be in a radiology report. For example, in the following sentences: "Impressions: Pleural effusion and concomitant compression atelectasis in both lungs. Nonspecific nodules in both lungs. Cardiomegaly and minimal pericardial effusion. Patient 14.10.", the sentence "Patient 14.10." is irrelevant and should not be included in the rewritten report.

You will be given the Findings and Impressions sections of a radiology report, separated by sentence. Each sentence is labeled with a sentence code, containing a letter and a number. The code letter will be either F, for Findings, or I, for Impressions depending on which section the sentence comes from. The code number refers to the sentence's number in its respective section. Thus, the first sentence in Findings has code F0, the second sentence in Findings has code F1, and so on, and the first sentence in Impressions has code I0, the second sentence in Impressions has code I1, and so on."

The output should be in JSON format, structured as follows:
* 'Findings' should map to a dictionary with the 7 categories. Include all sentences, including all details and measurements, from the 'Findings' section and use any information from the 'Impressions' section to correctly supplement the details (anatomical region, measurements, etc.) of these 'Findings' sentences. The 7 category keys should be 'Lungs/Airways/Pleura', 'Heart/Vessels', 'Mediastinum/Hila', 'Chest wall/Axilla', 'Lower neck', 'Bones', 'Upper abdomen'. The values should be lists that contain the rewritten sentences in the report that fall into the anatomical region specified by the key. DO NOT CREATE ANY NEW CATEGORIES. PLACE EACH FINDING SENTENCE INTO THE CORRECT CATEGORY. If the Findings section is "Not Given," all of the lists in each of the 7 categories should be empty.
* 'Impressions' should map to a string with the rewritten sentences in the 'Impressions' section of the report. If the Impressions section is "Not Given," keep "Not Given" as the value in this section.

BE SURE TO FOLLOW ALL RULES METICULOUSLY. Think through all rules and steps before answering.

Follow this example:
**Example**
**Input**:
{'F0': 'Trachea and both main bronchi were in the midline and no obstructive pathology was observed in the lumen.', 'F1': 'The mediastinum could not be evaluated optimally in the non-contrast examination.', 'F2': 'As far as can be seen; mediastinal main vascular structures, heart contour, size are normal.', 'F3': 'Pericardial effusion-thickening was not observed.', 'F4': 'Thoracic esophagus calibration was normal and no significant tumoral wall thickening was detected.', 'F5': 'No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected.', 'F6': 'When examined in the lung parenchyma window; A peripheral subcapsular crazy paving pattern was formed in the posterobasal segment of the right lung, nodular ground glass opacity was observed, and the appearance is highly suspicious for ultra-early Covid-19 pneumonia.', 'F7': 'It is recommended to be evaluated together with clinical and laboratory.', 'F8': 'A millimetric nonspecific parenchymal nodule was observed in the lateralabasal segment of the lower lobe of the left lung.', 'F9': 'No mass lesion with distinguishable borders was detected in both lungs.', 'F10': 'As far as can be seen in the sections, millimetric calculus was observed in the middle part of the right kidney.', 'F11': 'Accessory spleen with a diameter of 13 mm was observed adjacent to the lower pole of the spleen.', 'F12': 'Other upper abdominal organs are normal.', 'F13': 'No space-occupying lesion was detected in the liver that entered the cross-sectional area.', 'F14': 'Bilateral adrenal glands were normal and no space-occupying lesion was detected.', 'F15': 'Bone structures in the study area are natural.', 'F16': 'Vertebral corpus heights are preserved.', 'I0': 'High suspicious findings for ultra-early period Covid-19 pneumonia in the right lung posterobasal segment; it is recommended to be evaluated together with clinical and laboratory.', 'I1': 'Millimetric nonspecific parenchymal nodule in the lateralabasal segment of the left lung lower lobe .', 'I2': 'Right nephrolithiasis'}
**Output**:
{
  "Findings": {
    "Lungs/Airways/Pleura": [
      "The trachea and both main bronchi are centrally positioned and patent.",
      "In the lung parenchyma, there is a peripheral subcapsular crazy paving pattern in the posterobasal segment of the right lung along with nodular ground glass opacity, suggestive of potential early-stage COVID-19 pneumonia.",
      "No masses are identified in either lung.",
      "An indeterminate subcentimeter parenchymal nodule is noted in the lateral basal segment of the left lower lobe."
    ],
    "Heart/Vessels": [
      "Mediastinal large vessels and heart contour and size appear normal.",
      "No pericardial effusion or thickening is evident."
    ],
    "Mediastinum/Hila": [
      "Mediastinal evaluation is limited on this non-contrast exam.",
      "Thoracic esophagus is normal in caliber without significant wall thickening.",
      "No enlarged lymph nodes are detected in prevascular, pre-paratracheal, subcarinal, or bilateral hilar-axillary regions."
    ],
    "Chest wall/Axilla": [],
    "Lower neck": [],
    "Bones": [
      "Bone structures within the field of view are normal.",
      "Vertebral body heights are preserved."
    ],
    "Upper abdomen": [
      "A subcentimeter calculus is seen in the mid portion of the right kidney.",
      "An accessory spleen measuring 13 mm in diameter is located adjacent to the lower pole of the spleen.",
      "Other upper abdominal organs appear normal.",
      "There are no liver lesions within the field of view.",
      "The bilateral adrenal glands appear normal with no lesions identified."
    ]
  },
  "Impressions": "Findings are highly suspicious for early-stage COVID-19 pneumonia in the right lung posterobasal segment; further evaluation with clinical and laboratory correlation is recommended. An indeterminate subcentimeter parenchymal nodule is present in the lateral basal segment of the lower lobe of the left lung. Evidence of right nephrolithiasis is noted."
}
"""

EXTRACTION_PROPMT = """
You are an expert radiologist. You are helping process reports from chest CT scans.
You will be given the Findings and Impressions sections of a radiology report. The input report structure is a dictionary with two keys: "Findings" and "Impressions."
* Findings:
   - Contains nested keys that represent sentence codes for sentences in the "Findings" section of the report. The letters represent which region the finding is from (i.e. 'L' corresponds to Lung). The numbers following each letter correspond to the numbered sentences for that region.
* Impressions:
   - Contains nested keys that represent sentence codes for sentences in the "Impressions" section of the report. The letter 'I' represents the Impressions section. The numbers correspond to the numbered sentences in the 'Impressions' section, which is not broken down by region.
   
**OBJECTIVE**: Please extract distinct phrases from the radiology report which refer to objects, any findings, or anatomies that are visible in a CT scan, or the absence of such. For each phrase you extract, please also specify the sentence code(s) that the phrase comes from. Also specify if the phrase represents an abnormal finding or not and if the phrase contains terminology that refers to a prior report. The main objective is to extract phrases which refer to things which can be located on a chest CT scan, or confirmed not to be present."

**RULES**:
1. If a sentence describes multiple distinct findings (whether abnormal or normal), split them up into separate sentences so each sentence represents a distinct finding. For example, the sentence "Widespread peribronchial thickening and tree-in-bud opacities in the upper lobe apicoposterior, lingular segment, and lower lobes of the left lung" should be split into: "Widespread peribronchial thickening in the left lung" AND "Tree-in-bud opacities in the upper lobe apicoposterior segment, lingular segment, and lower lobes of the left lung"
2. If multiple sentences specify the same distinct finding, extract them together as one sentence and combine all details specified (including anatomical location, measurements, stable or not, etc.) across all sentences for the finding.
3. Exclude clinical speculation, interpretation, and suspicion. For example, please remove phrases like:
- "highly suggestive of pneumonia"
- "likely benign"
- "suspicious for metastasis"
- "concerning for metastasis"
4. If a phrase in a sentence ends in a question mark, do not consider it as a finding.
5. Exclude recommendations (e.g. "Recommend a CT").
6. For all centimeter measurements, reduce the number to one decimal point. For all millimeter measurements, round the number to the nearest whole number.
7. If you miss a finding that be visible in the CT, you will be penalized! You MUST INCLUDE ANY FINDING THAT IS VISIBLE. For example, although the phrase "Small lymph nodes are present in both axillary regions with a short axis measuring up to 8 mm" is not clinically significant, it can still be seen on a CT scan so you must include it in the output!!!
8. The output should be in JSON format, structured as follows:
- The keys should be 'SX', where X is the phrases enumerated.
- The values should be Lists with index 0 being the phrase you will extract, index 1 is the string of sentence codes will be the sentence code(s) that the phrase comes from, delimited by commas (i.e. 'H0,H1,I0' if the phrase comes from sentences H0, H1, and I0), index 2 is 'Y' or 'N' which indicates if the phrase represents an abnormality or not, and index 3 is 'Y' or 'N' which indicates if the phrase contains terminology that refers to a prior report; for example, "stable" and "unchanged" would indicate a prior report.

BE SURE TO FOLLOW ALL RULES METICULOUSLY. Think through all rules and steps before answering.

Follow this example:
**Example**
**Input**:
{
    "Findings": {
        "L0": "The trachea and both main bronchi are patent.",
        "L1": "Bilateral lower lobe pleuroparenchymal opacities extend from the central regions to the pleura, with significant bronchial wall thickening and a mass-like appearance of the left lower lobe bronchus, all of which are stable.",
        "L2": "A stable pleural effusion is noted in the right hemithorax.",
        "H0": "The mediastinal main vascular structures, heart contour, and size are within normal limits.",
        "H1": "The thoracic aorta exhibits a normal diameter.",
        "H2": "No pericardial effusion or thickening is identified.",
        "M0": "The thoracic esophagus is normal in caliber without significant tumoral thickening.",
        "M1": "Lymph nodes in the mediastinum with a short axis up to 1 cm appear stable.",
        "B0": "Bone structures within the study area are unremarkable.",
        "B1": "Vertebral body heights are maintained.",
        "A0": "Stable hypodense hepatic lesions suspicious for metastases and hepatomegaly are noted in the liver that is included in the field of view.",
        "A1": "A 28x17 mm lesion in the right adrenal gland, suspicious for metastasis, is stable.",
        "A2": "The left adrenal gland appears normal, with no lesions identified."
    },
    "Impressions": {
        "I0": "Stable mass encasing the bronchi of the left lower lobe.",
        "I1": "Stable pleuroparenchymal opacities with bronchial and pleural extension in the bilateral lower lobes, bronchial wall thickening, nonspecific ground glass densities, and right pleural effusion.",
        "I2": "Multiple hepatic mass lesions suspicious for metastases with associated hepatomegaly.",
        "I3": "Suspected right adrenal metastatic lesion.",
        "I4": "Stable mediastinal lymph nodes."
    }
}
**Output**: 
{
    "S1": ["Patent trachea and both main bronchi", "L0", "N", "N"],
    "S2": ["Bilateral lower lobe pleuroparenchymal opacities extending from central regions to pleura", "L1,I1", "Y", "Y"],
    "S3": ["Significant bronchial wall thickening", "L1,I1", "Y", "Y"],
    "S4": ["Mass-like appearance of left lower lobe bronchus", "L1,I0", "Y", "Y"],
    "S5": ["Stable pleural effusion in the right hemithorax", "L2,I1", "Y", "Y"],
    "S6": ["Normal heart contour and size", "H0", "N", "N"],
    "S7": ["Normal mediastinal main vascular structures", "H0", "N", "N"],
    "S8": ["Normal thoracic aorta diameter", "H1", "N", "N"],
    "S9": ["No pericardial effusion or thickening", "H2", "N", "N"],
    "S10": ["Normal thoracic esophagus caliber without tumoral thickening", "M0", "N", "N"],
    "S11": ["Stable mediastinal lymph nodes up to 1 cm", "M1,I4", "Y", "Y"],
    "S12": ["Unremarkable bone structures", "B0", "N", "N"],
    "S13": ["Maintained vertebral body heights", "B1", "N", "N"],
    "S14": ["Stable hypodense hepatic lesions", "A0,I2", "Y", "Y"],
    "S15": ["Hepatomegaly noted in the liver", "A0,I2", "Y", "Y"],
    "S16": ["Lesion in right adrenal gland, 28x17 mm", "A1,I3", "Y", "Y"],
    "S17": ["Normal left adrenal gland with no lesions", "A2", "N", "N"]
}
"""

SHORTENED_NAME_MAPPING = {'Lungs/Airways/Pleura': 'L', 
               'Heart/Vessels': 'H', 
               'Mediastinum/Hila': 'M', 
               'Chest wall/Axilla': 'C',
               'Lower neck': 'N', 
               'Bones': 'B',
               'Upper abdomen': 'A'
            }

LIMIT = 5

def load_original_reports(data_path, split, start_idx, end_idx, original_reports_dir):
    if split == 'val':
        csv_file_name = 'validation_reports_drop_dup.csv'
    elif split == 'train':
        csv_file_name = 'train_reports_drop_dup.csv'
    reports_df = pd.read_csv(os.path.join(data_path, csv_file_name))
    reports_df = reports_df.iloc[start_idx:end_idx] # slice the dataframe to process only a subset of reports

    full_reports = {}
    tokenized_reports = {}
    report_names = []
    for _, row in reports_df.iterrows():
        
        findings = row['Findings_EN']
        impressions = row['Impressions_EN']
        report_name = row['VolumeName']
        
        if pd.isna(findings):
            findings = 'Not Given'
        if pd.isna(impressions):
            impressions = 'Not Given'
        
        full_report = 'Findings: ' + findings + '\nImpressions: ' + impressions
        findings_sentences = sent_tokenize(findings)
        impressions_sentences = sent_tokenize(impressions)
    
        report_sentences_dict = {f'F{i}': sentence for i, sentence in enumerate(findings_sentences)}
        report_sentences_dict.update({f'I{i}': sentence for i, sentence in enumerate(impressions_sentences)})
        
        tokenized_reports[report_name] = str(report_sentences_dict)
        full_reports[report_name] = full_report
        report_names.append(report_name)
        
        with open(os.path.join(original_reports_dir, f'{report_name}.txt'), 'w') as f:
            f.write(full_report)
        with open(os.path.join(original_reports_dir, f'{report_name}_tokenized.txt'), 'w') as f:
            f.write(str(report_sentences_dict))
            
    return full_reports, tokenized_reports, report_names

def run_gpt(report, prompt, gpt_model="gpt41106", depth_limit=0, temp = 0.5):
    try:
        convo = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": report}
        ]
        response = client.chat.completions.create(
            model=gpt_model,
            messages=convo,
            max_tokens=3000,
            response_format = { "type": "json_object" },
            temperature=temp
        )
        if response is not None:
            return response
        else:
            if depth_limit <= LIMIT:
                time.sleep(2)
                return run_gpt(report, prompt, gpt_model, depth_limit+1)
            else:
                return None
    except Exception as e:
        if "content" in str(e):
            print(e)
            return None
        else:
            if "retry" in str(e):
                if depth_limit <= LIMIT:
                    time.sleep(2)
                    return run_gpt(report, prompt, gpt_model, depth_limit+1)
                else:
                    return None
            else:
                print(e)
                return run_gpt(report, prompt, gpt_model, depth_limit+1)

def estimate_cost(response):
    prompt_tokens = response.model_dump()["usage"]["prompt_tokens"]
    completion_tokens = response.model_dump()["usage"]["completion_tokens"]
    
    input_cost = 0.03
    output_cost = 0.06
    
    cost = (input_cost*prompt_tokens/1000 + output_cost*completion_tokens/1000)
    return cost

def check_key_structure(translation_json):
    if 'Findings' not in translation_json or 'Impressions' not in translation_json:
        return False
    for key in translation_json['Findings'].keys():
        if key not in SHORTENED_NAME_MAPPING:
            return False
    return True

def process_report(report_name, report, prompt, output_dir, temp, process_type, process_depth=0):
    start_process = time.time()
    response = run_gpt(report=report, prompt=prompt, temp=temp)
    
    if response is not None:
        content = response.model_dump()["choices"][0]["message"]["content"]
        
        if content is not None:
            # Check that json can be loaded correctly
            try:
                content_json = json.loads(content)
            except json.decoder.JSONDecodeError:
                if process_depth <= LIMIT:
                    time.sleep(2)
                    return process_report(report_name, report, prompt, output_dir, temp, process_type, process_depth + 1)
                else:
                    return 'Fail Due to Incorrect Json Content'
            
            # Check that keys are in the correct format
            if process_type == 'translation':
                if not check_key_structure(content_json):  # doesn't pass the key structure check
                    if process_depth <= LIMIT:
                        time.sleep(2)
                        return process_report(report_name, report, prompt, output_dir, temp, process_type, process_depth + 1)
                    else:
                        return 'Fail Due to Incorrect Key Structure'
            
            with open(os.path.join(output_dir, f"{report_name}.json"), "w") as f:
                json.dump(content_json, f)
            
            cost = estimate_cost(response)
            end_process = time.time()
            result = {
                "report_name": report_name,
                "report": content_json,
                "time": end_process - start_process,
                "cost": cost
            }
            return result
        else:
            if process_depth <= LIMIT:
                time.sleep(2)
                return process_report(report_name, report, prompt, output_dir, temp, process_type, process_depth + 1)
            else:
                return 'Fail Due to None Content'
    else:
        if process_depth <= LIMIT:
            time.sleep(2)
            return process_report(report_name, report, prompt, output_dir, temp, process_type, process_depth + 1)
        else:
            return 'Fail Due to None Response'

def restructure_for_extraction(translated_report):
    restructured_report = {'Findings': {}, 'Impressions': {}}
    
    for region, sentences in translated_report['Findings'].items():
        shortened_region_name = SHORTENED_NAME_MAPPING[region]
        for i, sentence in enumerate(sentences):
            restructured_report['Findings'][f'{shortened_region_name}{i}'] = sentence
    
    impressions_sentences = sent_tokenize(translated_report['Impressions'])
    restructured_report['Impressions'] = {f'I{j}': sentence for j, sentence in enumerate(impressions_sentences)}

    return str(restructured_report)

def save_stats(report_name, translation_result, extraction_result, stats_dir):
    if translation_result is None:
        translation_result = 'No Translation'
    if extraction_result is None:
        extraction_result = 'No Extraction'
    save_dict = {
    'translation': translation_result,
    'extraction': extraction_result
    }
    with open(os.path.join(stats_dir, f"{report_name}.json"), "w") as f:
        json.dump(save_dict, f, indent = 4)

def worker(report_name, tokenized_report, translated_reports_dir, extracted_reports_dir, stats_dir):
    translation_result = process_report(report_name, report=tokenized_report, prompt=TRANSLATION_PROMPT, output_dir=translated_reports_dir, temp=0.5, process_type = 'translation')
    if isinstance(translation_result, dict):
        restructured_report = restructure_for_extraction(translation_result['report'])
        extraction_result = process_report(report_name, report=restructured_report, prompt=EXTRACTION_PROPMT, output_dir=extracted_reports_dir, temp=0.2, process_type = 'extraction')
        save_stats(report_name, translation_result, extraction_result, stats_dir)
    else:
        save_stats(report_name, translation_result, None, stats_dir)

if __name__ == "__main__":
    # add argparse
    import argparse
    parser = argparse.ArgumentParser(description='Step 1: Translate/Structure AND Step 2: Extract reports')
    parser.add_argument('--split', type=str, default='val', help='Data split to process ("train" or "val")')
    parser.add_argument('--start_idx', default=None, type=int, help='Start index of the reports to process. Default is None (start from the beginning)')
    parser.add_argument('--end_idx', default=None, type=int, help='End index of the reports to process. Default is None (process until the last report)')
    parser.add_argument('--report_csv_dir', type=str, default='./data/', help='Directory containing the CSV files with reports to process')
    parser.add_argument('--output_dir', type=str, default='./outputs/round3', help='Directory to save outputs')
    parser.add_argument('--max_workers', type=int, default=8, help ='Maximum number of workers for multithreading')
    parser.add_argument('--api_key', type=str, default=None, help='Azure OpenAI API key. Defaults to environment variable AZURE_OPENAI_API_KEY')
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure OpenAI endpoint. Defaults to environment variable AZURE_OPENAI_ENDPOINT')
    args = parser.parse_args()
    
    split = args.split
    start_idx = args.start_idx
    end_idx = args.end_idx
    report_csv_dir = args.report_csv_dir
    output_dir = args.output_dir
    max_workers = args.max_workers
    
    # Use provided args or environment variables for Azure OpenAI
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not azure_endpoint:
        raise ValueError("Please provide Azure OpenAI API key and endpoint via arguments or environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")

    assert split in ['train', 'val'], "Split must be either 'train' or 'val'"
    print(f'Processing {split} reports from index {"0" if not start_idx else start_idx} to {"end" if not end_idx else end_idx}...')
    print(f'Using {max_workers} workers...')
    
    original_reports_dir = os.path.join(output_dir, split, 'original_reports')
    translated_reports_dir = os.path.join(output_dir, split, 'translated_reports')
    extracted_reports_dir = os.path.join(output_dir, split, 'extracted_reports')
    stats_dir = os.path.join(output_dir, split, 'stats')
    os.makedirs(original_reports_dir, exist_ok=True)
    os.makedirs(translated_reports_dir, exist_ok=True)
    os.makedirs(extracted_reports_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    load_reports_start = time.time()
    print("Loading original reports...")
    full_reports, tokenized_reports, report_names = load_original_reports(report_csv_dir, split, start_idx, end_idx, original_reports_dir)
    print(f'Finished loading original reports. Time: {time.time() - load_reports_start} seconds')
    
    client = AzureOpenAI(
        api_version="2023-05-15",
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )
    
    print(f"Processing {len(report_names)} reports...")
    start_time = time.time()
    
    with Pool(max_workers) as p:
        p.starmap(worker, [(report_name, tokenized_reports[report_name], translated_reports_dir, extracted_reports_dir, stats_dir) for report_name in report_names])
        
    end_time = time.time()
    print(f"Processing {len(report_names)} reports took: ", end_time - start_time, " seconds")
    print(f"Average time per report: {(end_time - start_time)/len(report_names)} seconds")
