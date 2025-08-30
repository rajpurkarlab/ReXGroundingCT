from openai import AzureOpenAI
import time
import json
import os
from tqdm import tqdm

CATEGORIZATION_PROMPT = """
You are an expert radiologist specializing in chest CT scan interpretation. Your task is to categorize positive findings from radiology reports according to a specific schema. The schema consists of 12 parent categories (including "Other") and subcategories within each parent category (except for the "Other" parent category).

Schema:
1. Typically non-focal lung/airway/pleural abnormalities
   a. Bronchial wall thickening
   b. Bronchiectasis
   c. Emphysema (including Centrilobular, Paraseptal, Bullous)
   d. Septal thickening (including Interlobular, Reticulation)
   e. Micronodules (including, Centrilobular, Tree-in-bud, Perilymphatic)
   f. Other
2. Typically focal lung/airway/pleural opacities
   a. Linear (including subsegmental atelectasis, scarring, fibrosis)
   b. Atelectasis, consolidation 
   c. Groundglass opacity
   d. Pulmonary nodules/masses
   e. Pleural effusion or thickening
   f. Honeycombing
   g. Pneumothorax
   h. Other
3. Non-pulmonary lesions
   a. Lymphadenopathy lesions
   b. Liver lesions
   c. Gallbladder lesions
   d. Renal/kidneys, collecting system, and ureters lesions
   e. Spleen lesions
   f. Adrenal lesions
   g. Pancreas lesions
   h. Thyroid lesions
   i. Skin/subcutaneous lesions
   j. Bone/Osseous structures lesions
   k. Other lesions
4. Bones (non-lesion)
   a. Fractures
   b. Degenerative joint disease, degenerative disc disease, arthritis
   c. Spinal curvature abnormalities: kyphosis, scoliosis
   d. Other
5. Stones/organ calcifications (non-lesion)
   a. Nephroliths, choleliths
   b. Granulomas
   c. Other
6. Hollow viscera abnormalities
   a. Hiatus hernia
   b. Wall thickening
   c. Dilated
   d. Diverticulum (including diverticulosis)
   e. Other
7. Skin/subcutaneous (non-lesion)
   a. Skin thickening
   b. Stranding
   c. Abdominal wall hernia: ventral, umbilical, inguinal
   d. Gynecomastia
   e. Other
8. Cardiovascular
   a. Atherosclerosis (including coronary, non-coronary)
   b. Vessel aneurysm, ectasia, enlargement
   c. Vessel occlusion or stenosis
   d. Cardiac chamber enlargement
   e. Valvular calcification
   f. Pericardial effusion
   g. Other
9. Body composition
   a. Visceral fat
   b. Superficial subcutaneous fat
   c. Skeletal muscle
   d. Osteoporosis
   e. Hepatic steatosis
   f. Other
10. Diffuse/whole organ 
    a. Organomegaly (including splenomegaly, multinodular goiter, thyromegaly, lung hyperinflation)
    b. Atrophy
    c. Other
11. Device
    a. Elongated (including catheter, pacemaker/defibrillator, spinal stimulator)
    b. Surgical clips 
    c. Other
12. Other

**OBJECTIVE**:
You will be given a list of positive findings from a CT Scan report. Your task is to assign each finding to the most appropriate category using the format "Xa", where X is the parent category number (1-12) and 'a' is the subcategory letter. These parent categories and subcategories are mutually exclusive and exhaustive, so you must assign one of the categories in the schema to each given positive finding.

**RULES**:
1. You should first think about what parent category the finding belongs to and then assign the most appropriate subcategory within that parent category.
2. If a finding doesn't fit into any parent category, use "12" for the general "Other" category.
3. If a finding doesn't fit into any specific subcategory, use the "Other" subcategory of the most relevant parent category. 
4. YOU MUST PLACE EACH POSITIVE FINDING INTO ONE OF THESE CATEGORIES. YOU WILL BE PENALIZED IF YOU ASSIGN A FINDING TO A NON-EXISTENT CATEGORY OR IF YOU LEAVE A FINDING UNASSIGNED.
5. ONLY USE THE CATEGORIES PROVIDED IN THE SCHEMA. DO NOT CREATE NEW CATEGORIES.
6. Note: Bone lesions should be categorized under 3j (Non-pulmonary lesions - Bone/Osseous structures), not under category 4.

**Input Format**:
The input will be in JSON format, structured as follows: A python dictionary where keys are "P1", "P2", etc., representing the ID of each positive finding, and the values are the text of the positive findings.

**Output Format**:
The output should be in JSON format, structured as follows: A python dictionary with the same keys as the input, but the values should be the assigned categories (e.g., "1a", "2c", "3f", "12", etc.) for each positive finding.

BE SURE TO FOLLOW ALL RULES METICULOUSLY. Think through all rules and steps before answering.

**Follow these examples**:
Input 1:
{
    "P1": "Scarring at apical level of right lung, in middle lobe, anterior-posterior segments of right upper lobe, and superior segment of lower lobe",
    "P2": "Scarring at level of minor fissure and in lingular segment",
    "P3": "Subpleural calcified nodules, 2-3 mm in diameter, along interlobular fissure in basal portion of left lower lobe",
    "P4": "Calcific atheromatous plaques in aortic arch, subclavian artery, and coronary arteries",
    "P5": "Mediastinal and bilateral hilar lymph nodes, largest on right measuring approximately 15x11 mm, some with partial calcification",
    "P6": "Mild hiatal hernia",
    "P7": "Degenerative changes in bone structure",
    "P8": "Diffuse idiopathic skeletal hyperostosis (DISH)",
    "P9": "Mild scoliosis with convexity to the left in thoracic spine",
    "P10": "Liver with decreased density consistent with steatosis",
    "P11": "Prominent dense formation within gallbladder consistent with cholelithiasis",
    "P12": "Nodular density adjacent to spleen compatible with accessory spleen"
}

Output 1:
{
    "P1": "2a",
    "P2": "2a",
    "P3": "2d",
    "P4": "8a",
    "P5": "3a",
    "P6": "6a",
    "P7": "4b",
    "P8": "4b",
    "P9": "4c",
    "P10": "9e",
    "P11": "5a",
    "P12": "3e"
}

Input 2:
{
    "P1": "Subsegmental and band-like atelectasis in the medial segment of the right middle lobe, inferior lingular segment of the left upper lobe, and basal segments of both lower lobes",
    "P2": "Emphysematous changes in the upper lobes of both lungs",
    "P3": "Tortuous and elongated thoracic aorta",
    "P4": "Increased pulmonary trunk diameter at 33 mm",
    "P5": "Small pericardial effusion",
    "P6": "Calcific atheroma plaques in supraaortic branches of thoracic aorta and coronary arteries",
    "P7": "Calcific plaques in abdominal aorta, its visceral branches, and proximal iliac arteries",
    "P8": "Small ventral hernia",
    "P9": "Displaced and impacted multipart fracture in right humeral head",
    "P10": "Thoracolumbar S-shaped scoliosis",
    "P11": "Osteophyte formations with bridging at right anterolateral vertebral corners",
    "P12": "Cholelithiasis with a 12 mm calculus within gallbladder lumen",
    "P13": "Two millimetric calculi in lower pole of left kidney",
    "P14": "Two cortical-parapelvic cysts in anterior midsection of right kidney, largest measuring 4.5 cm in diameter",
    "P15": "Nodular thickening in both adrenal glands",
    "P16": "Diffuse lytic bone lesions suggestive of multiple myeloma involvement throughout the visualized bones"
}

Output 2:
{
    "P1": "2b",
    "P2": "1c",
    "P3": "8b",
    "P4": "8b",
    "P5": "8e",
    "P6": "8a",
    "P7": "8a",
    "P8": "7c",
    "P9": "4a",
    "P10": "4c",
    "P11": "4b",
    "P12": "5a",
    "P13": "5a",
    "P14": "3d",
    "P15": "3f",
    "P16": "3j"
}

Please categorize the given findings according to this schema and format.
"""

LIMIT = 10

def run_gpt(report, prompt, gpt_model="gpt4omini20240718", depth_limit=0, temp = 0.5):
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

POSSIBLE_CATEGORIES = [
    '1a', '1b', '1c', '1d', '1e', '1f',
    '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h',
    '3a', '3b', '3c', '3d', '3e', '3f', '3g', '3h', '3i', '3j', '3k',
    '4a', '4b', '4c', '4d',
    '5a', '5b', '5c',
    '6a', '6b', '6c', '6d', '6e',
    '7a', '7b', '7c', '7d', '7e',
    '8a', '8b', '8c', '8d', '8e', '8f', '8g',
    '9a', '9b', '9c', '9d', '9e', '9f',
    '10a', '10b', '10c',
    '11a', '11b', '11c',
    '12'
]

def check_categories(categorized_json):
    for k, v in categorized_json.items():
        if v not in POSSIBLE_CATEGORIES:
            print(f"ERROR: {v} is not a valid category for {k}")
            return False
    return True

def process_report(report_name, report, prompt, output_dir, temp, process_depth=0):
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
                    return process_report(report_name, report, prompt, output_dir, temp, process_depth + 1)
                else:
                    return 'Fail Due to Incorrect Json Content'
            
            # Check that keys are in the correct format
            if not check_categories(content_json):
                if process_depth <= LIMIT:
                    time.sleep(2)
                    return process_report(report_name, report, prompt, output_dir, temp, process_depth + 1)
                else:
                    return 'Fail Due to Incorrect Value Categories'
            
            with open(os.path.join(output_dir, f"{report_name}.json"), "w") as f:
                json.dump(content_json, f)
            
            end_process = time.time()
            result = {
                "report_name": report_name,
                "report": content_json,
                "time": end_process - start_process,
            }
            return result
        else:
            if process_depth <= LIMIT:
                time.sleep(2)
                return process_report(report_name, report, prompt, output_dir, temp, process_depth + 1)
            else:
                return 'Fail Due to None Content'
    else:
        if process_depth <= LIMIT:
            time.sleep(2)
            return process_report(report_name, report, prompt, output_dir, temp, process_depth + 1)
        else:
            return 'Fail Due to None Response'

def save_stats(report_name, categorization_result, stats_dir):
    if categorization_result is None:
        categorization_result = 'No Categorization'
    save_dict = {
        'categorization': categorization_result
    }
    with open(os.path.join(stats_dir, f"{report_name}.json"), "w") as f:
        json.dump(save_dict, f, indent = 4)

def worker(report_name, report, categorized_findings_dir, stats_dir):
    if report == {}: # No positive findings were extracted
        categorization_result = 'No Positive Findings'
        with open(os.path.join(stats_dir, 'track_empty_reports.txt'), 'a') as f:
            f.write(f"{report_name}\n")
    else:
        categorization_result = process_report(report_name, report=str(report), prompt=CATEGORIZATION_PROMPT, output_dir=categorized_findings_dir, temp=0.2)
    save_stats(report_name, categorization_result, stats_dir)
    return categorization_result

if __name__ == "__main__":
    # add argparse
    import argparse
    parser = argparse.ArgumentParser(description='Step 3: Categorize Findings')
    parser.add_argument('--split', type=str, default='train', help='Data split to process ("train" or "val")')
    parser.add_argument('--extracted_findings_dir', type=str, default='./outputs/round3', help='Directory containing the extracted findings (organized by positive/negative findings)')
    parser.add_argument('--max_workers', type=int, default=1, help ='Maximum number of workers for multithreading')
    parser.add_argument('--api_key', type=str, default=None, help='Azure OpenAI API key. Defaults to environment variable AZURE_OPENAI_API_KEY')
    parser.add_argument('--azure_endpoint', type=str, default=None, help='Azure OpenAI endpoint. Defaults to environment variable AZURE_OPENAI_ENDPOINT')
    args = parser.parse_args()
    
    split = args.split
    extracted_findings_dir = args.extracted_findings_dir
    max_workers = args.max_workers
    
    assert split in ['train', 'val'], "Split must be either 'train' or 'val'"
    
    print(f'Categorizing {split}...')
    print(f'Using {max_workers} workers...')
    
    positive_finding_reports_dir = os.path.join(extracted_findings_dir, split, 'positive_finding_reports')
    assert os.path.isdir(positive_finding_reports_dir), f"{positive_finding_reports_dir} is not a directory. Need to run organize_positive_findings.py first!"
    
    categorized_findings_dir = os.path.join(extracted_findings_dir, split, 'categorized_findings', 'categorization')
    stats_dir = os.path.join(extracted_findings_dir, split, 'categorized_findings', 'stats')
    os.makedirs(categorized_findings_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Use provided args or environment variables for Azure OpenAI
    api_key = args.api_key or os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = args.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not azure_endpoint:
        raise ValueError("Please provide Azure OpenAI API key and endpoint via arguments or environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT)")

    client = AzureOpenAI(
        api_version="2023-05-15",
        api_key=api_key,
        azure_endpoint=azure_endpoint
    )

    start_time = time.time()
    
    print("Categorizing positive findings...")

    positive_finding_report_names = sorted(
        [x.split('.')[0] for x in os.listdir(positive_finding_reports_dir) if x.endswith('.json')],
        key=lambda name: (
            int(name.split('_')[1]),
            str(name.split('_')[2])
        )
    )
    print(f'Processing {len(positive_finding_report_names)} reports...')
    for positive_finding_report_name in positive_finding_report_names:
        print()
        print()
        print(f'Processing {positive_finding_report_name}...')
        with open(os.path.join(positive_finding_reports_dir, f'{positive_finding_report_name}.json'), 'r') as f:
            positive_finding_report = json.load(f)
            print(positive_finding_report)
            categorization_result = worker(positive_finding_report_name, positive_finding_report, categorized_findings_dir, stats_dir)
            print(categorization_result)
            
    end_time = time.time()
    print(f'Processing {len(positive_finding_report_names)} reports took {end_time - start_time} seconds')