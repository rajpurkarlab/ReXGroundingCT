# ReXGroundingCT

Data processing pipeline for preparing the CT-RATE dataset for annotation with Redbrick.

---

## Pipeline Steps

### Step 1: Process Radiology Reports
The first step is to process the radiology reports by running `src/process_reports.py`. This script translates and structures the reports.

**Arguments:**
- `--split`: Data split to process ("train" or "val"). Default: `val`.
- `--start_idx`: Start index of the reports to process. Default: `None`.
- `--end_idx`: End index of the reports to process. Default: `None`.
- `--report_csv_dir`: Directory containing the CSV files with reports. Default: `./data/`.
- `--output_dir`: Directory to save outputs. Default: `./outputs/round3`.
- `--max_workers`: Maximum number of workers for multithreading. Default: `8`.
- `--api_key`: Azure OpenAI API key. Can also be set via `AZURE_OPENAI_API_KEY` environment variable.
- `--azure_endpoint`: Azure OpenAI endpoint. Can also be set via `AZURE_OPENAI_ENDPOINT` environment variable.

**Example:**
```bash
python src/process_reports.py --split train --start_idx 0 --end_idx 100 --api_key YOUR_API_KEY --azure_endpoint YOUR_ENDPOINT
```

### Step 2: Check for Errors
After processing reports, check for any errors using `utils/check_for_errors_in_process_reports.py`. This script identifies and prints any reports that failed.

**Configuration:**
You need to manually edit the following variables inside the script:
- `round_num`: The output round to check (e.g., 'round3').
- `split`: The data split to check ('train' or 'val').

**Usage:**
```bash
python utils/check_for_errors_in_process_reports.py
```
You can then manually fix or re-process any failed reports.

### Step 3: Organize Positive Findings
Run `src/organize_positive_findings.py` to structure the findings for categorization.

**Configuration:**
You need to manually edit the following variable inside the script:
- `round_num`: The output round to process (e.g., 'round3').

**Usage:**
```bash
python src/organize_positive_findings.py
```

### Step 4: Categorize Findings
Categorize positive findings by running `src/categorize.py`.

**Arguments:**
- `--split`: Data split to process ("train" or "val"). Default: `train`.
- `--extracted_findings_dir`: Directory with extracted findings. Default: `./outputs/round3`.
- `--max_workers`: Maximum number of workers. Default: `1`.
- `--api_key`: Azure OpenAI API key. Can also be set via `AZURE_OPENAI_API_KEY` environment variable.
- `--azure_endpoint`: Azure OpenAI endpoint. Can also be set via `AZURE_OPENAI_ENDPOINT` environment variable.

**Example:**
```bash
python src/categorize.py --split train --api_key YOUR_API_KEY --azure_endpoint YOUR_ENDPOINT
```

### Step 5: Filter Scans by Category
After categorization, filter the scans to select which ones to annotate using `src/filter_scans_by_category.py`. This also generates plots to visualize the distribution.

**Configuration:**
You need to manually edit the following variable inside the script:
- `round_num`: The output round to process (e.g., 'round6').

**Usage:**
```bash
python src/filter_scans_by_category.py
```

### Step 6: Format Data for Google Cloud Storage
Format the scans for upload to a Google Cloud Storage bucket using `src/format_data_for_upload.py`.

**Configuration:**
You need to manually edit the following variables inside the script:
- `round_num`: The output round being processed.
- `src_path`: Path to the directory containing the categorized data.
- `dest`: Destination directory to save the formatted scans.

**Usage:**
```bash
python src/format_data_for_upload.py
```

### Step 7: Upload Scans to Google Cloud Storage
Follow the instructions in the [Redbrick documentation](https://docs.redbrickai.com/importing-data/import-cloud-data/configuring-gcs) to upload the formatted scans from the destination directory to a Google Cloud Storage bucket.

### Step 8: Prepare JSON for Redbrick Upload
Finally, generate the `.json` file for Redbrick using `src/format_json_for_redbrick_upload.py`. This file loads the scans from the Google Cloud bucket into the annotation platform.

**Configuration:**
You need to manually edit the following variables inside the script:
- `round_num`: The output round being processed.
- `scans_root_path`: The path to the directory where the formatted scans are stored.

**Usage:**
```bash
python src/format_json_for_redbrick_upload.py
```
This JSON file will contain the metadata about the scans and their locations in the cloud storage bucket.

---
## Directory Structure
- **data**: Directory containing the CSV files with reports to be processed.
- **sbatch**: Batch scripts for job submission (optional).
- **src**: Source code for processing the reports.
- **utils**: Utility scripts like error checking.
- **outputs**: Directory where the output report files (original, translated, categorized, etc.) as well as any plots or json files are saved.
- **scans**: Directory where the formatted scans are stored for upload to the Google Cloud storage bucket. You must specify where this directory is located.
---
