# ct-rate-grounding

Data processing pipeline for preparing the CT-RATE dataset for annotation with Redbrick.

---

## Pipeline Steps

### Step 1: Process Radiology Reports
The first step is to process the radiology reports by running `process_reports.py`.

Be sure to adjust the `--split`, `--start_idx`, `--end_idx`, `--output_dir`, and `--end_point` arguments depending on which reports you would like to process.

```bash
python src/process_reports.py
```

### Step 2: Check for Errors
After processing reports, check if there were any errors using `check_for_errors_in_process_reports.py`. This script identifies and prints out any reports that failed during the process.

Be sure to adjust the `round_num` and `split` variables accordingly.

```bash
python utils/check_for_errors_in_process_reports.py
```

Once you run this script, you can manually fix or process any reports that couldn't be processed.

### Step 3: Organize Positive Findings
Run `organize_positive_findings.py` to structure the findings that will be used for categorization.

Be sure to adjust the `round_num` variable accordingly.

```bash
python src/organize_positive_findings.py
```

### Step 4: Categorize Findings
Categorize positive findings by running the `categorize.py` script.

Be sure to adjust the `--split` and `--extracted_findings_dir` depending on which reports you are processing.

```bash
python src/categorize.py
```

### Step 5: Filter Scans by Category
After categorization, filter the scans to select the ones that should be annotated. This step will also generate plots to visualize the distribution of filtered scans.

Be sure to adjust the `round_num` variable accordingly.

```bash
python src/filter_scans_by_category.py
```

### Step 6: Format Data for Google Cloud Storage
The next step is to format the scans for upload to a Google Cloud Storage bucket, which will be accessed by the annotation platform, Redbrick. The formatted scans will be saved in the `scans` directory.

Be sure to adjust the `round_num` variable and directory to save scans accordingly.

```bash
python src/format_data_for_upload.py
```

### Step 7: Upload Scans to Google Cloud Storage
Follow the instructions in the [Redbrick documentation](https://docs.redbrickai.com/importing-data/import-cloud-data/configuring-gcs) to upload the formatted scans to a Google Cloud Storage bucket.

### Step 8: Prepare JSON for Redbrick Upload
Finally, generate the `.json` file that will be uploaded to Redbrick, which will load the scans from the Google Cloud storage bucket.

Be sure to adjust the `round_num` variable and directory where the scans are stored accordingly.

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
