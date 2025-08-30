import pandas as pd
import os
import nibabel as nib
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def flip_orientation(ct_arr):
    # Flip RAS to LPS
    return np.flip(np.flip(ct_arr, axis=0), axis=1)

def process_volume(file_name, src_metadata_lookup, dest):
    new_file_path = os.path.join(dest, file_name)
    if os.path.exists(new_file_path):
        print(f"file {new_file_path} already exsists")
        
    reconstruction_name = file_name.split('.')[0]
    patient_folder_name = '_'.join(reconstruction_name.split('_')[:2])
    study_folder_name = '_'.join(reconstruction_name.split('_')[:3])
    
    src_key = 'train' if 'train' in file_name else 'valid'
    src, metadata = src_metadata_lookup[src_key]
    
    file_path = os.path.join(src, patient_folder_name, study_folder_name, file_name)
    
    # Load the NIfTI file
    nii_file = nib.load(file_path)
    img_data = nii_file.get_fdata()
    affine = nii_file.affine
    
    # Rescale using metadata
    row = metadata[metadata['VolumeName'] == file_name]
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    img_data = slope * img_data + intercept

    # Flip the orientation
    img_data = flip_orientation(img_data)
    
    # Save the new NIfTI file
    new_nii = nib.Nifti1Image(img_data, affine)
    nib.save(new_nii, new_file_path)

def process_volume_wrapper(args):
    return process_volume(*args)

def main():
    # CHANGE THE FOLLOWING DEPENDING ON THE ROUND THAT YOU ARE PROCESSING
    round_num = 'round6'
    src_path = '/home/mob999/ReportGrounding/data/ct_rate_categorization'
    dest = f'/home/mob999/rajpurkarlab/datasets/ReXGroundingCT/round6/'  # AFTER RUNNING THIS SCRIPT, UPLOAD THIS DIRECTORY TO GOOGLE CLOUD STORAGE BUCKET FOR REDBRICK API

    ##########################################################
    print(f"Creating upload files for {round_num}...")
    os.makedirs(dest, exist_ok=True)

    # Load scans to annotate
    scans_to_annotate_path = f'{src_path}/{round_num}/{round_num}_scans_to_annotate.xlsx'
    if not os.path.exists(scans_to_annotate_path):
        raise FileNotFoundError(f"ERROR: {scans_to_annotate_path} does not exist. Please run 'python src/filter_scans_by_category.py' first.")
    scans_to_annotate = pd.read_excel(scans_to_annotate_path)

    train_src = '/home/mob999/rajpurkarlab/CT-RATE/dataset/train/'
    train_metadata = pd.read_csv('/home/mob999/rajpurkarlab/CT-RATE/dataset/metadata/train_metadata.csv')
    train_volumes = pd.read_csv('./data/train_reports_drop_dup_with_volumes.csv')

    valid_src = '/home/mob999/rajpurkarlab/CT-RATE/dataset/valid/'
    valid_metadata = pd.read_csv('/home/mob999/rajpurkarlab/CT-RATE/dataset/metadata/validation_metadata.csv')
    valid_volumes = pd.read_csv('./data/validation_reports_drop_dup_with_volumes.csv')

    all_volumes = pd.concat([train_volumes, valid_volumes], ignore_index=True)
    merged_df = pd.merge(scans_to_annotate, all_volumes, left_on="Unnamed: 0", right_on="VolumeName", how="left")
    
    if merged_df["VolumeToAnnotate"].isnull().any():
        print("Some volumes were not found in the train/validation CSV files.")

    volume_to_annotate_list = merged_df["VolumeToAnnotate"].tolist()

    # lookup dictionary for source metadata
    src_metadata_lookup = {
        'train': (train_src, train_metadata),
        'valid': (valid_src, valid_metadata)
    }

    # num_cores = int(os.getenv('SLURM_CPUS_PER_TASK')) if os.getenv('SLURM_CPUS_PER_TASK') else cpu_count()
    num_scans = len(volume_to_annotate_list)
    num_cores = 8

    print(f"Using {num_cores} cpus to process {num_scans} scans...")

    args = [(file_name, src_metadata_lookup, dest) for file_name in volume_to_annotate_list]
    with Pool(num_cores) as pool:
        for _ in tqdm(pool.imap_unordered(process_volume_wrapper, args), total=len(args)):
            pass

if __name__ == "__main__":
    main()
