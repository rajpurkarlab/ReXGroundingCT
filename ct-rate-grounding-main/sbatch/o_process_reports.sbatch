#!/bin/bash
#SBATCH --job-name="o_process_reports"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhi1@stanford.edu
#SBATCH --output=o_process_reports.out
#SBATCH --error=o_process_reports.err
#SBATCH -t 0-12:00:00
#SBATCH -p short
#SBATCH -c 10
#SBATCH --mem=16G

python src/process_reports.py --split val --start_idx 1000 --end_point obanerjee
