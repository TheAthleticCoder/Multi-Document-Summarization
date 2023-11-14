#!/bin/bash
# Change the name to what you want.
#SBATCH --job-name=bart_ms
# Replace with irel if you want to use irel account. Refrain from using irel if not necessary.
#SBATCH -A irel
# Specify the number of GPUs you need.
#SBATCH -c 20
# Specify the number of GPUs you need. It's zero in this script. Replace with 1 to 4, as you want.
#SBATCH -G 2
# Outputs from your job would get written to the file specified below.
#SBATCH -o bart_ms.out
# Time of the Job. The tie specified below if 4 days.
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
# Memory per CPU that you need (in MBs)
#SBATCH --mem-per-cpu=2999

# Run Jupyter Notebook
python bart_ms.py --model_name "sshleifer/distilbart-cnn-6-6" --topk 5000 --output_file "bart_multisci_1.csv"

