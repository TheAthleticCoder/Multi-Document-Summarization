# Install required packages
# !pip install evaluate
# !pip install rouge-score
# !conda install -c conda-forge rouge-score 

import argparse
import pandas as pd
import evaluate

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def evaluate_metrics(csv_file, gen_summary_column, ref_summary_column):
    data = pd.read_csv(csv_file)

    rouge_scores = rouge.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    bleu_scores = bleu.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
    )["bleu"]

    meteor_scores = meteor.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
    )

    return dict(rouge_scores, **{"bleu": bleu_scores}, **meteor_scores)

def main():
    parser = argparse.ArgumentParser(description="Evaluate summary metrics")
    parser.add_argument("csv_file", help="CSV file containing data")
    parser.add_argument("--gen_summary_column", default="Generated Summary", help="Generated summary column name")
    parser.add_argument("--ref_summary_column", default="Reference Summary", help="Reference summary column name")

    args = parser.parse_args()

    metrics = evaluate_metrics(
        args.csv_file,
        args.gen_summary_column,
        args.ref_summary_column
    )

    print("Metrics:", metrics)

if __name__ == "__main__":
    main()

#python evaluate_script.py data/bart_mn_1.csv --gen_summary_column "Generated_Summary" --ref_summary_column "Reference_Summary"
