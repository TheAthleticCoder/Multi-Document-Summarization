import argparse
import pandas as pd
import evaluate

# Load evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def evaluate_metrics(csv_file, gen_summary_column, ref_summary_column):
    # Load data from CSV file
    data = pd.read_csv(csv_file)

    # Compute ROUGE scores
    rouge_scores = rouge.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
        rouge_types=["rouge1", "rouge2", "rougeL"]
    )

    # Compute BLEU score
    bleu_scores = bleu.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
    )["bleu"]

    # Compute METEOR score
    meteor_scores = meteor.compute(
        predictions=data[gen_summary_column],
        references=data[ref_summary_column],
    )

    # Return all scores
    return dict(rouge_scores, **{"bleu": bleu_scores}, **meteor_scores)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate summary metrics")
    parser.add_argument("csv_file", help="CSV file containing data")
    parser.add_argument("--gen_summary_column", default="Generated Summary", help="Generated summary column name")
    parser.add_argument("--ref_summary_column", default="Reference Summary", help="Reference summary column name")

    args = parser.parse_args()

    # Evaluate metrics and print results
    metrics = evaluate_metrics(
        args.csv_file,
        args.gen_summary_column,
        args.ref_summary_column
    )

    print("Metrics:", metrics)

if __name__ == "__main__":
    main()