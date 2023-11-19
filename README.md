# Multi-Document Summarization

Shubh Agarwal \
Harshit Gupta \
Shikhar Saxena

## Project Description

Our project addresses the challenge of multi-document summarization with Large Language Models (LLMs), which are constrained by token length limitations. We propose a novel approach that combines the strengths of LLMs and Maximal Marginal Relevance (MMR). MMR helps select the most relevant sentences from multiple documents, and then LLMs generate concise summaries from these sentences. This method overcomes LLMs' token limit challenge and improves summary quality by emphasizing important content. Our tests show this approach effectively produces informative, coherent, and concise summaries.

## File Structure

```
├── bart_baseline_code
│   ├── bart_multi_news.py
│   ├── bart_multi_x_science.py
│   └── README.md
├── baseline_metrics
│   └── baseline_scores.ipynb
├── clustering
│   ├── bert_embeddings.ipynb
│   ├── evaluation.ipynb
│   ├── tf
│   │   └── tf.ipynb
│   ├── tf_idf_method_1
│   │   ├── details.md
│   │   └── tf_idf_method_1.ipynb
│   └── tf_idf_method_2
│       ├── details.md
│       └── tf_idf_method_2.ipynb
├── dataset_handling
│   ├── multi_news_dh.ipynb
│   └── multi_x_sci_dh.ipynb
├── gen_eval.py
├── llm_mmr_prompt
│   ├── multi_news.py
│   ├── multi_x_sci.py
│   └── README.md
├── mmr
│   └── mmr.ipynb
├── README.md
└── wandb_setup.ipynb
```

## Data

To access the full data, download from: https://drive.google.com/drive/folders/1_Rkr3CczybVh6YfzSrrY7YjF-4f4Kw4E?usp=sharing

## Data Handling

We have extensively used 'WANDB' to store our datasets and our preprocessed data. \
https://wandb.ai/ire-shshsh/mdes/artifacts/dataset/multi_x_science_modified_sample/v0

## BART Baseline Code

This script uses the BART model from the Hugging Face Transformers library to generate summaries of news articles. The script reads in a dataset of news articles, tokenizes the text, and feeds it into the BART model to generate a summary. The summaries are then saved to a CSV file.

### Usage

Run the script with the following command:

```sh
python bart_multi_news.py --input_file <path_to_input_file> --output_file <path_to_output_file>
```

## Baseline Metrics

It is used to calculate and display the baseline scores for the project. It uses the evaluate module to calculate various evaluation metrics like BLEU, METEOR, and ROUGE scores. These scores are used to evaluate the performance of the baseline models.

## Clustering

This directory contains scripts and notebooks for clustering the dataset. The clustering is done using different methods and techniques.

### Contents

- `bert_embeddings.ipynb`: This notebook contains the code for generating BERT embeddings for the dataset.

- `evaluation.ipynb`: This notebook is used to evaluate the performance of the clustering methods.

- `tf/`: This directory contains a notebook `tf.ipynb` for performing clustering using TensorFlow.

- `tf_idf_method_1/`: This directory contains a notebook `tf_idf_method_1.ipynb` and a detailed explanation `details.md` for performing clustering using the first method of TF-IDF.

- `tf_idf_method_2/`: This directory contains a notebook `tf_idf_method_2.ipynb` and a detailed explanation `details.md` for performing clustering using the second method of TF-IDF.

## Dataset Handling

This directory contains notebooks for the preprocessing of the datasets. We also had sampled the dataset so as to work with smaller datasets during development.

## LLM MMR Prompt

This directory contains scripts for generating summaries using a language model. The scripts use the Hugging Face Transformers library to load pre-trained models such as Mistral and generate summaries based on the input data.

### Usage

To run the scripts, use the following command:

```sh
python <script_name> --model_id <model_id> --file_path <input_file_path> --new_file_save_path <output_file_path>
```

Replace <script_name> with either multi_news.py or multi_x_sci.py, <model_id> with the ID of the Hugging Face model to use (for example, mistralai/Mistral-7B-Instruct-v0.1), <input_file_path> with the path to the input CSV file, and <output_file_path> with the path where the output CSV file should be saved.

## MMR

This notebook contains the implementation of the MMR algorithm. It includes functions for calculating similarity scores, selecting sentences based on MMR, and extracting the most important sentences for the summary.

## W&B Setup

This notebook contains the code for setting up W&B for the project. It is used to log the artifacts to W&B.
