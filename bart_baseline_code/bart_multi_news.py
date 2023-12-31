from datasets import load_dataset
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from torch import cuda

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate summaries using BART model")
parser.add_argument("--model_name", type=str, default="sshleifer/distilbart-cnn-6-6", help="Model checkpoint name")
parser.add_argument("--topk", type=int, default=1000, help="Number of sentences to consider from the test set")
parser.add_argument("--output_file", type=str, default="generated_summaries.csv", help="Output file name for generated summaries")
args = parser.parse_args()

dataset = load_dataset("multi_news")
#print(f"Features: {dataset['train'].column_names}")

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# device = f'cuda:1' if cuda.is_available() else 'cpu'
print(device)

model_ckpt = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BartForConditionalGeneration.from_pretrained(model_ckpt)
model.to(device)
print(model)

def convert_examples_to_features(example_batch, tokenizer):
    input_encodings = tokenizer(
        example_batch["document"],
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors="pt"
    )

    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(
            example_batch["summary"],
            max_length=256,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

    return {
        "input_ids": input_encodings["input_ids"],
        "attention_mask": input_encodings["attention_mask"],
        "labels": target_encodings["input_ids"]
    }

# Mapping features to the dataset and set the format
dataset_tf = dataset.map(
    lambda x: convert_examples_to_features(x, tokenizer),
    batched=True
)
columns = ["input_ids", "labels", "attention_mask"]
dataset_tf.set_format(type="torch", columns=columns)

seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir='bart-multi-news',
    num_train_epochs=3,
    warmup_steps=500,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    logging_steps=10,
    push_to_hub=False,
    evaluation_strategy='steps',
    eval_steps=500,
    save_steps=1e6,
    gradient_accumulation_steps=16
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=seq2seq_data_collator,
    train_dataset=dataset_tf["train"],
    eval_dataset=dataset_tf["validation"]
)

# Training
trainer.train()

# Consider the top 'topk' sentences from the test set
top_k = args.topk
test_dataset = dataset["test"].select(range(top_k))

# Generate summaries for the test dataset
generated_summaries = []
for idx, example in tqdm(enumerate(test_dataset)):
    input_ids = tokenizer(
        example["document"],
        max_length=1024,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    summaries = model.generate(
        input_ids=input_ids['input_ids'],
        attention_mask=input_ids['attention_mask'],
        max_length=256
    )

    decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
    generated_summaries.append(decoded_summaries[0])

# Store the document, reference summary, and generated summary in a CSV file
output_df = pd.DataFrame({
    "Document": [example["document"] for example in test_dataset],
    "Reference Summary": [example["summary"] for example in test_dataset],
    "Generated Summary": generated_summaries
})

# Save the results to a CSV file
output_file = args.output_file
output_df.to_csv(output_file, index=False)

print(f"Summaries for the top {top_k} sentences have been generated and saved to {output_file}.")
