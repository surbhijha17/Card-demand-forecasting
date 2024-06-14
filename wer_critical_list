pip install transformers datasets torch

import json
from datasets import Dataset, DatasetDict

def load_data(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

def convert_to_squad_format(data):
    squad_format = {"data": []}
    for item in data:
        paragraph = {
            "context": item["context"],
            "qas": [
                {
                    "id": str(hash(item["context"] + item["question"])),
                    "question": item["question"],
                    "answers": [
                        {
                            "text": json.dumps(item["answer"]),
                            "answer_start": item["context"].find(json.dumps(item["answer"]))
                        }
                    ],
                    "is_impossible": False if item["answer"] else True
                }
            ]
        }
        squad_format["data"].append({"title": "dataset", "paragraphs": [paragraph]})
    return squad_format

def create_dataset_dict(squad_data):
    dataset_dict = {"train": [], "validation": []}
    for i, item in enumerate(squad_data["data"]):
        if i % 5 == 0:  # Use 20% of the data for validation
            dataset_dict["validation"].append(item)
        else:
            dataset_dict["train"].append(item)
    return DatasetDict({
        "train": Dataset.from_dict({"data": dataset_dict["train"]}),
        "validation": Dataset.from_dict({"data": dataset_dict["validation"]}),
    })

# Load and prepare data
json_data = load_data('path_to_your_json_file.json')
squad_data = convert_to_squad_format(json_data)
datasets = create_dataset_dict(squad_data)
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering, Trainer, TrainingArguments
import torch

# Load tokenizer and model
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForQuestionAnswering.from_pretrained('roberta-base')

# Tokenize data
def preprocess_function(examples):
    questions = [q["qas"][0]["question"] for q in examples["data"]]
    contexts = [q["paragraphs"][0]["context"] for q in examples["data"]]
    answers = [q["qas"][0]["answers"][0]["text"] for q in examples["data"]]
    start_positions = [q["paragraphs"][0]["context"].find(a) for q, a in zip(examples["data"], answers)]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=50,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = answers[sample_index]
        start_char = start_positions[sample_index]
        end_char = start_char + len(answers)

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            tokenized_examples["start_positions"].append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
# Evaluate the model
eval_results = trainer.evaluate()

print(f"Evaluation Results: {eval_results}")
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
