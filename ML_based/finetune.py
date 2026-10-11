from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from shared import get_dataset

# finetune ModernBert-base on our data
MODEL_PATH = "answerdotai/ModernBERT-base"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset:Dataset = get_dataset(split="train[:10000]")
test_dataset:Dataset = get_dataset(split="test[:50]")

def process(sample):
    return {"label":sample['class_index']-1}

def tokenize(sample):
    return tokenizer(sample['review_text'], return_tensor="pt")

dataset = dataset.map(process)
dataset = dataset.map(tokenize)
test_dataset = test_dataset.map(process)
test_dataset = test_dataset.map(tokenize)
data_collator = DataCollatorWithPadding(tokenizer)

print(dataset, test_dataset)

#args taken from HF's text classification finetuning guide: https://huggingface.co/docs/transformers/v5.17.0/en/tasks/sequence_classification#preprocess
trainer_args = TrainingArguments(
    output_dir="Testing_mk_1",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model,
    trainer_args,
    data_collator,
    train_dataset=dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)

if __name__ == "__main__":
    trainer.train()
    trainer.push_to_hub(private=True)
