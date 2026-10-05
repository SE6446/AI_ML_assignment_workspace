from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from shared import get_dataset

# finetune ModernBert-base on our data
MODEL_PATH = "answerdotai/ModernBERT-base"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

dataset = get_dataset()

def process(sample):
    return {"token_ids": tokenizer(sample['review_text']), "label":sample['class_index']-1}

data_collator = DataCollatorWithPadding(tokenizer)

#args taken from HF's text classification finetuning guide: https://huggingface.co/docs/transformers/v5.17.0/en/tasks/sequence_classification#preprocess
trainer_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model,
    trainer_args,
    data_collator,
    train_dataset = dataset['train']
    eval_dataset = dataset['eval']
    processing_class=tokenizer
)

if __name__ == "__main__":
    trainer.train()
    trainer.push_to_hub()