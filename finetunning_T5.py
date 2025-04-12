import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, DataCollatorForSeq2Seq)
from sklearn.model_selection import train_test_split
import wandb
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Configuration utilisateur ---
MODEL_NAME = "google/flan-t5-small"
OUTPUT_DIR = "./flan-t5-small-rewriting"
CSV_PATH = "data_folder/paired_queries_train.csv"  
LEARNING_RATE = 1e-4
EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 128
PROJECT_NAME = "NLP_ENSAE"
RUN_NAME = "flan-t5-small-3epochs-test2"
SCHEDULER = "linear"
WARMUP_STEPS = 300

# --- Initialisation de W&B ---
wandb.init(
    entity=PROJECT_NAME,
    project="finetunning T5",
    name=RUN_NAME,
    config={
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_seq_length": MAX_LENGTH,
        "scheduler": SCHEDULER,
        "warmup_steps": WARMUP_STEPS
    }
)

# --- Chargement des données ---
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.03, random_state=42)


datasets = DatasetDict({
    "train": Dataset.from_pandas(train_df.rename(columns={"noisy_query": "input", "rewritten": "target"})),
    "validation": Dataset.from_pandas(val_df.rename(columns={"noisy_query": "input", "rewritten": "target"}))
})

# --- Tokenizer et modèle ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# --- Prétraitement ---
def preprocess(example):
    input_text = "Improve the query: " + example["input"]
    target_text = example["target"]
    return tokenizer(input_text, text_target=target_text, truncation=True, max_length=MAX_LENGTH)

tokenized_datasets = datasets.map(preprocess, remove_columns=datasets["train"].column_names)

# --- Collateur de données ---
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- Arguments d'entraînement ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type=SCHEDULER,
    warmup_steps=WARMUP_STEPS,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch",
    report_to="wandb",
    eval_strategy="steps",
    eval_steps=200,
    run_name=RUN_NAME,
    fp16=True
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# --- Entraînement ---
trainer.train()

# --- Sauvegarde finale ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- Fin de session W&B ---
wandb.finish()