# test_distilled.py

import wandb
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm.auto import tqdm

# match distillation.py settings
ENTITY = "NLP_ENSAE"
PROJECT = "finetunning T5"
ARTIFACT = f"{ENTITY}/{PROJECT}/distilled-small:latest"

TEST_CSV = "data_folder/paired_queries_test.csv"
OUTPUT_CSV = "test_predictions_distilled.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        entity=ENTITY, project=PROJECT, job_type="test", name="test-distilled"
    )
    artifact_dir = run.use_artifact(ARTIFACT).download()
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(artifact_dir).to(device).eval()

    df = pd.read_csv(TEST_CSV)
    preds = []
    for q in tqdm(df["noisy_query"], desc="gen"):
        batch = tokenizer(
            f"Rephrase: {q}", return_tensors="pt", truncation=True, padding=True
        ).to(device)
        out = model.generate(**batch, max_new_tokens=64, repetition_penalty=2.5)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))

    df["predicted"] = preds
    df.to_csv(OUTPUT_CSV, index=False)

    art = wandb.Artifact("test-predictions", type="predictions")
    art.add_file(OUTPUT_CSV)
    run.log_artifact(art)
    run.finish()


if __name__ == "__main__":
    main()
