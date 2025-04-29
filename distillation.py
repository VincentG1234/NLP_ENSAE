import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

ENTITY = "NLP_ENSAE"
PROJECT = "finetunning T5"
ARTIFACT = "NLP_ENSAE/finetunning T5/flan-t5-small-3epochs-gpu-2:latest"
STUDENT = "google/flan-t5-small"
CSV = "data_folder/paired_queries_train.csv"
OUT_DIR = "./distilled-small"
EPOCHS, BS, LR, T = 3, 8, 5e-5, 2.0


def batchify(df, tokenizer, device):
    enc = tokenizer(
        ["Rephrase: " + q for q in df.input],
        text_target=df.target.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    return enc.input_ids, enc.attention_mask, enc.labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name="distill-small-kd",
        job_type="distill",
    )

    teacher_dir = run.use_artifact(ARTIFACT).download()
    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_dir).to(device).eval()
    student = AutoModelForSeq2SeqLM.from_pretrained(STUDENT).to(device).train()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)

    df = pd.read_csv(CSV).rename(
        columns={"noisy_query": "input", "rewritten": "target"}
    )
    optim = torch.optim.AdamW(student.parameters(), lr=LR)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        for i in tqdm(range(0, len(df), BS), desc=f"Epoch {epoch}", unit="batch"):
            batch = df.iloc[i:i + BS]
            in_ids, attn, labels = batchify(batch, tokenizer, device)

            optim.zero_grad()
            with torch.no_grad():
                t_out = teacher(
                    input_ids=in_ids,
                    attention_mask=attn,
                    labels=labels,
                )
                t_logits = t_out.logits / T

            s_out = student(
                input_ids=in_ids,
                attention_mask=attn,
                labels=labels,
            )
            s_logits = s_out.logits / T

            loss = (
                kl_loss(
                    F.log_softmax(s_logits, dim=-1),
                    F.softmax(t_logits, dim=-1),
                )
                * (T * T)
            )
            loss.backward()
            optim.step()

        print(f"Epoch {epoch}/{EPOCHS} took {time.time() - t0:.1f}s loss={loss:.4f}")
        wandb.log({"epoch": epoch, "loss": loss.item()})

    student.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    artifact = wandb.Artifact("distilled-small", type="model")
    artifact.add_dir(OUT_DIR)
    run.log_artifact(artifact)

    print("Saved:", OUT_DIR)
    run.finish()


if __name__ == "__main__":
    main()
