# 🧠 NLP ENSAE – Query Rewriting Pipeline

This project provides a full pipeline for rewriting search queries to improve performance in retrieval systems (e.g., search engines, RAG). It includes:

- Synthetic data generation from public datasets (MS MARCO, Natural Questions, etc.)
- LLM-based query rewriting using Groq-hosted APIs or Hugging Face pipelines
- Fine-tuning of Flan-T5 models (with or without LoRA)
- Knowledge distillation from teacher to student models
- Evaluation, logging, and artifact tracking using Weights & Biases (W&B)

---

## 📁 Project Structure

Directory structure:
└── vincentg1234-nlp_ensae.git/
    ├── README.md
    ├── NB_analysis_data.ipynb
    ├── requirements.txt
    ├── data_engineering/
    │   ├── load_data_hf.py
    │   ├── noise_queries.py
    │   └── write_refined_queries_api_call.py
    ├── data_folder/
    │   ├── paired_queries.csv
    │   ├── paired_queries_test.csv
    │   ├── paired_queries_train.csv
    │   ├── raw_queries.csv
    │   └── refined_queries.csv
    ├── fine_tuning/
    │   ├── distillation.py
    │   ├── finetunning_T5.py
    │   └── finetunning_T5_LORA.py
    └── test/
        ├── test_distill.py
        └── test_predictions_distilled.csv



## 🚀 Training Commands

**Standard fine-tuning**

```bash
python finetunning_T5.py
```


**LoRA fine-tuning**
```bash
python finetunning_T5_LORA.py
```

**Distillation**
```bash
python distillation.py
```


## 📊 Dataset Files

- raw_queries.csv – Raw queries from HF datasets
- refined_queries.csv – Rewritten by LLMs
- paired_queries.csv – Noisy + refined aligned
- paired_queries_train.csv, paired_queries_test.csv – Train/test splits
