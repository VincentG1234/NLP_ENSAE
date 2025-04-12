from datasets import load_dataset
import pandas as pd
import random
from tqdm import tqdm

def sample_with_progress(data, n):
    return list(tqdm(random.sample(data, min(n, len(data))), desc="Sampling"))

def load_ms_marco(sample_size=20000):
    print("Loading microsoft/ms_marco...")
    ds = load_dataset("microsoft/ms_marco", "v1.1", split="train")["query"]
    queries = [q for q in tqdm(ds, desc="MS MARCO")]
    queries = list(set(q.strip() for q in queries if len(q.strip()) > 5))
    sampled = sample_with_progress(queries, sample_size)
    return pd.DataFrame({"query": sampled, "source": "ms_marco"})

def load_natural_questions(sample_size=20000):
    print("Loading sentence-transformers/natural-questions...")
    ds = load_dataset("sentence-transformers/natural-questions", split="train")["query"]
    queries = [q for q in tqdm(ds, desc="Natural Questions")]
    queries = list(set(q.strip() for q in queries if len(q.strip()) > 5))
    sampled = sample_with_progress(queries, sample_size)
    return pd.DataFrame({"query": sampled, "source": "natural_questions"})

def load_quora(sample_size=20000):
    print("Loading toughdata/quora-question-answer-dataset...")
    ds = load_dataset("toughdata/quora-question-answer-dataset", split="train")["question"]
    queries = [q for q in tqdm(ds, desc="Quora")]
    queries = list(set(q.strip() for q in queries if len(q.strip()) > 5))
    sampled = sample_with_progress(queries, sample_size)
    return pd.DataFrame({"query": sampled, "source": "quora"})

def load_rag_dataset(sample_size=20000):
    print("Loading neural-bridge/rag-dataset-12000...")
    ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")["question"]
    queries = [q for q in tqdm(ds, desc="RAG Dataset") if q and isinstance(q, str)]
    queries = list(set(q.strip() for q in queries if len(q.strip()) > 5))
    sampled = sample_with_progress(queries, sample_size)
    return pd.DataFrame({"query": sampled, "source": "rag_12000"})

def load_yahoo_answers(sample_size=20000):
    print("Loading sentence-transformers/yahoo-answers...")
    ds = load_dataset("sentence-transformers/yahoo-answers", 'question-answer-pair', split="train")["question"]
    questions = [q for q in tqdm(ds, desc="Yahoo Answers")]
    questions = list(set(q.strip() for q in questions if len(q.strip()) > 5))
    sampled = sample_with_progress(questions, sample_size)
    return pd.DataFrame({"query": sampled, "source": "yahoo_answers"})

def clean_queries(df):
    df["query"] = df["query"].str.strip()
    df = df[df["query"].str.len() > 5]
    df = df.drop_duplicates(subset=["query"])
    return df.reset_index(drop=True)

def merge_and_save(dfs, output_file="raw_queries.csv"):
    print("Merging datasets...")
    merged = pd.concat(dfs, ignore_index=True)
    merged = clean_queries(merged)
    print(f"✅ Total queries after cleaning: {len(merged)}")
    merged.to_csv(output_file, index=False)
    print(f"📁 Saved to: {output_file}")
    return merged

if __name__ == "__main__":
    random.seed(42)
    dfs = [
        load_ms_marco(),
        load_natural_questions(),
        load_quora(),
        load_rag_dataset(),
        load_yahoo_answers()
    ]
    merged_df = merge_and_save(dfs)
