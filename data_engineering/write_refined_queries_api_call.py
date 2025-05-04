import csv
import os
import time
import random

import pandas as pd
from openai import OpenAI


df = pd.read_csv("raw_queries.csv")

# Chemin vers le fichier persistant
save_path = "refined_queries.csv"

# Initialisation si le fichier n'existe pas encore
if not os.path.exists(save_path):
    with open(save_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "rewritten"])  # En-têtes

# Initialise le client Groq (à faire une seule fois)
client = OpenAI(
    api_key="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",
    base_url="https://api.groq.com/openai/v1"
)

# Prompt template avec {query}
PROMPT_TEMPLATE = """Rewrite this search query to improve retrieval:
- Fix grammar or spelling
- Rephrase clearly
- Add synonyms to informative words (in parentheses)

return only the query. no explanation.
Example:
Input: danger screen sleep
Output: What are the risks (hazards, dangers) of using screens (monitors, displays) before sleep (slumber, bedtime)?

Input: {query}
"""


def rewrite_query_llm(query: str, temperature: float = 0.5) -> str:
    prompt = PROMPT_TEMPLATE.format(query=query)

    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=128
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


queries = df.sample(10000, random_state=123)['query'].values[7563:]

for query in queries:
    # Sleep entre 5.5 et 8.5 secondes (autour de 35s ± 5s)
    sleep_duration = random.uniform(30, 40)
    time.sleep(sleep_duration)

    rewritten = rewrite_query_llm(query)  # ta fonction de rewriting
    # Append sécurisé
    with open(save_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow([query, rewritten])
