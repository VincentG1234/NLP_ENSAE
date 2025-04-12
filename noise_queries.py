import pandas as pd
import numpy as np
import random
import string

# --- Bruitage clavier QWERTY simplifié ---
keyboard_neighbors = {
    'a': 'qs', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr', 'f': 'dg', 'g': 'fh', 'h': 'gj',
    'i': 'uo', 'j': 'hk', 'k': 'jl', 'l': 'k;', 'm': 'nj', 'n': 'bm', 'o': 'ip', 'p': 'o[',
    'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry', 'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc',
    'y': 'tu', 'z': 'xs', ' ': ' '
}

def noisy_version(query, p_geom=0.55):
    if random.random() < 1/3:
        return query  # pas de bruit

    query = list(query)
    L = len(query)

    # Lois géométriques (commencent à 0)
    X1 = np.random.geometric(p_geom + 0.1) - 1  # swap
    X2 = np.random.geometric(p_geom) - 1  # delete
    X3 = np.random.geometric(p_geom - 0.1) - 1  # replace

    # --- 1. SWAP ---
    for _ in range(X1):
        if L < 2: break
        i = random.randint(0, L - 2)
        query[i], query[i + 1] = query[i + 1], query[i]

    # --- 2. DELETE ---
    for _ in range(X2):
        if len(query) == 0: break
        i = random.randint(0, len(query) - 1)
        del query[i]

    # --- 3. REPLACE ---
    for _ in range(X3):
        if len(query) == 0: break
        i = random.randint(0, len(query) - 1)
        char = query[i].lower()
        if char in keyboard_neighbors:
            replacement = random.choice(keyboard_neighbors[char])
            query[i] = replacement
        else:
            query[i] = random.choice(string.ascii_lowercase)

    return ''.join(query)

# --- Application sur DataFrame ---
def add_noisy_column(df, query_col="query"):
    df = df.copy()
    df["noisy_query"] = df[query_col].apply(noisy_version)
    return df

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    refined_queries_df = pd.read_csv("refined_queries.csv")
    df = pd.read_csv("raw_queries.csv")
    paired_queries_df = pd.merge(refined_queries_df, df, on="query", how="left")
    df_noisy = add_noisy_column(paired_queries_df)
    df_noisy.to_csv("paired_queries.csv", index=False)
