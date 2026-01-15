import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    if not documents:
        return np.empty((0,)), []  

    tokenized_docs = [doc.lower().split() for doc in documents]

    
    all_tokens = [token for doc in tokenized_docs for token in doc]
    if not all_tokens:
        return np.empty((0,)), []

    vocabulary = sorted(set(all_tokens))
    V = len(vocabulary)
    N = len(documents)
    vocab_index = {term: i for i, term in enumerate(vocabulary)}

    
    df = np.zeros(V)
    for tokens in tokenized_docs:
        unique = set(tokens)
        for term in unique:
            df[vocab_index[term]] += 1

    
    
    idf = np.log(N / np.maximum(df, 1))

    tfidf_matrix = np.zeros((N, V))

    for doc_idx, tokens in enumerate(tokenized_docs):
        if not tokens:
            continue  

        counts = Counter(tokens)
        total = len(tokens)
        for term, count in counts.items():
            j = vocab_index[term]
            tf = count / total
            tfidf_matrix[doc_idx, j] = tf * idf[j]

    return tfidf_matrix, vocabulary