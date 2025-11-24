import os

# Load all speeches
def load_documents(folder):
    docs = []
    filenames = []
    for file in sorted(os.listdir(folder)):
        if file.startswith("speech_") and file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(f.read())
                filenames.append(file)
    return docs, filenames

documents, filenames = load_documents(".")
print("Loaded", len(documents), "documents.")

# Load queries
queries = []
with open("queries1.txt", "r", encoding="utf-8") as f:
    for line in f:
        queries.append(line.strip())

print("Total queries loaded:", len(queries))

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]

from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_docs)

query = queries[0]   # take first query for example
tokenized_query = word_tokenize(query.lower())

scores = bm25.get_scores(tokenized_query)

# Sort results
ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

print("\nBM25 Ranking for Query:", query)
for file, score in ranked[:10]:
    print(file, ":", score)

from collections import Counter

# Build corpus statistics
corpus = " ".join(documents).lower()
corpus_tokens = word_tokenize(corpus)
corpus_freq = Counter(corpus_tokens)
corpus_len = len(corpus_tokens)

def jm_score(doc_tokens, query_tokens, lamb=0.7):
    doc_freq = Counter(doc_tokens)
    doc_len = len(doc_tokens)
    score = 1.0

    for term in query_tokens:
        p_doc = doc_freq[term] / doc_len if doc_len > 0 else 0
        p_corpus = corpus_freq[term] / corpus_len
        p = lamb * p_doc + (1 - lamb) * p_corpus
        score *= p if p > 0 else 1e-9
    return score
print("\nJelinek-Mercer Ranking for Query:", query)
jm_scores = []

for file, doc_tokens in zip(filenames, tokenized_docs):
    score = jm_score(doc_tokens, tokenized_query)
    jm_scores.append((file, score))

jm_ranked = sorted(jm_scores, key=lambda x: x[1], reverse=True)

for file, score in jm_ranked[:10]:
    print(file, ":", score)
