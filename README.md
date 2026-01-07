# SciFact Retrieval: BM25 (Sparse) vs Dense (Sentence-Transformers + FAISS)

This project implements and evaluates two classic Information Retrieval (IR) approaches on the **SciFact** dataset (from the **BEIR** benchmark):

- **Sparse retrieval**: BM25 (term-based matching)
- **Dense retrieval**: Sentence-Transformers embeddings + FAISS similarity search

The goal is to compare retrieval quality and practical trade-offs (speed, memory, indexing time).

## Project Structure

```
.
├── scripts/
│   ├── download_data.py        # downloads SciFact from BEIR
│   ├── bm25_retriever.py       # builds BM25 index + retrieves top-k
│   ├── dense_retriever.py      # builds embeddings + FAISS index + retrieves top-k
│   ├── evaluate.py             # evaluates run files using BEIR metrics
│   └── utils.py                # shared helpers (I/O, tokenization, etc.)
├── outputs/
│   ├── bm25_run.json           # retrieval results for BM25
│   ├── dense_run.json          # retrieval results for Dense retriever
│   └── metrics.json            # saved evaluation metrics
├── requirements.txt
└── README.md
```

> `outputs/` is generated after running the pipeline.


## Requirements

- Python 3.9+ (recommended)
- `beir` (dataset download + evaluation)
- `rank-bm25` (BM25 implementation)
- `sentence-transformers` (dense embeddings)
- `faiss-cpu` or `faiss-gpu` (vector similarity search)

All dependencies are listed in `requirements.txt`.


## Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\\Scripts\\activate  # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```


## Running the Project

### Step A — Download SciFact (BEIR)

```bash
python scripts/download_data.py --dataset scifact --out_dir data/
```

This creates `data/scifact/` containing the corpus, queries, and qrels.

### Step B — BM25 retrieval (Sparse)

```bash
python scripts/bm25_retriever.py \
  --data_dir data/scifact \
  --top_k 100 \
  --output outputs/bm25_run.json
```

### Step C — Dense retrieval (Sentence-Transformers + FAISS)

```bash
python scripts/dense_retriever.py \
  --data_dir data/scifact \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --top_k 100 \
  --output outputs/dense_run.json
```

You can swap the model (often higher quality but slower/larger), e.g.:

- `sentence-transformers/all-mpnet-base-v2`

### Step D — Evaluate both runs

```bash
python scripts/evaluate.py \
  --data_dir data/scifact \
  --bm25_run outputs/bm25_run.json \
  --dense_run outputs/dense_run.json \
  --output outputs/metrics.json
```

This prints and saves BEIR-style metrics (commonly `nDCG@k`, `Recall@k`, and optionally `MAP@k`, depending on your evaluator settings).

## Results and Discussion

After you run evaluation, paste your numbers into the table below and summarize what you observed.

| Method | nDCG@10 | Recall@100 | Notes |
|-------:|:-------:|:----------:|------|
| BM25 (Sparse) | _TBD_ | _TBD_ | Fast + strong lexical matching |
| Dense (ST + FAISS) | _TBD_ | _TBD_ | Better semantic matching; more compute/memory |

### Which retriever performed better?

- **Observed winner:** _TBD_

In your explanation, connect performance to the task:
- **Dense can win** when queries and relevant documents use different wording (synonyms/paraphrases), because embeddings capture semantic similarity.
- **BM25 can win** in scientific text when exact terminology and entity overlap are critical, since term matching is very strong and reliable.

### Trade-offs observed (quality vs speed vs memory)

**BM25 (Sparse)**
- ✅ Very fast indexing and retrieval on CPU
- ✅ Lower memory usage than storing dense vectors
- ❌ Limited semantic generalization (misses paraphrases)

**Dense (Sentence-Transformers + FAISS)**
- ✅ Strong semantic retrieval (paraphrases, related phrasing)
- ✅ FAISS enables efficient top-k similarity search
- ❌ One-time embedding cost can be slow (corpus encoding)
- ❌ Higher memory footprint (stores dense vectors)


## Reproducibility Notes

- Use the same `top_k` for both systems when comparing.
- Keep preprocessing consistent (tokenization / text fields) to ensure a fair comparison.
- If reporting speed, include your machine specs (CPU/GPU, RAM) and whether you used `faiss-cpu` or `faiss-gpu`.

