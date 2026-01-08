# SciFact Retrieval: BM25 (Sparse) vs Dense (Sentence-Transformers + FAISS)

This project implements and evaluates two classic Information Retrieval (IR) approaches on the **SciFact** dataset (from the **BEIR** benchmark):

- **Sparse retrieval**: BM25 (term-based matching)
- **Dense retrieval**: Sentence-Transformers embeddings + FAISS similarity search

The goal is to compare retrieval quality and practical trade-offs (speed, memory, indexing time).

This project uses PyTorch with CUDA 12.8+ (CU120 support)
for GPU execution.

## Project Structure

```
.
├── datasets/     # SciFact Dataset
├── src/
│   ├── download_data.ipynb        # downloads SciFact from BEIR
│   ├── bm25_retriever.ipynb     # builds BM25 index + retrieves top-k
│   ├── dense_retriever.ipynb      # builds embeddings + FAISS index + retrieves top-k
│   └── evaluate.py             # evaluates run files using BEIR metrics
├── outputs/
│   ├── dense_results.json           # retrieval results for BM25
│   ├── sparse_results.json          # retrieval results for Dense 
│   └── eval_result_*.txt            # saved evaluation metrics
├── requirements.txt
└── README.md
```

> `outputs/` is generated after running the scripts.


## Core Requirements

- Python 3.9+ (recommended)
- `beir` (dataset download + evaluation)
- `rank-bm25` (BM25 implementation)
- `sentence-transformers` (dense embeddings)
- `faiss-cpu`

All dependencies are listed in `requirements.txt`.


## Setup

### 1) Create a virtual environment (conda)

```bash
conda create -n sdis python=3.12
conda activate sdis
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

## Running the Project

### Step A — Download SciFact (BEIR)

Run the ipynb `download_data.ipynb`
This creates `data/scifact/` containing the corpus, queries, and qrels.

### Step B — BM25 retrieval (Sparse)

Run the ipynb `bm25_retriever.ipynb`

### Step C — Dense retrieval (Sentence-Transformers + FAISS)

Run the ipynb `dense_retriever.ipynb`

### Step D — Evaluate both runs

```bash
python src/evaluate.py \
  datasets/scifact \
  outputs/dense_results.json

python src/evaluate.py \
  datasets/scifact \
  outputs/sparse_results.json
```

This prints and saves BEIR-style metrics.

## Results and Discussion
Metric | Sparse (BM25) | Dense (all-MiniLM-L6-v2) | Improvement
----- | ----- | ----- | -----
nDCG@10 | 0.5597 | 0.6451 | +15.3%
Recall@100 | 0.7929 | 0.9250 | +16.7%
MAP@10 | 0.5147 | 0.5959 | +15.8%


### Which retriever performed better?
- The Dense Retriever performs better for this task. It achieved a much higher Recall@100 (92.5%), meaning that for nearly every query, the correct evidence was successfully retrieved within the top 100 results. Its nDCG@10 score of 0.6451 also indicates that it is much more effective at ranking the most relevant documents at the very top of the list compared to BM25.
- I believe in this specific task (scientific facts), dense retriever performs better because the query and the corpus may be using different terminologies. The transformer based model embeds these different vocabularies into similar vector space because they share the same context, which would do a better job than keyword based ranking and matching.
- While Dense Retriever provides much higher quality, it is often slower than sparse retriever. In large production environment, a hybrid approach (sparse + dense) may be used to balance performance and speed.