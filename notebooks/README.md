# Notebooks

| Notebook | Task | Status |
|---|---|---|
| `01_task1_eda_preprocessing.ipynb` | EDA + Preprocessing | Interim |
| `02_task2_chunking_embeddings.ipynb` | Chunking + Embedding + Vector Store | Interim |

## How to run

Start Jupyter from the **project root**:

```bash
jupyter notebook
```

Then open any notebook from the `notebooks/` folder.
Each notebook adds `..` to `sys.path` automatically so `src/` imports work.

## Output locations

- Figures → `data/processed/figures/`
- Filtered dataset → `data/processed/filtered_complaints.csv`
- Vector store → `vector_store/`