# Example 3: Legal Document Classification (NLP Multiclass)

**Use Case Type**: Multi-class document classification (contracts, filings, case law)

**Input Format**: Raw text (PDF, TXT, DOCX)

**Pipeline Structure**:
- NLP preprocessor: tokenization, embedding (FastText/BERT)
- Dimensionality reducer (UMAP → 2D/5D latent)
- Classification DAG (multi-head attention → logistic output)

**Advanced Features Triggered**:
- `dag_mutator.py` (transformer-specialized heads)
- `distill_compiler.py` (compressed export)
- `gui_pipeline_builder.py` (visual design)

**Expected Output**:
- Notebook with class activation maps
- Exported PyTorch + ONNX model
- Classifier CLI (text in → label out)

---

## Usage

- Pipeline YAML: [`pipeline_dag.yaml`](./pipeline_dag.yaml)
- Walkthrough Notebook: [`notebook.ipynb`](./notebook.ipynb)

**Run via CLI:**

```bash
python c60_executor.py --input examples/03_legal_document_classification/pipeline_dag.yaml
```

**Or in Python:**

```python
from c60.engine.agent import AutoDAGAgent
agent = AutoDAGAgent()
agent.load_yaml('examples/03_legal_document_classification/pipeline_dag.yaml')
agent.run()
```
