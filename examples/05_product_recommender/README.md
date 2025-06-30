# Example 5: Real-Time Product Recommender (Online Multi-Label)

**Use Case Type**: E-commerce real-time multi-label recommendation

**Input Format**: User clickstream, product graph

**Pipeline Structure**:
- Graph embedding preprocessor
- Hybrid rule + neural net (user affinity DAG)
- Real-time recommender retraining using `online_adapter.py`

**Advanced Features Triggered**:
- `hybrid_induction.py` (symbolic + neural)
- `online_adapter.py` (live retraining)
- `kg_federation.py` (semantic product graph)

**Expected Output**:
- Deployed inference engine for recommendations
- Adaptive personalization model per session
- Evaluation notebook on CTR/lift metrics

---

## Usage

- Pipeline YAML: [`pipeline_dag.yaml`](./pipeline_dag.yaml)
- Walkthrough Notebook: [`notebook.ipynb`](./notebook.ipynb)

**Run via CLI:**

```bash
python c60_executor.py --input examples/05_product_recommender/pipeline_dag.yaml
```

**Or in Python:**

```python
from c60.engine.agent import AutoDAGAgent
agent = AutoDAGAgent()
agent.load_yaml('examples/05_product_recommender/pipeline_dag.yaml')
agent.run()
```
