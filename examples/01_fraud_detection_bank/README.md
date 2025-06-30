# Example 1: Credit Card Fraud Detection (Imbalanced Binary Class)

**Use Case Type**: Financial anomaly detection

**Input Format**: Transaction logs (tabular)

**Pipeline Structure**:
- DAG starts with heavy preprocessing (imbalanced handling: SMOTE/under-sampling)
- Feature engineering via auto-binarizers and PCA + TSNE
- Classifier node (XGBoost, LightGBM, or dynamic SVM) evolves based on F1-score

**Advanced Features Triggered**:
- `hybrid_induction.py` (interpretable rules for fraud)
- `audit.py` (for traceable prediction explanations)
- `auto_weights_mutator.py` (for class-weight tuning)

**Expected Output**:
- Trained pipeline DAG
- Audit logs per flagged transaction
- Jupyter notebook with visual explanations

---

## Usage

- Pipeline YAML: [`pipeline_dag.yaml`](./pipeline_dag.yaml)
- Walkthrough Notebook: [`notebook.ipynb`](./notebook.ipynb)

**Run via CLI:**

```bash
python c60_executor.py --input examples/01_fraud_detection_bank/pipeline_dag.yaml
```

**Or in Python:**

```python
from c60.engine.agent import AutoDAGAgent
agent = AutoDAGAgent()
agent.load_yaml('examples/01_fraud_detection_bank/pipeline_dag.yaml')
agent.run()
```
